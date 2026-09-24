"""Check exact loss identities, risk-only reconstruction and unchanged-parent gradients."""

from itertools import combinations
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import torch

from resnet_release import CifarResNet, BasicBlock
from permission_model import (SemanticPermissionModel, balanced_query_basis,
                              expected_permission_brier, permission_probabilities,
                              recover_class_probabilities)


def state_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(list(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main(directory):
    started = time.perf_counter()
    settings = json.loads((directory / "config.json").read_text())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(37737)
    checks = []
    for count in (2, 4, 6, 8):
        masks = torch.zeros((len(list(combinations(range(count), count//2))), count), dtype=torch.float64)
        for index, subset in enumerate(combinations(range(count), count//2)):
            masks[index, list(subset)] = 1
        logits = torch.randn((3, count), dtype=torch.float64, requires_grad=True)
        probabilities = logits.softmax(-1)
        targets = torch.tensor([0, count//2, count-1])
        direct = ((permission_probabilities(probabilities, masks)-masks[:, targets].T)**2).mean(-1)
        analytic = expected_permission_brier(probabilities, targets)
        value_error = float((direct-analytic).abs().max())
        direct_gradient = torch.autograd.grad(direct.mean(), logits, retain_graph=True)[0]
        analytic_gradient = torch.autograd.grad(analytic.mean(), logits)[0]
        gradient_error = float((direct_gradient-analytic_gradient).abs().max())
        assert value_error < 1e-12 and gradient_error < 1e-12
        direction = torch.randn_like(logits)
        step = 1e-7
        plus = expected_permission_brier((logits.detach()+step*direction).softmax(-1), targets).mean()
        minus = expected_permission_brier((logits.detach()-step*direction).softmax(-1), targets).mean()
        difference_error = abs(float((plus-minus)/(2*step)-(analytic_gradient*direction).sum()))
        assert difference_error < 1e-8
        recovered = recover_class_probabilities(permission_probabilities(probabilities.detach(), balanced_query_basis(count, torch.float64)))
        reconstruction_error = float((recovered-probabilities.detach()).abs().max())
        assert reconstruction_error < 1e-12
        checks.append({"classes": count, "value_error": value_error, "gradient_error": gradient_error,
                       "finite_difference_error": difference_error, "reconstruction_error": reconstruction_error})
    for invalid_count in (1, 3, 7):
        try:
            balanced_query_basis(invalid_count)
        except ValueError:
            pass
        else:
            raise AssertionError("odd class count accepted")
    try:
        permission_probabilities(torch.full((1, 4), .25), torch.ones((1, 4)))
    except ValueError:
        pass
    else:
        raise AssertionError("unbalanced mask accepted")
    parent_root = directory.parent
    checkpoint = parent_root / "acquisition01/cifar100_resnet56.pt"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == "f2eff4c8461ca1e0d39af83a65f7243bf7d29ec421efee3064bcee93a3caaa73"
    parent = CifarResNet(BasicBlock, [9, 9, 9], num_classes=100).eval()
    parent.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    data = parent_root / "data_validation01"
    receipt = json.loads((data / "results.json").read_text())
    for filename, field in [("selected_images.npy", "images_sha256"), ("evaluation_labels.npz", "evaluation_labels_sha256")]:
        assert hashlib.sha256((data / filename).read_bytes()).hexdigest() == receipt[field]
    labels = np.load(data / "evaluation_labels.npz", allow_pickle=False)
    model = SemanticPermissionModel(parent, torch.tensor(labels["mapping"])).eval()
    parent_before = state_digest(parent)
    before = state_digest(model)
    images = torch.from_numpy(np.load(data / "selected_images.npy", allow_pickle=False)[:8].copy()).float()/255
    images = (images-torch.tensor(settings["mean"])[None, :, None, None])/torch.tensor(settings["standard_deviation"])[None, :, None, None]
    targets = torch.tensor(labels["fine"][:8].copy(), dtype=torch.long)
    with torch.no_grad():
        fine, coarse = model.distributions(images)
        fine_risk, coarse_risk = model(images, balanced_query_basis(100), balanced_query_basis(20))
        fine_recovered, coarse_recovered = recover_class_probabilities(fine_risk), recover_class_probabilities(coarse_risk)
        fine_error = float((fine-fine_recovered).abs().max())
        coarse_error = float((coarse-coarse_recovered).abs().max())
        assert max(fine_error, coarse_error) < 2e-6
        assert torch.equal(fine.argmax(-1), fine_recovered.argmax(-1))
        assert torch.equal(coarse.argmax(-1), coarse_recovered.argmax(-1))
    loss = model.expected_loss(images, targets).mean()
    loss.backward()
    gradients = {"first_convolution": float(parent.conv1.weight.grad.norm()),
                 "final_classifier": float(parent.fc.weight.grad.norm())}
    assert all(np.isfinite(value) and value > 0 for value in gradients.values())
    assert all(value.grad is None or torch.isfinite(value.grad).all() for value in model.parameters())
    assert state_digest(parent) == parent_before == "f682a4d5419ca404b8461551d1f1828e644bab04049ffb654b05b2b915ee9abd"
    assert state_digest(model) == before
    result = {"status": "complete", "component_checks": checks, "integrated_loss": float(loss.detach()),
              "fine_probability_reconstruction_error": fine_error, "coarse_probability_reconstruction_error": coarse_error,
              "gradient_norms": gradients, "parent_sha256": parent_before, "integrated_sha256": before,
              "all_parameters_unchanged": True, "optimizer_steps": 0, "neural_training": False,
              "wall_seconds": time.perf_counter()-started}
    (directory / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
