"""Restore a pinned classifier and perform inference without reading labels."""

from pathlib import Path
import hashlib
import json
import platform
import sys
import time
import numpy as np
import torch
from resnet_release import CifarResNet, BasicBlock


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
    torch.manual_seed(settings["selection_seed"])
    def input_access_check(event, arguments):
        if event == "open" and isinstance(arguments[0], str) and Path(arguments[0]).name == "evaluation_labels.npz":
            raise PermissionError("Model process cannot read evaluation labels")
    sys.addaudithook(input_access_check)
    try:
        (directory.parent / "data_validation01/evaluation_labels.npz").read_bytes()
    except PermissionError:
        pass
    else:
        raise AssertionError("label access was not rejected")
    path = directory.parent / "acquisition01/cifar100_resnet56.pt"
    checkpoint_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert checkpoint_digest.startswith("f2eff4c8")
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(state, dict) and all(isinstance(value, torch.Tensor) for value in state.values())
    model = CifarResNet(BasicBlock, [9, 9, 9], num_classes=100).eval()
    model.load_state_dict(state, strict=True)
    before = state_digest(model)
    image_path = directory.parent / "data_validation01/selected_images.npy"
    assert hashlib.sha256(image_path.read_bytes()).hexdigest() == settings["images_sha256"]
    images = np.load(image_path, allow_pickle=False)[settings["start_index"]:settings["end_index"]]
    mean = torch.tensor(settings["mean"])[None, :, None, None]
    deviation = torch.tensor(settings["standard_deviation"])[None, :, None, None]
    predictions = []
    permutation_error = None
    inference_started = time.perf_counter()
    with torch.inference_mode():
        for index in range(0, len(images), settings["batch_size"]):
            values = torch.from_numpy(images[index:index+settings["batch_size"]].copy()).float()/255
            logits = model((values-mean)/deviation)
            assert torch.isfinite(logits).all()
            if index == 0:
                reversed_logits = model(((values-mean)/deviation).flip(0)).flip(0)
                permutation_error = float((logits-reversed_logits).abs().max())
                assert permutation_error <= 1e-5
            predictions.append(logits.numpy().copy())
    inference_seconds = time.perf_counter()-inference_started
    assert before == state_digest(model)
    logits = np.concatenate(predictions)
    np.save(directory / "logits.npy", logits, allow_pickle=False)
    result = {"status": "complete", "checkpoint_sha256": checkpoint_digest,
              "state_before": before, "state_after": state_digest(model),
              "tensor_count": len(state), "parameter_count": sum(value.numel() for value in model.parameters()),
              "tensor_shapes": {name: list(value.shape) for name, value in state.items()},
              "start_index": settings["start_index"], "end_index": settings["end_index"],
              "input_images_sha256": settings["images_sha256"],
              "logits_sha256": hashlib.sha256((directory / "logits.npy").read_bytes()).hexdigest(),
              "inference_seconds": inference_seconds, "wall_seconds": time.perf_counter()-started,
              "python_version": platform.python_version(), "torch_version": torch.__version__,
              "neural_training": False, "optimizer_steps": 0, "labels_read_by_model_process": False}
    result["input_permutation_error"] = permutation_error
    result["label_access_control_rejected"] = True
    (directory / "results.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value for key, value in result.items() if key != "tensor_shapes"}))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
