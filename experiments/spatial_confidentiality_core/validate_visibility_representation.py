"""Independent integration, derivative and initialization checks without training."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import time
import numpy as np
import torch
from visibility_representation import (
    VisibilityRepresentation, continuous_brier_loss, initial_probabilities,
)


def integrated_reference(probabilities, target, support):
    boundaries = sorted(set([0.0, 1.0, float(target), *support.tolist()]))
    total = 0.0
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        threshold = (left+right)/2
        prediction = probabilities[support > threshold].sum()
        total += (right-left)*(prediction-float(target > threshold))**2
    return total


def state_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().numpy().tobytes())
    return digest.hexdigest()


def main(directory, root):
    started = time.perf_counter()
    configuration = json.loads((directory/'config.json').read_text())
    torch.set_num_threads(1)
    torch.manual_seed(configuration['seed'])
    generator = np.random.default_rng(configuration['seed'])
    maximum_integration_error = 0.0
    maximum_pairwise_error = 0.0
    incorrect_sign_rejections = 0
    checks = 0
    minimum_mean_slack = float('inf')
    for count in [2, 3, 9, 65]:
        support = torch.linspace(0, 1, count, dtype=torch.float64)
        probabilities = torch.randn(32, count, dtype=torch.float64).softmax(-1)
        targets = torch.rand(32, dtype=torch.float64)
        targets[:2] = torch.tensor([0., 1.])
        loss = continuous_brier_loss(probabilities, targets, support)
        distances = (support[:, None]-support[None, :]).abs()
        pairwise = (probabilities*(support-targets[:, None]).abs()).sum(-1)
        dispersion = torch.einsum('bi,ij,bj->b', probabilities, distances, probabilities)/2
        maximum_pairwise_error = max(maximum_pairwise_error, float((loss-(pairwise-dispersion)).abs().max()))
        for row in range(len(targets)):
            reference = integrated_reference(probabilities[row].numpy(), float(targets[row]), support.numpy())
            maximum_integration_error = max(maximum_integration_error, abs(float(loss[row])-reference))
            incorrect_sign_rejections += abs(float(pairwise[row]+dispersion[row])-reference) > 1e-8
            checks += 1
        survival = 1-probabilities.cumsum(-1)
        assert (survival.diff(dim=-1) <= 1e-14).all()
        means = (probabilities*support).sum(-1)
        minimum_mean_slack = min(minimum_mean_slack, float((loss-(means-targets)**2).min()))
    assert maximum_integration_error < 1e-12
    assert maximum_pairwise_error < 1e-12
    assert minimum_mean_slack >= -1e-12
    assert incorrect_sign_rejections > 0

    support = torch.linspace(0, 1, 9, dtype=torch.float64)
    means = torch.tensor([.173, .469, .827], dtype=torch.float64, requires_grad=True)
    residual = torch.randn(3, 9, dtype=torch.float64, requires_grad=True)
    targets = torch.tensor([.231, .515, .734], dtype=torch.float64)

    def objective(values, logits):
        probabilities = (initial_probabilities(values, 9).log()+logits).softmax(-1)
        return continuous_brier_loss(probabilities, targets, support)

    derivative_check = torch.autograd.gradcheck(objective, (means, residual), eps=1e-6, atol=1e-5, rtol=1e-4)
    module = VisibilityRepresentation().double()
    before_digest = state_digest(module)
    features = torch.randn(2, 32, 12, 16, dtype=torch.float64, requires_grad=True)
    parent_means = (.1+.8*torch.rand(2, 12, 16, dtype=torch.float64)).requires_grad_()
    probabilities, recovered_means = module(features, parent_means)
    target = torch.rand_like(parent_means)
    loss = continuous_brier_loss(probabilities, target, module.support).mean()
    loss.backward()
    gradient_norms = {
        'parent_means': float(parent_means.grad.norm()),
        'projection_weight': float(module.residual_projection.weight.grad.norm()),
        'projection_bias': float(module.residual_projection.bias.grad.norm()),
        'features': float(features.grad.norm()),
    }
    for name in ['parent_means', 'projection_weight', 'projection_bias']:
        assert np.isfinite(gradient_norms[name]) and gradient_norms[name] > 0
    # The zero residual projection intentionally starts with zero direct feature gradient.
    # The original depth pathway still carries a nonzero derivative into its parent.
    assert gradient_norms['features'] == 0
    assert before_digest == state_digest(module)

    maximum_initialization_error = 0.0
    maximum_formula_error = 0.0
    sample_count = 0
    for index in range(32):
        values = np.load(root/f'inference{index//4+1:02d}'/f'prediction_{index:03d}.npy', allow_pickle=False).ravel()
        selected = generator.choice(len(values), size=configuration['samples_per_image'], replace=False)
        values = np.clip(values[selected].astype(np.float64), .1, 10)/10
        if index == 0:
            values = np.concatenate([values, [0, 1e-12, .5, 1-1e-12, 1]])
        values = torch.tensor(values, dtype=torch.float64)
        initial = initial_probabilities(values)
        restored = (initial*module.support).sum(-1)
        formula = (1-module.uniform_fraction)*values + module.uniform_fraction/2
        maximum_initialization_error = max(maximum_initialization_error, float((restored-values).abs().max()))
        maximum_formula_error = max(maximum_formula_error, float((restored-formula).abs().max()))
        assert (initial > 0).all()
        assert float((initial.sum(-1)-1).abs().max()) < 1e-12
        sample_count += len(values)
    assert maximum_initialization_error <= configuration['uniform_fraction']/2 + 1e-12
    assert maximum_formula_error < 1e-12
    summary = {
        'status': 'complete', 'integration_cases': checks,
        'maximum_integration_error': maximum_integration_error,
        'maximum_pairwise_error': maximum_pairwise_error,
        'minimum_mean_loss_slack': minimum_mean_slack,
        'incorrect_sign_rejections': int(incorrect_sign_rejections),
        'derivative_check': derivative_check, 'gradient_norms': gradient_norms,
        'initialization_samples': sample_count,
        'maximum_initialization_error_normalized': maximum_initialization_error,
        'maximum_initialization_error_metres': 10*maximum_initialization_error,
        'maximum_initialization_formula_error': maximum_formula_error,
        'added_parameters': sum(parameter.numel() for parameter in module.parameters()),
        'state_sha256_before': before_digest, 'state_sha256_after': state_digest(module),
        'optimizer_steps': 0, 'original_model_loaded': False,
        'neural_training': False, 'destructive_mechanism_established': False,
        'versions': {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__},
        'wall_seconds': time.perf_counter()-started,
    }
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
