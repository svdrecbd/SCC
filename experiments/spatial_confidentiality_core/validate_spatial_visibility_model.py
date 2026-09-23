"""Validate the integrated pretrained model without changing its parameters."""
from pathlib import Path
import inspect
import json
import shutil
import sys
import time
import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch
from transformers import AutoImageProcessor, DepthAnythingConfig, DepthAnythingForDepthEstimation
from data import read_frame
from evaluate_model import state_digest
from spatial_visibility_model import SpatialVisibilityModel
from visibility_representation import continuous_brier_loss


def main(directory, root):
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(32438)
    selected = json.loads((directory/'selection.json').read_text())
    model_directory = root/'acquisition01'/'model'
    configuration = DepthAnythingConfig.from_pretrained(str(model_directory), local_files_only=True)
    parent = DepthAnythingForDepthEstimation(configuration).float().cpu().eval()
    parent.load_state_dict(load_file(str(model_directory/'model.safetensors'), device='cpu'), strict=True)
    processor = AutoImageProcessor.from_pretrained(str(model_directory), local_files_only=True, use_fast=False)
    for instance, name in [(parent, 'runtime_model.py'), (processor, 'runtime_processor.py')]:
        shutil.copyfile(inspect.getfile(type(instance)), directory/name)
    module = SpatialVisibilityModel(parent).eval()
    original_digest = state_digest(parent)
    before_digest = state_digest(module)
    image, truth, input_digest = read_frame(root/'acquisition01', selected[0])
    inputs = processor(images=Image.fromarray(image), return_tensors='pt')
    with torch.inference_mode():
        result = module(**inputs)
        support = module.visibility.support
        probabilities = result['probabilities']
        means = result['depth_metres']/10
        initial = result['parent_depth_metres'].clamp(.1, 10)/10
        expected = (1-module.visibility.uniform_fraction)*initial + module.visibility.uniform_fraction/2
        maximum_formula_error = float((means-expected).abs().max())
        assert maximum_formula_error < 5e-7
        normalization_error = float((probabilities.sum(-1)-1).abs().max())
        roundoff_bound = len(support)*torch.finfo(probabilities.dtype).eps
        roundoff_bound /= 1-roundoff_bound
        print(json.dumps({'normalization_error': normalization_error, 'roundoff_bound': roundoff_bound}), flush=True)
        assert normalization_error < roundoff_bound
        prior_risk = torch.ones_like(means)
        maximum_risk_error = 0.0
        cumulative = probabilities.cumsum(-1)
        for threshold in [.03, .127, .333, .61, .9, 1.0]:
            risk = probabilities[..., support > threshold].sum(-1)
            index = int((support <= threshold).sum())-1
            alternate = 1-cumulative[..., index]
            maximum_risk_error = max(maximum_risk_error, float((risk-alternate).abs().max()))
            assert (risk <= prior_risk+roundoff_bound).all()
            prior_risk = risk
        print(json.dumps({'maximum_risk_error': maximum_risk_error}), flush=True)
        assert maximum_risk_error < roundoff_bound
        target = torch.tensor(truth.astype(np.float32)/10)[None]
        losses = continuous_brier_loss(probabilities, target, support)
        mask = torch.zeros_like(target, dtype=torch.bool)
        mask[:, 45:471, 41:601] = True
        mask &= (target > .01) & (target < 1) & torch.isfinite(target)
        initial_brier = float(losses[mask].mean())
        initial_mse = float(((10*means-10*target)**2)[mask].mean())
        reference = np.load(root/'inference01'/'prediction_000.npy', allow_pickle=False)
        parent_error = float(np.max(np.abs(result['parent_depth_metres'][0].numpy()-reference)))
        assert parent_error < 1e-6
        np.save(directory/'initial_depth_metres.npy', result['depth_metres'][0].numpy(), allow_pickle=False)
    assert original_digest == state_digest(parent)
    assert before_digest == state_digest(module)
    summary = {
        'status': 'complete', 'selection_index': 0, 'input_sha256': input_digest,
        'parent_state_sha256_before': original_digest, 'parent_state_sha256_after': state_digest(parent),
        'integrated_state_sha256_before': before_digest, 'integrated_state_sha256_after': state_digest(module),
        'parent_parameters': sum(value.numel() for value in parent.parameters()),
        'total_parameters': sum(value.numel() for value in module.parameters()),
        'maximum_parent_reproduction_error_metres': parent_error,
        'maximum_initialization_formula_error_normalized': maximum_formula_error,
        'maximum_normalization_error': normalization_error, 'fp32_sum_tolerance': roundoff_bound,
        'maximum_risk_error': maximum_risk_error,
        'initial_native_brier_single_scene': initial_brier,
        'initial_depth_mse_single_scene_metres_squared': initial_mse,
        'valid_pixels': int(mask.sum()), 'neural_training': False, 'optimizer_steps': 0,
        'wall_seconds': time.perf_counter()-started,
    }
    (directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
