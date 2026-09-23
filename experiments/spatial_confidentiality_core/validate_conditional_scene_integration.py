"""Validate exact history-conditioned risk gradients through the unchanged parent."""

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
from conditional_scene_distribution import conditional_scene, disclosure_path, marginal_bin_log_probabilities
from data import read_frame
from evaluate_model import state_digest
from joint_scene_mixture import JointSceneMixture
from spatial_program_assessment import SpatialProgramAssessmentModel


def main(directory, root):
    started = time.perf_counter()
    configuration = json.loads((directory/'config.json').read_text())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(configuration['seed'])
    selected = json.loads((directory/'selection.json').read_text())
    model_directory = root/'acquisition01'/'model'
    parent = DepthAnythingForDepthEstimation(DepthAnythingConfig.from_pretrained(str(model_directory), local_files_only=True)).float().cpu().eval()
    parent.load_state_dict(load_file(str(model_directory/'model.safetensors'), device='cpu'), strict=True)
    processor = AutoImageProcessor.from_pretrained(str(model_directory), local_files_only=True, use_fast=False)
    for instance, name in ((parent, 'runtime_model.py'), (processor, 'runtime_processor.py')):
        shutil.copyfile(inspect.getfile(type(instance)), directory/name)
    module = SpatialProgramAssessmentModel(parent, seed=configuration['seed'], mixture_components=4).eval()
    parent_before = state_digest(parent)
    integrated_before = state_digest(module)
    image, truth, input_digest = read_frame(root/'acquisition01', selected[0])
    coordinates = [[row, column] for row in configuration['rows'] for column in configuration['columns']]
    inputs = processor(images=Image.fromarray(image), return_tensors='pt')
    forward_started = time.perf_counter()
    result = module(**inputs, ray_coordinates=coordinates)
    forward_seconds = time.perf_counter()-forward_started
    unique_coordinates = result['unique_coordinates'].tolist()
    measured = torch.tensor([truth[row, column] for row, column in unique_coordinates], dtype=torch.float64).log()
    reference = np.load(root/'inference01'/'prediction_000.npy', allow_pickle=False)
    expected = torch.tensor([reference[row,column] for row,column in unique_coordinates])
    parent_error = float((result['parent_depth_metres']-expected).abs().max().detach())
    assert parent_error <= 1e-6
    initial = result['distribution']
    distribution = JointSceneMixture(initial.logits.double(), initial.locations.double(), initial.factors.double(), initial.diagonal.double())
    observed = list(range(configuration['observed_rays']))
    targets = list(range(len(observed), len(unique_coordinates)))
    calculation_started = time.perf_counter()
    conditional, observation_density = conditional_scene(distribution, observed, measured[observed], targets)
    boundaries = torch.linspace(np.log(configuration['minimum_boundary_metres']), np.log(configuration['maximum_boundary_metres']), configuration['finite_boundaries'], dtype=torch.float64)
    log_probabilities = marginal_bin_log_probabilities(conditional, boundaries)
    outcomes = torch.bucketize(measured[targets], boundaries, right=False)
    paths = disclosure_path(log_probabilities, outcomes)
    selected_log_mass = log_probabilities[torch.arange(len(targets)), outcomes]
    error = float((paths['selected_log_probabilities'].sum(-1)-selected_log_mass).abs().max().detach())
    assert error < configuration['path_tolerance']
    assert torch.logsumexp(log_probabilities, -1).abs().max() < 1e-11
    objective = -paths['selected_log_probabilities'].sum(-1).mean()
    conditional_seconds = time.perf_counter()-calculation_started
    backward_started = time.perf_counter()
    objective.backward()
    backward_seconds = time.perf_counter()-backward_started
    gradients = {'parent_final_weight':float(parent.head.conv3.weight.grad.norm())}
    gradients.update({name:float(parameter.grad.norm()) for name, parameter in module.scene_distribution.named_parameters()})
    assert all(np.isfinite(value) and value>0 for value in gradients.values()), gradients
    assert state_digest(parent)==parent_before and state_digest(module)==integrated_before
    np.savez(directory/'conditional_predictions.npz', coordinates=np.array(unique_coordinates), observed_indices=observed,
             target_indices=targets, observed_log_depths=measured[observed].numpy(), target_log_depths=measured[targets].numpy(),
             boundaries=boundaries.numpy(), log_probabilities=log_probabilities.detach().numpy(),
             branch_log_probabilities=paths['selected_log_probabilities'].detach().numpy(),
             risk_probabilities=paths['upper_probabilities'].detach().numpy(), risk_outcomes=paths['upper_outcomes'].numpy())
    summary = {'status':'complete', 'input_sha256':input_digest, 'observed_rays':len(observed), 'target_rays':len(targets),
               'parent_state_sha256_before_and_after':parent_before, 'integrated_state_sha256_before_and_after':integrated_before,
               'parent_reproduction_error_metres':parent_error, 'maximum_path_log_identity_error':error,
               'mean_quantized_negative_log_likelihood_nats':float(objective.detach()),
               'observed_history_log_density':float(observation_density.detach()),
               'posterior_component_weights':conditional.logits.softmax(-1).detach().tolist(),
               'gradient_norms':gradients, 'forward_seconds':forward_seconds, 'conditional_seconds':conditional_seconds,
               'backward_seconds':backward_seconds, 'neural_training':False, 'optimizer_steps':0,
               'qualification':'implementation validation on a reused scene; no public comparison',
               'wall_seconds':time.perf_counter()-started}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
