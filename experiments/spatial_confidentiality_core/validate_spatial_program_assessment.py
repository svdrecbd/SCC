"""Verify full pretrained joint assessment and one backward pass without updates."""

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
from joint_scene_assessment import evaluate_program, sample_joint_scene, estimate_brier_gradient
from spatial_program_assessment import SpatialProgramAssessmentModel, remap_program_rays


def main(directory, root):
    started = time.perf_counter()
    configuration = json.loads((directory / "config.json").read_text())
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(configuration["seed"])
    selected = json.loads((directory / "selection.json").read_text())
    model_directory = root / "acquisition01" / "model"
    parent_configuration = DepthAnythingConfig.from_pretrained(str(model_directory), local_files_only=True)
    parent = DepthAnythingForDepthEstimation(parent_configuration).float().cpu().eval()
    parent.load_state_dict(load_file(str(model_directory / "model.safetensors"), device="cpu"), strict=True)
    processor = AutoImageProcessor.from_pretrained(str(model_directory), local_files_only=True, use_fast=False)
    for instance, name in [(parent, "runtime_model.py"), (processor, "runtime_processor.py")]:
        shutil.copyfile(inspect.getfile(type(instance)), directory / name)
    module = SpatialProgramAssessmentModel(parent, seed=configuration["seed"],
                                           mixture_components=configuration.get("mixture_components", 0)).eval()
    parent_before = state_digest(parent)
    integrated_before = state_digest(module)
    image, _, input_digest = read_frame(root / "acquisition01", selected[configuration["selection_index"]])
    inputs = processor(images=Image.fromarray(image), return_tensors="pt")
    forward_started = time.perf_counter()
    result = module(**inputs, ray_coordinates=configuration["ray_coordinates"])
    forward_seconds = time.perf_counter() - forward_started
    coordinates = result["unique_coordinates"].tolist()
    assert len(coordinates) == 5
    original_to_unique = result["original_to_unique"]
    assert int(original_to_unique[0]) == int(original_to_unique[4])
    reference = np.load(root / "inference01" / "prediction_000.npy", allow_pickle=False)
    expected = torch.tensor([reference[row, column] for row, column in coordinates])
    parent_error = float((result["parent_depth_metres"] - expected).abs().max().detach())
    distribution = result["distribution"]
    if configuration.get("mixture_components", 0):
        assert torch.equal(distribution.locations, result["parent_depth_metres"].log()[None, :].expand_as(distribution.locations))
    median_error = float((distribution.loc.exp() - result["parent_depth_metres"]).abs().max().detach())
    assert parent_error <= configuration["parent_depth_tolerance_metres"]
    assert median_error <= configuration["parent_depth_tolerance_metres"]
    threshold = float(result["parent_depth_metres"][int(original_to_unique[0])].detach())
    original_programs = [
        [["above", 0, threshold]],
        [["less", 0, 4]],
        [["above", 0, threshold], ["above", 4, threshold], ["xor", 0, 1]],
        [["above", 0, threshold], ["not", 0]],
        [["above", 0, threshold], ["above", 4, threshold], ["and", 0, 1]],
    ]
    programs = [remap_program_rays(program, original_to_unique) for program in original_programs]
    generator = torch.Generator().manual_seed(configuration["seed"])
    samples = sample_joint_scene(distribution, configuration["samples_per_batch"], generator)
    events = torch.stack([evaluate_program(samples, program) for program in programs], dim=1)
    probabilities = events.float().mean(0)
    assert not events[:, 1].any() and not events[:, 2].any()
    assert torch.equal(~events[:, 0], events[:, 3])
    assert torch.equal(events[:, 0], events[:, 4])
    assert abs(float(probabilities[0]) - 0.5) < configuration["median_probability_tolerance"]
    backward_started = time.perf_counter()
    loss = estimate_brier_gradient(distribution, [programs[0]], [0], configuration["samples_per_batch"], generator)
    loss["gradient_surrogate"].backward()
    backward_seconds = time.perf_counter() - backward_started
    gradients = {"parent_final_weight": float(parent.head.conv3.weight.grad.norm())}
    gradients.update({name: float(parameter.grad.norm()) for name, parameter in module.scene_distribution.named_parameters()})
    assert all(np.isfinite(value) and value > 0 for value in gradients.values()), gradients
    assert parent_before == state_digest(parent)
    assert integrated_before == state_digest(module)
    summary = {"status": "complete", "input_sha256": input_digest,
               "parent_state_sha256_before_and_after": parent_before,
               "integrated_state_sha256_before_and_after": integrated_before,
               "parent_parameters": sum(parameter.numel() for parameter in parent.parameters()),
               "total_parameters": sum(parameter.numel() for parameter in module.parameters()),
               "mixture_components": configuration.get("mixture_components", 0),
               "original_ray_references": configuration["ray_coordinates"], "unique_ray_coordinates": coordinates,
               "original_to_unique": original_to_unique.tolist(), "diagnostic_program_probabilities": probabilities.tolist(),
               "diagnostic_threshold_metres": threshold,
               "maximum_parent_reproduction_error_metres": parent_error,
               "maximum_initial_median_error_metres": median_error,
               "gradient_norms": gradients, "forward_seconds": forward_seconds, "backward_seconds": backward_seconds,
               "neural_training": False, "optimizer_steps": 0, "wall_seconds": time.perf_counter() - started}
    (directory / "programs.json").write_text(json.dumps({"original": original_programs, "remapped": programs}, indent=2) + "\n")
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
