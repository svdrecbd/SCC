"""Check integrated program derivatives through an unchanged visual parent."""

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
from conditional_program_assessment import integrated_brier_objective
from data import read_frame
from evaluate_model import state_digest
from spatial_program_assessment import SpatialProgramAssessmentModel


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
    for instance, name in ((parent, "runtime_model.py"), (processor, "runtime_processor.py")):
        shutil.copyfile(inspect.getfile(type(instance)), directory / name)
    module = SpatialProgramAssessmentModel(parent, seed=configuration["seed"], mixture_components=4).eval()
    parent_before = state_digest(parent)
    integrated_before = state_digest(module)
    image, truth, input_digest = read_frame(root / "acquisition01", selected[0])
    coordinates = [[row, column] for row in configuration["rows"] for column in configuration["columns"]]
    assert len(coordinates) == 64
    inputs = processor(images=Image.fromarray(image), return_tensors="pt")
    forward_started = time.perf_counter()
    result = module(**inputs, ray_coordinates=coordinates)
    forward_seconds = time.perf_counter()-forward_started
    assert len(result["unique_coordinates"]) == 64
    reference = np.load(root / "inference01" / "prediction_000.npy", allow_pickle=False)
    expected = torch.tensor([reference[row, column] for row, column in result["unique_coordinates"].tolist()])
    parent_error = float((result["parent_depth_metres"]-expected).abs().max().detach())
    assert parent_error <= 1e-6
    true_depths = np.array([truth[row, column] for row, column in coordinates])
    assert np.isfinite(true_depths).all() and (true_depths > 0).all()
    comparisons = true_depths[::2] < true_depths[1::2]
    mapping = result["original_to_unique"].tolist()
    distribution = result["distribution"]
    generator = torch.Generator().manual_seed(configuration["seed"])
    objectives = []
    records = []
    for length in (1, 8, 32):
        pairs = [[mapping[2*index], mapping[2*index+1]] for index in range(length)]
        for operation in ("xor", "majority"):
            target = int(comparisons[:length].sum() % 2 if operation == "xor"
                         else comparisons[:length].sum()*2 > length)
            first = torch.randn((configuration["samples_per_batch"], 4, 4), generator=generator)
            second = torch.randn((configuration["samples_per_batch"], 4, 4), generator=generator)
            estimate = integrated_brier_objective(distribution, pairs, operation, target, first, second)
            objectives.append(estimate["objective"])
            location_derivative = torch.autograd.grad(estimate["objective"], distribution.locations, retain_graph=True)[0]
            records.append({"length": length, "operation": operation, "target": target,
                            "first_probability": float(estimate["first_probability"].detach()),
                            "second_probability": float(estimate["second_probability"].detach()),
                            "maximum_absolute_location_derivative": float(location_derivative.abs().max()),
                            "nonzero_location_derivatives": int(torch.count_nonzero(location_derivative))})
    objective = torch.stack(objectives).mean()
    backward_started = time.perf_counter()
    objective.backward()
    backward_seconds = time.perf_counter()-backward_started
    gradients = {"parent_final_weight": float(parent.head.conv3.weight.grad.norm())}
    gradients.update({name: float(parameter.grad.norm()) for name, parameter in module.scene_distribution.named_parameters()})
    invariant_parameters = ("component_mean_projection.bias", "component_factor_projection.bias")
    assert all(np.isfinite(value) for value in gradients.values()), gradients
    assert all(gradients[name] == 0 for name in invariant_parameters), gradients
    assert all(value > 0 for name, value in gradients.items() if name not in invariant_parameters), gradients
    assert state_digest(parent) == parent_before and state_digest(module) == integrated_before
    result = {"status": "complete", "input_sha256": input_digest, "parent_state_sha256_before_and_after": parent_before,
              "integrated_state_sha256_before_and_after": integrated_before, "parent_reproduction_error_metres": parent_error,
              "objective": float(objective.detach()), "programs": records, "gradient_norms": gradients,
              "exact_comparison_invariances": list(invariant_parameters),
              "forward_seconds": forward_seconds, "backward_seconds": backward_seconds,
              "neural_training": False, "optimizer_steps": 0, "wall_seconds": time.perf_counter()-started}
    (directory / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
