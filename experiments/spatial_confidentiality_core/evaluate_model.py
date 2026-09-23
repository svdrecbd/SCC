"""Run the unchanged metric depth model on a fixed block of selected scenes."""
from pathlib import Path
import hashlib
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


def state_digest(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def main(directory, acquisition):
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    configuration = json.loads((directory / "config.json").read_text())
    selected = json.loads((directory / "selection.json").read_text())
    model_directory = acquisition / "model"
    model_configuration = DepthAnythingConfig.from_pretrained(str(model_directory), local_files_only=True)
    model = DepthAnythingForDepthEstimation(model_configuration).float().cpu().eval()
    state = load_file(str(model_directory / "model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    del state
    processor = AutoImageProcessor.from_pretrained(str(model_directory), local_files_only=True, use_fast=False)
    for instance, name in [(model, "runtime_model.py"), (processor, "runtime_processor.py")]:
        shutil.copyfile(inspect.getfile(type(instance)), directory / name)
    before = state_digest(model)
    records = []
    for index in configuration["selection_indices"]:
        record = selected[index]
        assert record["selection_index"] == index
        rgb, _, digest = read_frame(acquisition, record)
        inference_started = time.perf_counter()
        inputs = processor(images=Image.fromarray(rgb), return_tensors="pt")
        with torch.inference_mode():
            output = model(**inputs).predicted_depth
            resized = torch.nn.functional.interpolate(output.unsqueeze(1), size=(480, 640),
                                                      mode="bilinear", align_corners=False)
        prediction = resized[0, 0].cpu().numpy().astype(np.float32)
        assert prediction.shape == (480, 640) and np.isfinite(prediction).all()
        np.save(directory / f"prediction_{index:03d}.npy", prediction, allow_pickle=False)
        records.append({"selection_index": index, "group": record["group"], "input_sha256": digest,
                        "processor_shape": list(inputs["pixel_values"].shape),
                        "minimum_depth": float(prediction.min()), "maximum_depth": float(prediction.max()),
                        "inference_seconds": time.perf_counter() - inference_started})
        (directory / "cases.jsonl").write_text("".join(json.dumps(row) + "\n" for row in records))
        print(json.dumps(records[-1]), flush=True)
    after = state_digest(model)
    assert before == after
    summary = {"status": "complete", "case_count": len(records),
               "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
               "state_sha256_before": before, "state_sha256_after": after,
               "processor_class": type(processor).__name__, "model_class": type(model).__name__,
               "neural_training": False, "wall_seconds": time.perf_counter() - started}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())
