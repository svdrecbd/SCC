"""Evaluate a pinned, unchanged spatial model on reflected development inputs."""
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


def main(directory, root):
    started = time.perf_counter()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    configuration = json.loads((directory / 'config.json').read_text())
    selected = json.loads((directory / 'selection.json').read_text())
    model_directory = root / 'acquisition01' / 'model'
    parent = DepthAnythingForDepthEstimation(DepthAnythingConfig.from_pretrained(str(model_directory), local_files_only=True)).float().cpu().eval()
    parent.load_state_dict(load_file(str(model_directory / 'model.safetensors'), device='cpu'), strict=True)
    processor = AutoImageProcessor.from_pretrained(str(model_directory), local_files_only=True, use_fast=False)
    for instance, name in ((parent, 'runtime_model.py'), (processor, 'runtime_processor.py')):
        shutil.copyfile(inspect.getfile(type(instance)), directory / name)
    before = state_digest(parent)
    assert before == configuration['parent_state_sha256']
    cases = []
    for index in configuration['selection_indices']:
        image, _, digest = read_frame(root / 'acquisition01', selected[index])
        inputs = processor(images=Image.fromarray(image[:, ::-1].copy()), return_tensors='pt')
        began = time.perf_counter()
        with torch.inference_mode():
            output = parent(**inputs).predicted_depth
            resized = torch.nn.functional.interpolate(output.unsqueeze(1), size=(480, 640), mode='bilinear', align_corners=False)
        aligned = resized[0, 0].cpu().numpy()[:, ::-1].copy().astype(np.float32)
        assert np.isfinite(aligned).all()
        np.save(directory / f'reflected_prediction_{index:03d}.npy', aligned, allow_pickle=False)
        cases.append({'selection_index': index, 'input_sha256': digest,
                      'inference_seconds': time.perf_counter()-began})
        (directory / 'cases.json').write_text(json.dumps(cases, indent=2)+'\n')
        print(json.dumps(cases[-1]), flush=True)
    assert state_digest(parent) == before
    summary = {'status': 'complete', 'case_count': len(cases), 'parent_state_sha256': before,
               'neural_training': False, 'optimizer_steps': 0,
               'wall_seconds': time.perf_counter()-started}
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
