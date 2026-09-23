"""Save unchanged parent mixture fields for four sequential development scenes."""

from pathlib import Path
import json
import sys
import time
import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch
from transformers import AutoImageProcessor, DepthAnythingConfig, DepthAnythingForDepthEstimation
from data import read_frame
from evaluate_model import state_digest
from spatial_program_assessment import SpatialProgramAssessmentModel


def main(directory,root,block):
    started=time.perf_counter()
    configuration=json.loads((directory/'config.json').read_text())
    selection=json.loads((directory/'selection.json').read_text())
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    torch.manual_seed(configuration['model_initialization_seed'])
    model_directory=root/'acquisition01'/'model'
    parent=DepthAnythingForDepthEstimation(DepthAnythingConfig.from_pretrained(str(model_directory),local_files_only=True)).float().cpu().eval()
    parent.load_state_dict(load_file(str(model_directory/'model.safetensors'),device='cpu'),strict=True)
    processor=AutoImageProcessor.from_pretrained(str(model_directory),local_files_only=True,use_fast=False)
    module=SpatialProgramAssessmentModel(parent,seed=configuration['model_initialization_seed'],mixture_components=4).eval()
    before=state_digest(module);parent_before=state_digest(parent)
    coordinates=[[row,column] for row in configuration['rows'] for column in configuration['columns']]
    records=[]
    for index in configuration['development_indices'][4*block:4*(block+1)]:
        image,truth,digest=read_frame(root/'acquisition01',selection[index])
        inputs=processor(images=Image.fromarray(image),return_tensors='pt')
        with torch.no_grad():
            result=module(**inputs,ray_coordinates=coordinates)
        ordered=result['unique_coordinates'].tolist()
        assert ordered==coordinates
        reference=np.load(root/f'inference{index//4+1:02d}'/f'prediction_{index:03d}.npy',allow_pickle=False)
        expected=torch.tensor([reference[row,column] for row,column in coordinates])
        error=float((result['parent_depth_metres']-expected).abs().max())
        assert error<=1e-6
        depths=np.array([truth[row,column] for row,column in coordinates],dtype=np.float64)
        assert np.isfinite(depths).all() and (depths>0).all()
        distribution=result['distribution']
        np.savez(directory/f'scene_{index:03d}.npz',coordinates=np.array(coordinates),depths=depths,
                 logits=distribution.logits.numpy(),locations=distribution.locations.numpy(),
                 factors=distribution.factors.numpy(),diagonal=distribution.diagonal.numpy())
        records.append({'selection_index':index,'input_sha256':digest,'parent_error_metres':error})
    assert state_digest(parent)==parent_before and state_digest(module)==before
    summary={'status':'complete','scenes':records,'parent_sha256_before_and_after':parent_before,
             'integrated_sha256_before_and_after':before,'neural_training':False,'optimizer_steps':0,
             'wall_seconds':time.perf_counter()-started}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main(Path(sys.argv[1]),Path(sys.argv[2]),int(sys.argv[3]))
