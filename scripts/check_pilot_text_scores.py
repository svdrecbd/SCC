"""Recompute reported calibrated text losses from a checkpoint and tiny input export."""

import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from scc.developmental_run import configure
from scc.model import Transformer,ModelConfig
from scc.provenance import atomic_json,file_digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--measurement',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    configure('cpu',2)
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    c=state.get('configuration') or state['contract']['configuration']
    model=Transformer(ModelConfig(**c['model'])).eval();model.load_state_dict(state['model']);del state
    inputs=np.load(a.inputs/'scoring-inputs.npz');manifest=json.loads((a.inputs/'manifest.json').read_text())
    assert file_digest(a.inputs/'scoring-inputs.npz')==manifest['npz_sha256']
    measured=json.loads(a.measurement.read_text());checks=[]
    for branch in measured['branches']:
        reader=branch['reader'];indices=list(range(260))
        indices[52:62]=[52+x for x in reader['digit_sources']]
        per_source={}
        with torch.no_grad():
            for source in ('wikimedia','pressbooks','libretexts','gutenberg'):
                tokens=torch.from_numpy(inputs['validation_'+source+'_tokens'])
                targets=torch.from_numpy(inputs['validation_'+source+'_targets'])
                summed=0.;count=0
                for x,y in zip(tokens.split(64),targets.split(64)):
                    z=model(x)[...,indices]*reader['sign']/branch['temperature']
                    summed+=float(F.cross_entropy(z.flatten(0,1),y.flatten(),ignore_index=-100,reduction='sum'))
                    count+=int((y!=-100).sum())
                value=summed/count;stored=branch['layouts']['original']['calibrated_text'][source]['nll']
                error=abs(value-stored)
                per_source[source]={'recomputed':value,'stored':stored,'absolute_error':error,'tokens':count}
                if error>2e-5:raise AssertionError((source,value,stored))
        checks.append({'reader':reader,'temperature':branch['temperature'],'sources':per_source})
    result={'passed':True,'checkpoint_sha256':file_digest(a.checkpoint),'measurement_sha256':file_digest(a.measurement),
            'scoring_inputs_sha256':manifest['npz_sha256'],'checks':checks,'absolute_tolerance':2e-5,
            'scope':'CPU recalculation of GPU/CPU text losses from exact tokens; not a cognition destruction test'}
    atomic_json(a.output,result);print(json.dumps({'passed':True,'branches':len(checks),'checkpoint':str(a.checkpoint)}))


if __name__=='__main__':main()
