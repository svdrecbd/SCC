"""Full-sized actual-update calibration, explicitly separate from learned results."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scc.coupling import nll
from scc.developmental_run import configure,TextBank,Streams,environment
from scc.discrete_models import DiscreteModel,discrete_config
from scc.discrete_train import guarded_construction_step
from scc.discrete_objective import discrete_objective
from scc.portfolio_objective import make_contraction_episode
from scc.provenance import atomic_json,snapshot_sources,source_manifest,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    source=snapshot_sources(a.output/'source');shutil.copyfile(__file__,a.output/'runner.py')
    device=configure('cpu',2);bank=TextBank(a.data,2);started=time.monotonic();records=[]
    for bits in (32,128):
        for hard in (False,True):
            for ordinal in (3,4):
                torch.manual_seed(23)
                model=DiscreteModel(discrete_config(bits,hard)).eval()
                episode=make_contraction_episode(bank,ordinal,device,batch_size=2,inner_steps=8)
                ordinary=Streams(bank,101,device,2,.5).task('lookup','authorized')
                record=guarded_construction_step(model,episode,ordinary)
                verified,details=discrete_objective(model,episode,create_graph=False)
                tolerance=1e-6*max(1.,abs(record['after']))
                assert abs(float(verified.detach())-record['after'])<=tolerance
                if record['accepted']:
                    assert record['after']<record['before']-1e-6*max(1.,abs(record['before']))
                    selected=record['candidate_checks'][record['selected_candidate']]
                    assert selected['ordinary_guard_passed']
                counts=details['code_counts']
                if hard:assert all(v['exact_binary'] for v in counts.values())
                row={'bits':bits,'hard':hard,'scope':episode['configuration']['inner_scope'],
                     'update':record,'verification_score':float(verified.detach()),'post_codes':counts}
                records.append(row);atomic_json(a.output/'progress.json',records)
                print(json.dumps({'bits':bits,'hard':hard,'scope':row['scope'],'accepted':record['accepted'],
                    'before':record['before'],'after':record['after'],'gradient_norm':record['proposal_gradient_norm'],
                    'selected_orientation':None if record['selected_candidate'] is None else record['candidate_checks'][record['selected_candidate']]['orientation']}),flush=True)
                del model,verified
    assert source_manifest()==source
    atomic_json(a.output/'result.json',{'status':'complete','checks':records,'source_unchanged':True,
                'elapsed_seconds':time.monotonic()-started,'environment':environment(device),
                'scope':'Actual score and ordinary-batch acceptance checks at full-sized initialization; not intact qualification, developmental effectiveness, or SCC.'})


if __name__=='__main__':main()
