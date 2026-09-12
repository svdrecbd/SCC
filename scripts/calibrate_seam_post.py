"""Open intact-performance calibration for the SEAM adaptation, with all cells saved."""

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from scc.checkpoint import load_checkpoint
from scc.developmental_run import TextBank,configure,evaluate
from scc.developmental_tasks import evaluation_rows
from scc.model import ModelConfig,Transformer
from scc.pilot_objectives import seam_episode
from scc.provenance import atomic_json,file_digest,snapshot_sources
from scc.recovered_pilot import configuration
from scc.recovered_pilot_evaluation import challenge
from run_seam_post import train,readiness


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--data',required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);snapshot_sources(a.output/'source')
    for name in ('calibrate_seam_post.py','run_seam_post.py'):shutil.copyfile(Path(__file__).with_name(name),a.output/name)
    shutil.copyfile(Path(__file__).resolve().parents[1]/'protocols/SCC_SEAM_INTACT_CALIBRATION_V1.md',a.output/'protocol.md')
    device=configure(a.device,4 if a.device=='cuda' else 2);bank=TextBank(a.data,blocks=128)
    c=configuration();c.update(steps=500,lr=2e-5,meta_batch_size=8,meta_weight=1.)
    readiness(bank,c,a.output/'readiness',device)
    saved=load_checkpoint(a.parent);parent_hash=file_digest(a.parent)
    assert saved['contract']['text']['data_sha256']==bank.manifest()['data_sha256']
    cells={};models={};started=time.monotonic()
    for name,lr,alpha in (('lr2e-6',2e-6,1.),('lr2e-7',2e-7,1.),('alpha10',2e-5,10.)):
        cfg=copy.deepcopy(c);cfg.update(lr=lr,seam_alpha=alpha)
        model=train(saved['model'],bank,cfg,a.output/name,device)
        model.eval()
        scores={key:evaluate(model,bank,evaluation_rows(128,seed=cfg['evaluation_seed'],reordered=r))
                for key,r in (('original',False),('reordered',True))}
        qualified=all(v['qualification']['passed'] for v in scores.values())
        # Selection data are fresh train episodes, independent of modification tests.
        cosines=[]
        for ordinal in range(5000,5007):
            _,detail=seam_episode(model,bank,cfg,ordinal,device);cosines.append(detail['gradient_cosine'])
        cells[name]={'configuration':cfg,'evaluation':scores,'qualified':qualified,
                     'selection_gradient_cosines':cosines,'mean_gradient_cosine':sum(cosines)/len(cosines)}
        atomic_json(a.output/name/'calibration.json',cells[name]);models[name]=model
        print(json.dumps({'cell':name,'qualified':qualified,'mean_gradient_cosine':cells[name]['mean_gradient_cosine']}),flush=True)
    eligible=[name for name,result in cells.items() if result['qualified']]
    selected=min(eligible,key=lambda name:(cells[name]['mean_gradient_cosine'],name)) if eligible else None
    results={'cells':cells,'selected':selected,'parent_sha256':parent_hash,'status':'complete',
             'evidence_class':'Open method adaptation calibration; selection is not confirmation',
             'positive_scc_result':False}
    atomic_json(a.output/'selection.json',results)
    if selected:
        result=challenge(models[selected],bank,cells[selected]['configuration'],a.output/'challenges-selected',started+1700)
        results['challenges']=result
    assert file_digest(a.parent)==parent_hash
    atomic_json(a.output/'result.json',results)
    if os.environ.get('GMN_RESULT_PATH'):atomic_json(os.environ['GMN_RESULT_PATH'],{'status':'complete','selected':selected,'positive_scc_result':False})


if __name__=='__main__':main()
