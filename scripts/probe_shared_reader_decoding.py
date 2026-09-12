"""Calibrate both threshold orientations on frozen shared-reader artifacts.

This is a follow-up expanded-interface probe. Permission keeps the original
reader and threshold; a cognitive-only interpreter is explicitly outside the
shared-call weight-edit boundary.
"""

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from audit_shared_reader import numpy_weights, scalar_predict


def fit_orientation(raw_scores, labels):
    scores=np.array(raw_scores,dtype=float)
    labels=np.array(labels,dtype=int)
    choices=[]
    for polarity in (1,-1):
        values=polarity*scores
        order=np.argsort(values,kind='stable')
        ordered=values[order]
        positives=np.concatenate(([0],labels[order].cumsum()))
        cuts=np.arange(len(values)+1)
        correct=cuts-positives+positives[-1]-positives
        valid=np.concatenate(([True],np.diff(ordered)>1e-8,[True]))
        best=int(np.argmax(np.where(valid,correct,-1)))
        threshold=float(ordered[0]-1 if best==0 else ordered[-1]+1 if best==len(values)
                        else (ordered[best-1]+ordered[best])/2)
        accuracy=float(((polarity*scores>threshold)==labels).mean())
        assert accuracy==int(correct[best])/len(values)
        choices.append({'polarity':polarity,'threshold':threshold,'calibration_accuracy':accuracy})
    # Deterministic positive-orientation preference on ties.
    return max(choices,key=lambda row:row['calibration_accuracy'])


def main():
    assert fit_orientation([-3,-2,2,3],[1,1,0,0])['calibration_accuracy']==1.
    assert fit_orientation([-3,-2,2,3],[1,1,0,0])['polarity']==-1
    assert fit_orientation([1,1+1e-12,1-1e-12,1],[1,0,1,0])['calibration_accuracy']==.5
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    root,out=Path(args.parent),Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str((root/'source').resolve()))
    from scc.shared_reader import SharedReader,evaluate
    from scc.provenance import atomic_json,file_digest
    torch.set_num_threads(1)
    files=[Path(__file__),Path(__file__).with_name('audit_shared_reader.py'),Path(__file__).with_name('audit_shared_predicate.py')]
    for file in files: shutil.copyfile(file,out/file.name)
    atomic_json(out/'contract.json',{'scope':'Post-run bidirectional decoding probe; expanded cognitive-only interpretation',
        'parent_contract_sha256':file_digest(root/'contract.json'),'parent':str(root),
        'source':{f.name:file_digest(out/f.name) for f in files},'minimum_threshold_gap':1e-8,
        'selection':'Use only the original 8192 calibration calls; never select on task evaluation',
        'gpu_cost_usd':0})
    run=json.loads((root/'result.json').read_text())
    domains=json.loads((root/'problems.json').read_text())
    result={'arms':{},'numpy_serial_predictions':0,'scc_mechanism_established':False}
    for arm,summary in run['arms'].items():
        result['arms'][arm]={}
        for case in summary['cases']:
            record=json.loads((root/arm/f'{case}.json').read_text())
            if 'threshold_probe' not in record: continue
            selection=fit_orientation(record['threshold_probe']['scores'],record['threshold_probe']['labels'])
            checkpoint=root/arm/f'{case}.pt'
            assert file_digest(checkpoint)==record['checkpoint_sha256']
            saved=torch.load(checkpoint,weights_only=True)
            model=SharedReader()
            if saved['separate_reader']: model.separate_permission_reader()
            model.load_state_dict(saved['model'])
            if model.permission_reader is None: model.separate_permission_reader()
            with torch.no_grad():
                model.reader[-1].weight.mul_(selection['polarity'])
                model.reader[-1].bias.mul_(selection['polarity'])
            evaluated=evaluate(model,domains,record['primary_policy'],selection['threshold'])
            assert evaluated['policies']==record['policies'],'The diagnostic changed permission'
            if selection['polarity']==1:
                for name,task in evaluated['tasks'].items():
                    assert task['predictions']==record['threshold_probe_evaluation']['tasks'][name]['predictions']
            state={k:v.detach().numpy() for k,v in model.state_dict().items()}
            weights=numpy_weights(state)
            for name,rows in domains.items():
                for index in np.random.default_rng(682).choice(len(rows),4,replace=False):
                    expected=scalar_predict(state,weights,rows[index],selection['threshold'])
                    assert expected==evaluated['tasks'][name]['predictions'][index]
                    result['numpy_serial_predictions']+=1
            full={'selection':selection,'evaluation':evaluated,'parent_checkpoint_sha256':file_digest(checkpoint)}
            atomic_json(out/f'{arm}-{case}.json',full)
            result['arms'][arm][case]={'selection':selection,'task_exact':{k:v['exact'] for k,v in evaluated['tasks'].items()}}
    result['status']='completed_and_replayed'
    atomic_json(out/'result.json',result)
    print(json.dumps({'status':result['status'],'numpy_serial_predictions':result['numpy_serial_predictions'],
                      'negative_orientations':sum(v['selection']['polarity']==-1 for a in result['arms'].values() for v in a.values())}))


if __name__=='__main__':main()
