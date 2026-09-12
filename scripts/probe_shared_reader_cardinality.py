"""Follow-up payload-statistics probe on frozen shared-reader endpoints."""

import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from audit_shared_reader import numpy_weights, scalar_read


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    root,out=Path(args.parent),Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str((root/'source').resolve()))
    from scc.shared_reader import SharedReader, confusion
    from scc.provenance import atomic_json, digest, file_digest
    torch.set_num_threads(1)
    config=json.loads((root/'contract.json').read_text())
    files=[Path(__file__),Path(__file__).with_name('audit_shared_reader.py'),Path(__file__).with_name('audit_shared_predicate.py')]
    for f in files: shutil.copyfile(f,out/f.name)
    atomic_json(out/'contract.json',{'scope':'Post-run development probe; fixed reader-only endpoints, no best-case selection',
        'parent':str(root),'parent_contract_sha256':file_digest(root/'contract.json'),
        'source':{f.name:file_digest(out/f.name) for f in files},
        'calls_per_payload_count':512,'gpu_cost_usd':0})
    gen=torch.Generator().manual_seed(980000)
    calls=[]
    for count in range(17):
        queries=torch.randint(16,(512,),generator=gen)
        keys=torch.arange(16).expand(512,-1)
        ranks=torch.rand(512,16,generator=gen).argsort(-1).argsort(-1)
        payload=(ranks<count).double()[...,None]
        labels=payload[torch.arange(512),queries]
        calls.append((queries,keys,payload,labels))
    torch.save(calls,out/'calls.pt')
    result={'arms':{},'numpy_permission_logits_verified':0,'scc_mechanism_established':False}
    for kind in ('sparse','matched'):
        for seed in config['configuration']['seeds']:
            arm=f'{kind}-seed-{seed}'
            reference=torch.load(root/arm/'clean.pt',weights_only=True)['model']
            result['arms'][arm]={}
            for case in ('clean','reader_only'):
                checkpoint=root/arm/f'{case}.pt'
                record=json.loads(checkpoint.with_suffix('.json').read_text())
                assert file_digest(checkpoint)==record['checkpoint_sha256']
                saved=torch.load(checkpoint,weights_only=True)
                model=SharedReader()
                model.load_state_dict(saved['model'])
                if case=='reader_only':
                    for key,value in saved['model'].items():
                        if not key.startswith('reader.'):
                            assert torch.equal(value,reference[key]),key
                state={k:v.numpy() for k,v in saved['model'].items()}
                numpy_table=numpy_weights(state)
                rows=[]
                with torch.no_grad():
                    weights=model.weights()
                    for count,(queries,keys,payload,labels) in enumerate(calls):
                        logits=model.read_logits(weights,queries,keys,payload,permission=True)
                        row={'payload_ones':count,**confusion(logits,labels),
                             'calls_sha256':digest([item.tolist() for item in calls[count]])}
                        for index in (0,101,307):
                            _,values=scalar_read(state,numpy_table,int(queries[index]),keys[index].tolist(),
                                payload[index,:,0].long().tolist(),1,0,permission=True,separate=False)
                            assert abs(values[0]-float(logits[index,0]))<1e-10
                            result['numpy_permission_logits_verified']+=1
                        rows.append(row)
                atomic_json(out/f'{arm}-{case}.json',{'rows':rows,'checkpoint_sha256':file_digest(checkpoint)})
                result['arms'][arm][case]=[{'payload_ones':r['payload_ones'],'false_acceptance':r['false_acceptance'],
                                          'true_acceptance':r['true_acceptance']} for r in rows]
    result['status']='completed_and_replayed'
    atomic_json(out/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='arms'}))


if __name__=='__main__':main()
