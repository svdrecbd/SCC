"""Verify frozen benchmark outputs and summarize measured blocks, not model quality."""
import argparse
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from scc.provenance import atomic_json,file_digest


def audit(run,output):
    output.mkdir(parents=True,exist_ok=False)
    read=lambda p:json.loads(p.read_text())
    hashes={}
    for name,folder in [('artifact-manifest.json',run),('source-manifest.json',run/'source'),('inputs/manifest.json',run/'inputs')]:
        manifest=read(run/name)
        for path,expected in manifest.items():assert file_digest(folder/path)==expected,path
        hashes[name]=len(manifest)
    config=read(run/'configuration.json');result=read(run/'result.json')
    assert result['status']==('fixture-complete' if config['fixture'] else 'complete')
    assert read(run/'numerical.json')['passed'] and read(run/'cpu-trajectory-equivalence.json')['passed']
    pack=torch.load(run/'inputs/windows.pt',weights_only=True)
    target,labels=pack['target'][0],pack['labels'][0]
    comparisons=0;largest={}
    for condition in ('hidden_only','lookup_projection','always_projection','neither'):
        for endpoint in ('origin','repaired'):
            for precision in ('float32','float64'):
                baseline=None
                tol=(1e-4,1e-3) if precision=='float32' else (1e-9,1e-7)
                for device in config['devices']:
                    record=torch.load(run/f'{condition}-{endpoint}-{precision}-{device}.pt',weights_only=True)
                    assert all(torch.isfinite(v).all() for v in record.values() if isinstance(v,torch.Tensor))
                    # Separate reconstruction of the pre-update objective from saved logits.
                    raw=record['policy_logits'];desired=raw.new_tensor([1.,0.,0.,1.]).expand_as(raw).clone()
                    desired[:,:,1]=target.to(raw.dtype)
                    task=[];selected=[];other=[]
                    for j in range(4):
                        mask=torch.zeros((32,4),dtype=torch.bool);mask[:,1]=target[:,j]
                        task.append(F.cross_entropy(record['logits'][:,j],labels[:,j]))
                        selected.append(F.binary_cross_entropy_with_logits(raw[:,j][mask],desired[:,j][mask]))
                        other.append(F.binary_cross_entropy_with_logits(raw[:,j][~mask],desired[:,j][~mask]))
                    loss=torch.stack(task).mean()+.5*torch.stack(selected).mean()+.5*torch.stack(other).mean()
                    torch.testing.assert_close(loss,record['loss'],atol=tol[0],rtol=tol[1])
                    if baseline is None:baseline=record
                    else:
                        for key,value in record.items():
                            if isinstance(value,torch.Tensor):
                                torch.testing.assert_close(value,baseline[key],atol=tol[0],rtol=tol[1])
                                if value.is_floating_point():largest[key]=max(largest.get(key,0),float((value-baseline[key]).abs().max()))
                            else:assert value==baseline[key]
                        assert torch.equal(record['logits'].argmax(-1),baseline['logits'].argmax(-1))
                        assert torch.equal(record['admitted'],baseline['admitted'])
                        comparisons+=1
                physical=torch.load(run/f'{condition}-{endpoint}-{precision}-physical.pt',weights_only=True)
                for key,value in physical['reduced'].items():
                    torch.testing.assert_close(value,physical['physical'][key],atol=tol[0],rtol=tol[1])
                    torch.testing.assert_close(value,baseline[key][:4],atol=tol[0],rtol=tol[1])
    timings=read(run/'timings.json')
    expected=config['repetitions']*4*len(config['devices'])*2
    assert len(timings)==result['timing_rows']==expected
    keys=set();groups={}
    for row in timings:
        key=(row['condition'],row['device'],row['mode'],row['repetition'])
        assert key not in keys;keys.add(key)
        assert len(row['logs'])==config['measured']
        assert 0<row['seconds_per_update']<float('inf')
        assert abs(row['seconds_per_update']*config['measured']-row['timed_seconds'])<1e-9
        def finite(value):
            if isinstance(value,dict):return all(finite(v) for v in value.values())
            if isinstance(value,list):return all(finite(v) for v in value)
            return math.isfinite(value)
        assert finite(row['logs'])
        group='/'.join(key[:3]);groups.setdefault(group,[]).append(row['seconds_per_update'])
    summary={key:{'median_seconds_per_update':statistics.median(values),'repetitions':len(values),'values':values} for key,values in groups.items()}
    receipt={'passed':True,'hash_counts':hashes,'fixture':config['fixture'],'gpu_comparisons':comparisons,
             'max_cpu_gpu_errors':largest,'timing_rows':expected,'timings':summary,
             'gpu':config['gpu'],'python_native_triton_enabled':config.get('python_native_triton_enabled','default/unrecorded'),
             'scope':'Saved tensors/objective/physical checks and timing contract; no independent retraining; fixture timings are not a throughput qualification'}
    atomic_json(output/'audit.json',receipt)
    (output/'audit-source.py').write_text(Path(__file__).read_text())
    print(json.dumps({k:v for k,v in receipt.items() if k!='timings'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.run,a.output)
