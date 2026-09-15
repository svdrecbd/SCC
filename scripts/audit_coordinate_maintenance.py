"""Independent saved-output rescore and native nn.GRU reference for LN-115."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from scc.binding_bank import parameter_shapes
from scc.persistent_tasks import TOKEN_COUNT
from scc.provenance import atomic_json,file_digest


@torch.no_grad()
def audit(run,dest):
    dest.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    read=lambda p:json.loads(p.read_text())
    counts={}
    for manifest,folder in [('artifact-manifest.json',run),('source-manifest.json',run/'source'),
                            ('input-manifest.json',run/'inputs')]:
        entries=read(run/manifest)
        for name,h in entries.items():assert file_digest(folder/name)==h,name
        counts[manifest]=len(entries)
    data=read(run/'inputs/requests.json');rows=data['evaluation']
    labels=torch.tensor([[r['label'] for r in s] for s in rows])
    target=torch.tensor([[r['family']=='lookup' for r in s] for s in rows])
    payload=torch.load(run/'inputs/trained.pt',weights_only=True)['physical']
    summary=read(run/'summary.json');assert summary['status']=='complete'
    assert len(summary['results'])==14
    native={};native_errors={};rescored=0
    for dtype in (torch.float32,torch.float64):
        parameters={};offset=0
        for name,shape in parameter_shapes(128).items():
            count=math.prod(shape);parameters[name]=payload[offset:offset+count].reshape(shape).to(dtype);offset+=count
        gru=torch.nn.GRU(TOKEN_COUNT,128,batch_first=True).to(dtype)
        gru.load_state_dict({k.removeprefix('recurrent.'):v for k,v in parameters.items() if k.startswith('recurrent.')})
        hidden=torch.zeros(1,4,128,dtype=dtype);logits=[];states=[]
        for panel in ('warmup','evaluation'):
            ids=torch.tensor([[r['tokens'] for r in s] for s in data[panel]])
            for j in range(ids.shape[1]):
                _,hidden=gru(F.one_hot(ids[:,j],TOKEN_COUNT).to(dtype),hidden)
                if panel=='evaluation':
                    states.append(hidden[0].clone())
                    logits.append(F.linear(hidden[0],parameters['readout.weight'],parameters['readout.bias']))
        native[str(dtype)]={'logits':torch.stack(logits,1),'hidden':torch.stack(states,1)}
        torch.save(native[str(dtype)],dest/f'native-reference-{dtype}.pt')
    for entry in summary['results']:
        case,precision=entry['case'],entry['precision']
        out=torch.load(run/f'{case}-{precision}.pt',weights_only=True)
        assert all(torch.isfinite(v).all() for v in out.values())
        pred=out['logits'].argmax(-1);correct=pred==labels;metrics=entry['metrics']
        rescored+=labels.numel()
        assert float(correct.double().mean())==metrics['task_accuracy']
        gates=[]
        for key,cell in metrics['cells'].items():
            family,layout=key.split('/')
            subset=[(i,j) for i,s in enumerate(rows) for j,r in enumerate(s) if (r['family'],r['layout'])==(family,layout)]
            n=len(subset);k=sum(int(correct[i,j]) for i,j in subset);accuracy=k/n
            late=[(i,j) for i,j in subset if j>=len(rows[0])//2]
            late_accuracy=sum(int(correct[i,j]) for i,j in late)/len(late)
            z=1.959963984540054
            lower=(accuracy+z*z/(2*n)-z*math.sqrt(accuracy*(1-accuracy)/n+z*z/(4*n*n)))/(1+z*z/n)
            qualified=n>=128 and accuracy>=.95 and lower>=.9 and late_accuracy>=.95
            assert (n,k,accuracy,len(late),late_accuracy,qualified)==(cell['n'],cell['correct'],cell['accuracy'],cell['late_n'],cell['late_accuracy'],cell['qualified'])
            assert abs(lower-cell['wilson_lower'])<1e-14
            frequencies=Counter(int(labels[i,j]) for i,j in subset)
            assert frequencies[0]/n==cell['constant_zero_accuracy']
            assert max(frequencies.values())/n==cell['majority_accuracy']
            gates.append(qualified)
        selective=case in ('output_only','fixed_coordinates','repacked')
        desired=torch.tensor([True,False,False,True]).expand_as(out['admitted']).clone()
        if selective:desired[:,:,1]=target
        errors=int((desired!=out['admitted']).sum())
        emitted=torch.where(out['admitted'],pred[:,:,None],3)
        assert torch.equal(out['emitted'],emitted)
        target_correct=int((emitted[:,:,1][target]==labels[target]).sum())
        assert (int(target.sum()),target_correct,int(out['admitted'][:,:,1][target].sum()))==(metrics['target_n'],metrics['target_correct_answers'],metrics['target_admissions'])
        if selective:
            assert errors==metrics['exception_rule_errors']==0
            assert metrics['diagnostic_recovery_qualified']==(all(gates) and target_correct/int(target.sum())>=.95 and errors==0)
        else:
            original=torch.tensor([True,False,False,True])
            group_acc=[float((out['admitted'][:,:,i]==original[i]).double().mean()) for i in range(4)]
            assert metrics['qualified']==(all(gates) and all(v>=.95 for v in group_acc))
        # Independent inverse: scatter encoded coordinates into logical positions.
        decoded=torch.empty_like(out['encoded'])
        indexes=(torch.arange(128)[None,None,:]+out['key'][:,:,None])%128
        signs=torch.where(out['key']>=128,-1.,1.).to(decoded.dtype)
        decoded.scatter_(2,indexes,out['encoded']*signs[:,:,None])
        assert torch.equal(decoded,out['hidden'])
        if case!='reader_mismatch':
            reference=native[precision];atol,rtol=(1e-4,1e-3) if precision=='torch.float32' else (1e-9,1e-7)
            for k in ('logits','hidden'):torch.testing.assert_close(out[k],reference[k],atol=atol,rtol=rtol)
            assert torch.equal(pred,reference['logits'].argmax(-1))
            native_errors[f'{case}-{precision}']=float((out['logits']-reference['logits']).abs().max())
        intact=torch.load(run/f'intact-{precision}.pt',weights_only=True)
        for key,expected in entry['equal_to_intact'].items():assert torch.equal(out[key],intact[key])==expected
        assert len(out['maintenance_key'].unique())==entry['distinct_maintenance_keys']
    output={'passed':True,'hash_counts':counts,'saved_panels':14,'task_predictions_rescored':rescored,
            'native_nn_gru_decisions_match':True,'native_max_logit_errors':native_errors,
            'candidate_rejected':True,'scope':'Verified saved outputs, physical encoding and independent native GRU forward execution'}
    atomic_json(dest/'audit.json',output)
    (dest/'audit-source.py').write_text(Path(__file__).read_text())
    atomic_json(dest/'manifest.json',{str(p.relative_to(dest)):file_digest(p) for p in dest.iterdir() if p.is_file() and not p.name.startswith('._')})
    print(json.dumps({k:v for k,v in output.items() if k!='native_max_logit_errors'}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.run,a.output)
