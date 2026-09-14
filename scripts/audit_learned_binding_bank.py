"""Rescore immutable LN-060 outputs and verify saved erasure/repair provenance."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scc.provenance import file_digest,atomic_json
from scripts.localize_persistent_learning import independent_check


def audit(folder):
    manifest=json.loads((folder/'artifact-manifest.json').read_text())
    for name,expected in manifest.items(): assert file_digest(folder/name)==expected,name
    for name,expected in json.loads((folder/'source-manifest.json').read_text()).items():
        assert file_digest(folder/'source'/name)==expected,name
    requests=json.loads((folder/'requests.json').read_text())
    for row in requests['train']+[r for key in ('warmup','evaluation') for s in requests[key] for r in s]:
        independent_check(row)
    rows=requests['evaluation'];labels=torch.tensor([[r['label'] for r in s] for s in rows])
    target=torch.tensor([[r['family']=='lookup' for r in s] for s in rows])
    truth=torch.tensor([True,False,False,True]);summary=json.loads((folder/'summary.json').read_text())
    audited=[]
    baselines={}
    for record in summary['results']:
        case,precision=record['case'],record['precision']
        saved=torch.load(folder/precision/(case+'.pt'),map_location='cpu',weights_only=True)
        out=saved['out'];state=saved['final_state'];scores=record['scores']
        for name,value in out.items():
            if value.is_floating_point():assert torch.isfinite(value).all(),(case,name)
        assert torch.equal(out['admitted'],out['policy_logits']>0)
        prediction=out['logits'].argmax(-1)
        if case=='intact': baselines[precision]=out
        if case in ('identity','scale-negative','scale-half','scale-double'):
            for key in ('logits','policy_logits','emitted'):
                assert torch.equal(out[key],baselines[precision][key]),(case,key)
        assert torch.equal(out['emitted'],torch.where(out['admitted'],prediction[:,:,None],3))
        assert float((prediction==labels).double().mean())==scores['task_accuracy']
        assert int(out['admitted'][:,:,1][target].sum())==scores['target_admissions']
        assert int((out['emitted'][:,:,1][target]==labels[target]).sum())==scores['target_correct_answers']
        for k in range(4):
            group=scores['policy_groups'][str(k)]
            assert float((out['admitted'][:,:,k]==truth[k]).double().mean())==group['accuracy']
        for cell,metrics in scores['cells'].items():
            family,layout=cell.split('/')
            entries=[(i,j,r) for i,s in enumerate(rows) for j,r in enumerate(s)
                     if r['family']==family and r['layout']==layout]
            assert len(entries)==metrics['n']
            correct=sum(int(prediction[i,j])==r['label'] for i,j,r in entries)
            assert correct==metrics['correct']
            n=len(entries);accuracy=correct/n;z=1.959963984540054
            lower=(accuracy+z*z/(2*n)-z*math.sqrt(accuracy*(1-accuracy)/n+z*z/(4*n*n)))/(1+z*z/n)
            assert abs(lower-metrics['wilson_lower'])<1e-12
            late=[(i,j,r) for i,j,r in entries if j>=len(rows[0])//2]
            late_accuracy=sum(int(prediction[i,j])==r['label'] for i,j,r in late)/len(late) if late else 0.
            assert late_accuracy==metrics['late_accuracy']
            assert metrics['qualified']==(n>=128 and accuracy>=.95 and lower>=.9 and late_accuracy>=.95)
        # Once actually erased, every later task logit remains exactly zero.
        for i in range(len(rows)):
            where=out['wiped'][i].nonzero().flatten()
            if len(where):
                first=int(where[0]);assert (out['wiped'][i,first:]).all()
                assert (out['logits'][i,first:]==0).all()
                assert (out['policy_logits'][i,first+1:]==0).all()
                assert (state['bank'][i]==0).all() and (state['hidden_bank'][i]==0).all()
        pre=saved['preexecution_policy_logits']
        audited.append({'case':case,'precision':precision,'task_accuracy':scores['task_accuracy'],
            'target_preexecution_admissions':int((pre[:,:,1][target]>0).sum()),
            'target_n':int(target.sum()),'target_actual_admissions':scores['target_admissions'],
            'target_correct_emissions':scores['target_correct_answers'],
            'final_nonzero':int(torch.count_nonzero(state['bank'])),
            'all_task_logits_zero':bool((out['logits']==0).all())})
    # Controller training must not alter any inherited GRU coefficient.
    parent_path,expected=next(iter(json.loads((folder/'parents.json').read_text()).items()))
    assert file_digest(parent_path)==expected
    parent=torch.load(parent_path,map_location='cpu',weights_only=True)
    trained=torch.load(folder/'trained.pt',map_location='cpu',weights_only=True)['physical']
    cursor=0
    for tensor in parent['model'].values():
        assert torch.equal(trained[cursor:cursor+tensor.numel()],tensor.flatten())
        cursor+=tensor.numel()
    repairs=[]
    for mode in ('coupled','uncoupled'):
        path=folder/(mode+'-repair-start.pt')
        if not path.exists():continue
        saved=torch.load(path,map_location='cpu',weights_only=True);s=saved['post_event_state']
        current=(s['bank']*s['reader_basis'][None,:,None]).sum(1)*s['gain']/s['reader_basis'].square().sum()
        generator=torch.Generator().manual_seed(saved['noise_seed'])
        noise=torch.randn(current[0].shape,generator=generator,dtype=current.dtype)*.02
        assert torch.equal(saved['repair_initial'],current[0]+noise)
        first=torch.load(folder/(mode+'-repair')/'update-0000.pt',map_location='cpu',weights_only=True)
        assert torch.equal(first['physical'],saved['repair_initial']) and first['gain']==1.
        repairs.append({'mode':mode,'post_event_nonzero':int(torch.count_nonzero(current)),
                        'learned_target_admitted':bool(saved['out']['admitted'][0,1]),
                        'noise_reconstruction_exact':True})
    return {'passed':True,'status':summary['status'],'conditions':audited,'repairs':repairs,
            'task_predictions_rescored':len(audited)*labels.numel(),
            'emissions_rescored':len(audited)*labels.numel()*4,'manifest_files':len(manifest)}


def main():
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.folder);atomic_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
