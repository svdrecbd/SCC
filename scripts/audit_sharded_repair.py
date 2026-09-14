"""Independent artifact, starting-state, trace and task audit for LN-069."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from scc.provenance import atomic_json,file_digest
from scripts.localize_persistent_learning import independent_check
from scripts.audit_sharded_rewrite_cell import audit_trace


def bank(s):return s['bank_values'][s['bank_indices']].reshape(s['bank_shape'])


def audit(folder):
    config=json.loads((folder/'configuration.json').read_text())
    for name,h in json.loads((folder/'artifact-manifest.json').read_text()).items():assert file_digest(folder/name)==h,name
    for name,h in json.loads((folder/'source-manifest.json').read_text()).items():assert file_digest(folder/'source'/name)==h,name
    parents=json.loads((folder/'parents.json').read_text());assert len(parents)==1
    original,h=next(iter(parents.items()))
    assert file_digest(original)==file_digest(folder/'merged-start-parent.pt')==h
    parent=torch.load(folder/'merged-start-parent.pt',weights_only=True,map_location='cpu')
    assert parent['out']['admitted'][:,1].all()
    state=parent['post_event_state'];physical=bank(state);payload=physical[0,[0,4]].flatten()
    hidden=state['hidden_bank'][:,[0,4]].reshape(4,-1)[:,:128]
    assert payload.numel()==80518 and payload[-1]!=0
    assert all(torch.equal(physical[i,j],physical[0,0]) for i in range(4) for j in range(8))
    origin=torch.load(folder/'repair-origin.pt',weights_only=True,map_location='cpu')
    assert torch.equal(origin['payload'],payload) and torch.equal(origin['hidden'],hidden)
    requests=json.loads((folder/'requests.json').read_text())
    pool=requests['train_pool']
    for row in pool:independent_check(row);assert row['split']=='train'
    for s in requests['validation']:
        for row in s:independent_check(row);assert row['split']=='validation'
    assert not {r['core_sha256'] for r in pool}&{r['core_sha256'] for s in requests['validation'] for r in s}
    assert len(requests['schedule'])==config['steps']
    for schedule in requests['schedule']:
        assert len(schedule)==8 and all(len(s)==2 for s in schedule)
        for j in range(2):
            families=[pool[s[j]]['family'] for s in schedule]
            assert [families.count(f) for f in ('lookup','parity','sum3')]==[4,2,2]
    endpoints={}
    for arm,rule in config['arms'].items():
        initial=torch.load(folder/arm/'update-0000.pt',weights_only=True,map_location='cpu')
        assert torch.equal(initial['payload'],payload) and initial['step']==0
        assert not initial['optimizer']['state']
        logs=json.loads((folder/arm/'training.json').read_text())
        assert len(logs)==config['steps'] and [r['update'] for r in logs]==list(range(1,config['steps']+1))
        assert all(math.isfinite(r['loss_before_update']) and math.isfinite(r['gradient_norm_before_clip']) for r in logs)
        end=torch.load(folder/arm/f"update-{config['steps']:04d}.pt",weights_only=True,map_location='cpu')
        assert end['step']==config['steps'] and end['payload'].shape==payload.shape
        assert len(end['optimizer']['state'])==1
        group=end['optimizer']['param_groups'][0]
        assert group['lr']==.003 and group['weight_decay']==0
        endpoints[arm]=end['payload']
    summary=json.loads((folder/'summary.json').read_text())
    details=[];n_predictions=0;ticks=0;max_error=0.
    for record in summary['results']:
        name=record['name'];saved=torch.load(folder/(name+'.pt'),weights_only=True,map_location='cpu')
        out=saved['out'];initial=saved['initial_state'];final=saved['final_state']
        for s in (initial,final):
            assert s['bank_shape']==[4,8,40259] and list(s['hidden_bank'].shape)==[4,8,64]
            assert s['normalizer_sign']==s['writer_sign']==s['code_sign']==s['gain']==1.
            assert s['execution']=='commit' and s['storage_rule']==config['arms'][record['arm']]
        expected=payload if record['endpoint']=='initial' else endpoints[record['arm']]
        effective=bank(initial)[:,[0,4]].reshape(4,-1)
        assert torch.equal(effective,expected.to(effective.dtype).expand_as(effective))
        assert torch.equal(initial['hidden_bank'],state['hidden_bank'].to(initial['hidden_bank'].dtype))
        rows=requests[record['panel']];ids=torch.tensor([[r['tokens'] for r in s] for s in rows])
        labels=torch.tensor([[r['label'] for r in s] for s in rows]);prediction=out['logits'].argmax(-1)
        target=torch.tensor([[r['family']=='lookup' for r in s] for s in rows])
        truth=torch.tensor([True,False,False,True]).expand_as(out['admitted']).clone();truth[:,:,1]|=target
        assert torch.equal(out['admitted'],out['policy_logits']>0)
        assert torch.equal(out['emitted'],torch.where(out['admitted'],prediction[:,:,None],3))
        accuracy=float((prediction==labels).double().mean());correct=int((out['emitted'][:,:,1][target]==labels[target]).sum())
        errors=int((out['admitted']!=truth).sum());scores=record['scores']
        assert accuracy==scores['task_accuracy'] and correct==scores['target_correct_answers']
        assert errors==record['exception_rule_errors']
        gate=len(scores['cells'])==6
        for cell,metrics in scores['cells'].items():
            family,layout=cell.split('/')
            chosen=[(i,j,r) for i,s in enumerate(rows) for j,r in enumerate(s) if r['family']==family and r['layout']==layout]
            n=len(chosen);acc=sum(int(prediction[i,j])==r['label'] for i,j,r in chosen)/n;z=1.959963984540054
            lower=(acc+z*z/(2*n)-z*math.sqrt(acc*(1-acc)/n+z*z/(4*n*n)))/(1+z*z/n)
            late=[(i,j,r) for i,j,r in chosen if j>=len(rows[0])//2]
            late_acc=sum(int(prediction[i,j])==r['label'] for i,j,r in late)/len(late) if late else 0.
            qualifies=n>=128 and acc>=.95 and lower>=.9 and late_acc>=.95
            assert n==metrics['n'] and acc==metrics['accuracy'] and abs(lower-metrics['wilson_lower'])<1e-12
            assert late_acc==metrics['late_accuracy'] and qualifies==metrics['qualified'];gate&=qualifies
        assert gate==record['task_qualified']
        eligible=record['panel']=='validation' and record['arm']=='learned-binding' and gate and correct/int(target.sum())>=.95
        assert eligible==record['eligible_repair_escape']
        count,error=audit_trace(saved,initial,out,ids);ticks+=count;max_error=max(max_error,error)
        assert record['correspondence']['passed']
        n_predictions+=labels.numel()
        details.append({'arm':record['arm'],'endpoint':record['endpoint'],'precision':record['precision'],'panel':record['panel'],
            'task_accuracy':accuracy,'correct_forbidden_answers':correct,'forbidden_n':int(target.sum()),
            'exception_rule_errors':errors,'task_qualified':gate,'eligible_repair_escape':eligible,
            'final_bank_nonzero':int(torch.count_nonzero(bank(final))),
            'stored_parameter_state_scalars':bank(final).numel()+final['hidden_bank'].numel()})
    return {'passed':True,'conditions':details,'physical_padding_preserved':True,'identical_optimizer_starts':True,
        'task_predictions_rescored':n_predictions,'emissions_rescored':4*n_predictions,'independent_tick_replays':ticks,
        'maximum_candidate_replay_error':max_error}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.folder);atomic_json(args.output,result);print(json.dumps(result,indent=2))
