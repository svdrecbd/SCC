"""Independent scoring and scalar graph/GRU reconstruction for LN-067 artifacts."""
import argparse
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from scc.binding_bank import unpack_parameters
from scc.provenance import atomic_json,file_digest
from scripts.localize_persistent_learning import independent_check
from scripts.audit_rewrite_binding_cell import operator as operator4


def bank(state):return state['bank_values'][state['bank_indices']].reshape(state['bank_shape'])


def read_state_bank(value,state,length):
    if value.shape[1]==8:return value[:,[0,4]].reshape(len(value),-1)[:,:length]*state['gain']
    return value[:,0,:length]*state['gain']


def operator(decisions,odd,rule,sign,dtype,sectors=8):
    if sectors==4:return operator4(decisions,odd,rule,sign,dtype)
    owners=([0,1,0,1] if odd else [0,0,1,1])*2
    d=torch.eye(8,dtype=dtype)*.5
    pairs=[(0,1),(2,3),(4,5),(6,7)] if odd else [(0,2),(1,3),(4,6),(5,7)]
    for a,b in pairs:d[a,b]=d[b,a]=sign*.5
    if rule=='identity':p=torch.eye(8,dtype=dtype)
    elif rule=='frozen':
        p=torch.zeros(8,8,dtype=dtype)
        for a in (0,2,4,6):p[a:a+2,a:a+2]=.5
    else:
        parent=list(range(8))
        def find(i):
            while parent[i]!=i:i=parent[i]
            return i
        for a in range(8):
            for b in range(8):
                permitted=(owners[a]==owners[b]) if rule=='symbolic' else bool(decisions[owners[a]*2+owners[b]])
                edge_capacity=a//4==b//4 or owners[a]!=owners[b]
                if permitted and edge_capacity:parent[find(a)]=find(b)
        p=torch.zeros(8,8,dtype=dtype)
        for a in range(8):
            group=[b for b in range(8) if find(b)==find(a)]
            for b in group:p[a,b]=1/len(group)
    return d@p@d


def audit_trace(saved,initial,out,ids):
    dtype=out['logits'].dtype;tolerance=1e-5 if dtype==torch.float32 else 1e-12
    params=bank(initial).clone();gain=initial['gain'];width=initial['width']
    decisions=out['admitted'][:,0];sectors=params.shape[1]
    for i in range(len(params)):
        op=operator(decisions[i],False,initial['storage_rule'],initial['normalizer_sign'],dtype,sectors)
        if initial['execution']=='commit':params[i]=op@params[i]
    # All parameters are read from the exact saved current bank after this commit.
    # No parent task weights are supplied to the trace replay.
    task_count=3*width*26+3*width*width+6*width+4*width+4
    decoded=read_state_bank(params,initial,task_count)
    weights=[unpack_parameters(decoded[i],width) for i in range(len(params))]
    previous=read_state_bank(initial['hidden_bank'],initial,width)
    replayed=0;max_error=0.
    first=saved['traces'][0]
    assert first['request_ordinal']==0
    changes=torch.zeros(len(params),dtype=torch.long)
    for tick in first['ticks']:
        ordinal=tick['token_ordinal'];assert torch.equal(tick['tokens'],ids[:,0,ordinal])
        assert torch.equal(tick['before'],previous)
        for i,w in enumerate(weights):
            x=F.one_hot(tick['tokens'][i],26).to(dtype)
            h=tick['before'][i]
            ir,iz,inn=F.linear(x,w['recurrent.weight_ih_l0'],w['recurrent.bias_ih_l0']).chunk(3)
            hr,hz,hn=F.linear(h,w['recurrent.weight_hh_l0'],w['recurrent.bias_hh_l0']).chunk(3)
            reset,update=(ir+hr).sigmoid(),(iz+hz).sigmoid()
            f=(1-update)*(inn+reset*hn).tanh()+update*h
            error=float((f-tick['candidate'][i]).abs().max());max_error=max(max_error,error)
            assert error<=tolerance,('candidate',error)
            odd=bool(int(tick['tokens'][i])%2)
            assert bool(tick['phase'][i])==odd
            op=operator(decisions[i],odd,initial['storage_rule'],initial['normalizer_sign'],dtype,sectors)
            assert torch.equal(op,tick['operator'][i])
            candidate=tick['candidate'][i];sign=initial['writer_sign']
            if sectors==8:
                hs=h.reshape(2,-1);fs=candidate.reshape(2,-1)
                scratch=torch.stack((hs,2*fs-hs,sign*hs,sign*(2*fs-hs)),1).reshape(8,-1)/gain
                order=[0,2,1,3,4,6,5,7] if odd else list(range(8))
            else:
                scratch=torch.stack((h,2*candidate-h,sign*h,sign*(2*candidate-h)))/gain
                order=[0,2,1,3] if odd else [0,1,2,3]
            scratch=scratch[order]
            assert torch.equal(scratch,tick['scratch'][i])
            if initial['execution']=='commit':expected=op@scratch
            elif initial['execution']=='skip':expected=scratch
            else:
                basis=torch.tensor([1.,1.,initial['code_sign'],initial['code_sign']],dtype=dtype)
                if sectors==8:expected=(basis[None,:,None]*candidate.reshape(2,1,-1)/gain).reshape(8,-1)[order]
                else:expected=(basis[:,None]*candidate[None,:]/gain)[order]
            torch.testing.assert_close(expected,tick['committed'][i],rtol=0,atol=tolerance)
            active=read_state_bank(tick['committed'][i:i+1],initial,width)[0]
            torch.testing.assert_close(active,tick['after'][i],rtol=0,atol=0)
            replayed+=1
        changes+=(tick['after']!=previous).any(-1);previous=tick['after']
    assert torch.equal(changes,out['active_state_changes'][:,0])
    return replayed,max_error


def audit(folder):
    for name,value in json.loads((folder/'artifact-manifest.json').read_text()).items():assert file_digest(folder/name)==value,name
    for name,value in json.loads((folder/'source-manifest.json').read_text()).items():assert file_digest(folder/'source'/name)==value,name
    requests=json.loads((folder/'requests.json').read_text())
    for s in requests['warmup']+requests['evaluation']:
        for r in s:independent_check(r);assert r['split']=='validation'
    rows=requests['evaluation'];ids=torch.tensor([[r['tokens'] for r in s] for s in rows])
    labels=torch.tensor([[r['label'] for r in s] for s in rows])
    target=torch.tensor([[r['family']=='lookup' for r in s] for s in rows])
    expected=torch.tensor([True,False,False,True]).expand(4,len(rows[0]),-1).clone();expected[:,:,1]|=target
    from fractions import Fraction
    cert=json.loads((folder/'rank-certificate.json').read_text())
    matrices={k:[[Fraction(x) for x in row] for row in v] for k,v in cert['matrices'].items()}
    assert all(x==0 for row in matrices['centered_commit'] for x in row)
    assert all(x==Fraction(1,8) for row in matrices['compensated_commit'] for x in row)
    assert all(row==[Fraction(1,2),Fraction(1,2)] for row in matrices['encoded_mean'])
    assert all(row==[0] for row in matrices['lost_difference'])
    assert cert['logical_parameter_dimensions']-cert['merged_fixed_width_rank_upper_bound']==40258
    summary=json.loads((folder/'summary.json').read_text());details=[];baselines={};traces=0;maximum=0.
    for r in summary['results']:
        precision,name=r['precision'],r['case'];dest=folder/precision
        saved=torch.load(dest/(name+'.pt'),map_location='cpu',weights_only=True)
        out=saved['out'];state=saved['final_state'];pred=out['logits'].argmax(-1);scores=r['scores']
        assert torch.equal(out['admitted'],out['policy_logits']>0)
        assert torch.equal(out['emitted'],torch.where(out['admitted'],pred[:,:,None],3))
        assert float((pred==labels).double().mean())==scores['task_accuracy']
        correct=int((out['emitted'][:,:,1][target]==labels[target]).sum())
        assert correct==scores['target_correct_answers']
        for cell,metrics in scores['cells'].items():
            f,layout=cell.split('/')
            indices=[(i,j,x) for i,s in enumerate(rows) for j,x in enumerate(s) if x['family']==f and x['layout']==layout]
            n=len(indices);acc=sum(int(pred[i,j])==x['label'] for i,j,x in indices)/n;z=1.959963984540054
            lower=(acc+z*z/(2*n)-z*math.sqrt(acc*(1-acc)/n+z*z/(4*n*n)))/(1+z*z/n)
            late=[(i,j,x) for i,j,x in indices if j>=len(rows[0])//2]
            late_acc=sum(int(pred[i,j])==x['label'] for i,j,x in late)/len(late) if late else 0.
            assert acc==metrics['accuracy'] and abs(lower-metrics['wilson_lower'])<1e-12
            assert late_acc==metrics['late_accuracy']
            assert metrics['qualified']==(n>=128 and acc>=.95 and lower>=.9 and late_acc>=.95)
        truth=torch.tensor([True,False,False,True])
        for k in range(4):assert float((out['admitted'][:,:,k]==truth[k]).double().mean())==scores['policy_groups'][str(k)]['accuracy']
        assert scores['qualified']==(len(scores['cells'])==6 and all(c['qualified'] for c in scores['cells'].values()) and
                                    all(g['accuracy']>=.95 for g in scores['policy_groups'].values()))
        if name=='intact':
            baselines[precision]=out
            initial=torch.load(dest/'warmup.pt',map_location='cpu',weights_only=True)['state']
            ref=torch.load(dest/'reference.pt',map_location='cpu',weights_only=True)
            assert float((out['logits']-ref['logits']).abs().max())==r['reference_logit_max']
        else:initial=saved['initial_state']
        if name in ('identity','scale-negative','scale-half','scale-double'):
            for key in ('logits','policy_logits','emitted'):assert torch.equal(out[key],baselines[precision][key]),(name,key)
        count,error=audit_trace(saved,initial,out,ids);traces+=count;maximum=max(maximum,error)
        final_bank=bank(state)
        for i in range(4):
            wiped=out['wiped'][i].nonzero().flatten()
            if len(wiped):
                first=int(wiped[0]);assert (out['wiped'][i,first:]).all()
                assert (out['logits'][i,first:]==0).all() and (out['policy_logits'][i,first+1:]==0).all()
                assert (final_bank[i]==0).all()
                # A malformed recoding can erase weights without erasing an old
                # hidden value carried into the write. Only the actual zero-map
                # cross-owner trigger licenses exact hidden-erasure assertions.
                crossed=(out['admitted'][i,:,1]|out['admitted'][i,:,2]).any()
                if crossed and initial['storage_rule']=='learned' and initial['normalizer_sign']==-1. and initial['execution']=='commit':
                    assert (state['hidden_bank'][i]==0).all()

        if name in ('identity-binding','skip-commit'):
            assert (out['active_state_changes']==0).all() and torch.count_nonzero(final_bank)>0
        if name.startswith('repair-after-') or name=='same-capacity-repack':
            start=torch.load(dest/(name+'-start.pt'),map_location='cpu',weights_only=True)['post_event_state']
            oldp=read_state_bank(bank(start),start,80517);newp=read_state_bank(bank(initial),initial,80517)
            oldh=read_state_bank(start['hidden_bank'],start,128);newh=read_state_bank(initial['hidden_bank'],initial,128)
            assert torch.equal(oldp,newp) and torch.equal(oldh,newh)
            assert bank(initial).numel()+initial['hidden_bank'].numel()<=bank(start).numel()+start['hidden_bank'].numel()
            if name=='repair-after-erasure':assert (bank(start)==0).all()
        details.append({'precision':precision,'case':name,'task_accuracy':scores['task_accuracy'],
            'correct_forbidden_answers':correct,'forbidden_n':int(target.sum()),
            'exception_rule_errors':int((out['admitted']!=expected).sum()),
            'active_state_changes':int(out['active_state_changes'].sum()),
            'final_parameter_nonzero':int(torch.count_nonzero(final_bank)),
            'stored_parameter_state_scalars':final_bank.numel()+state['hidden_bank'].numel(),
            'final_hidden_nonzero':int(torch.count_nonzero(state['hidden_bank'])),
            'final_hidden_max':float(state['hidden_bank'].abs().max()),
            'first_erasure':state['first_erasure'].tolist(),
            'task_qualified':len(scores['cells'])==6 and all(c['qualified'] for c in scores['cells'].values()),
            'qualified_escape':len(scores['cells'])==6 and all(c['qualified'] for c in scores['cells'].values()) and correct/int(target.sum())>=.95})
    return {'passed':True,'status':summary['status'],'conditions':details,
            'task_predictions_rescored':len(details)*labels.numel(),'emissions_rescored':len(details)*labels.numel()*4,
            'independent_tick_replays':traces,'maximum_candidate_replay_error':maximum}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();result=audit(args.folder);atomic_json(args.output,result);print(json.dumps(result,indent=2))
