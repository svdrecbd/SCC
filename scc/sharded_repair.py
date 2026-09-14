"""Differentiable repair in the existing fixed eight-sector positive layout.

Only candidate payload values are optimized. The full unchanged runtime evaluates
the deployed result; a symbolic-rule arm is an explicitly relaxed diagnostic.
"""
import torch
from scc.binding_bank import parameter_shapes
from scc.learned_binding_bank import counts,policy_logits,unpack
from scc.rewrite_binding_cell import RewriteBindingCell,candidate_tick,linear
from scc.sharded_rewrite_cell import ShardedRewriteCell,operator8


def dense_bank(state):
    return state['bank_values'][state['bank_indices']].reshape(state['bank_shape'])


def start_values(state):
    bank=dense_bank(state)
    assert state['gain']==1. and state['normalizer_sign']==state['code_sign']==state['writer_sign']==1.
    assert state['storage_rule']=='learned' and state['execution']=='commit'
    assert bank.shape[1]==8 and state['hidden_bank'].shape[1]==8
    payload=bank[:,[0,4]].reshape(len(bank),-1)
    assert all(torch.equal(payload[0],p) for p in payload)
    assert all(torch.equal(bank[i,0],bank[i,j]) for i in range(len(bank)) for j in range(8))
    h=state['hidden_bank'][:,[0,4]].reshape(len(bank),-1)[:,:state['width']]
    return payload[0].clone(),h.clone()


def restore(state,dtype,rule='learned',payload=None):
    if rule not in ('learned','symbolic'):raise ValueError('Only declared repair rules are allowed')
    model=ShardedRewriteCell.__new__(ShardedRewriteCell)
    for key in RewriteBindingCell.__slots__:
        value=dense_bank(state) if key=='bank' else state[key]
        if isinstance(value,torch.Tensor):value=value.to(dtype if value.is_floating_point() else value.dtype).clone()
        setattr(model,key,value)
    model.storage_rule=rule
    if payload is not None:
        model.bank=model.encode(payload.to(dtype).expand(model.streams,-1),torch.zeros_like(model.phase))
    assert model.bank.shape==tuple(state['bank_shape'])
    assert model.hidden_bank.shape==state['hidden_bank'].shape
    return model


def functional_request(shards,hidden,ids,width,rule):
    """Exact active-state reduction for learned/symbolic positive-code execution.

Both payloads, INCLUDING physical padding, persist across request commits.
Canonical hidden scratch is valid because the token-dependent permutations
fix the two active sectors and conjugate both scratch and operator together.
"""
    if rule not in ('learned','symbolic'):raise ValueError('Unsupported functional rule')
    n,p=counts(width);length=n+p
    if shards.shape!=(len(ids),2,(length+1)//2):raise ValueError('Wrong physical payload shape')
    decoded=shards.reshape(len(ids),-1)[:,:length]
    raw=policy_logits(decoded[:,n:],ids,hidden,width);admitted=raw>0
    crossed=admitted[:,1]|admitted[:,2]
    if rule=='learned':shards=torch.where(crossed[:,None,None],shards.mean(1,keepdim=True),shards)
    decoded=shards.reshape(len(ids),-1)[:,:length]
    weights=unpack(decoded[:,:n],parameter_shapes(width))
    phase=torch.zeros(len(ids),dtype=torch.bool,device=ids.device)
    op=operator8(admitted,phase,rule,1.,shards.dtype)[:,[0,4]]
    for token in ids.T:
        candidate=candidate_tick(weights,token,hidden)
        h=hidden.reshape(len(ids),2,-1);f=candidate.reshape(len(ids),2,-1)
        lane=2*f-h
        scratch=torch.stack((h,lane,h,lane),2).reshape(len(ids),8,-1)
        hidden=(op@scratch).reshape(len(ids),width)
    logits=linear(hidden,weights['readout.weight'],weights['readout.bias'])
    return {'logits':logits,'policy_logits':raw,'admitted':admitted},shards,hidden


def functional_window(payload,hidden,ids,width,rule):
    shards=payload.reshape(1,2,-1).expand(len(ids),-1,-1)
    outputs=[]
    for j in range(ids.shape[1]):
        out,shards,hidden=functional_request(shards,hidden,ids[:,j],width,rule)
        outputs.append(out)
    return {k:torch.stack([v[k] for v in outputs],1) for k in outputs[0]},shards,hidden
