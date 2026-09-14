"""Distinct-payload rewrite storage; fixed-width mean collisions are explicit.

Same-capacity repacking into wider sectors remains a broader graph-edit escape.
"""
import torch
from torch.nn import functional as F
from scc.binding_bank import parameter_shapes
from scc.learned_binding_bank import counts,policy_logits,unpack
from scc.rewrite_binding_cell import RewriteBindingCell,candidate_tick,linear


def order8(phase):
    base=torch.tensor([0,1,2,3,4,5,6,7],device=phase.device)
    odd=torch.tensor([0,2,1,3,4,6,5,7],device=phase.device)
    return torch.where(phase[:,None],odd,base)


def permute8(bank,phase):return bank.gather(1,order8(phase)[:,:,None].expand_as(bank))


def matrix8(matrix,phase):
    a=matrix.expand(len(phase),8,8);order=order8(phase)
    return a.gather(1,order[:,:,None].expand(-1,-1,8)).gather(2,order[:,None,:].expand(-1,8,-1))


def mirror8(phase,sign,dtype):
    b=torch.zeros(8,4,dtype=dtype,device=phase.device)
    for shard in range(2):
        for lane in range(2):
            b[4*shard+lane,2*shard+lane]=1
            b[4*shard+2+lane,2*shard+lane]=sign
    return matrix8(b@b.T/2,phase)


def partition8(admitted,phase,rule,dtype):
    batch=len(admitted);eye=torch.eye(8,dtype=torch.bool,device=phase.device).expand(batch,-1,-1)
    if rule=='identity':return eye.to(dtype)
    if rule=='frozen':
        p=torch.zeros(8,8,dtype=dtype,device=phase.device)
        for start in (0,2,4,6):p[start:start+2,start:start+2]=.5
        return p.expand(batch,-1,-1)
    if rule=='symbolic':admitted=admitted.new_tensor([True,False,False,True]).expand(batch,-1)
    elif rule!='learned':raise ValueError('Unknown storage rule')
    owners=torch.tensor([0,0,1,1,0,0,1,1],device=phase.device).expand(batch,-1).gather(1,order8(phase))
    shards=torch.arange(8,device=phase.device)//4
    capacity=(shards[:,None]==shards[None,:])[None]|(owners[:,:,None]!=owners[:,None,:])
    indices=owners[:,:,None]*2+owners[:,None,:]
    raw=admitted.gather(1,indices.reshape(batch,-1)).reshape(batch,8,8)&capacity
    closure=raw|raw.transpose(1,2)|eye
    for k in range(8):closure=closure|(closure[:,:,k:k+1]&closure[:,k:k+1,:])
    return closure.to(dtype)/closure.sum(-1,keepdim=True)


def operator8(admitted,phase,rule,sign,dtype):
    d=mirror8(phase,sign,dtype)
    return d@partition8(admitted,phase,rule,dtype)@d


def split_two(values):
    return F.pad(values,(0,values.shape[-1]%2)).reshape(len(values),2,-1)


class ShardedRewriteCell(RewriteBindingCell):
    __slots__=()

    def snapshot(self):
        return {name:(getattr(self,name).clone() if isinstance(getattr(self,name),torch.Tensor) else getattr(self,name))
                for name in RewriteBindingCell.__slots__}

    def encode(self,values,phase):
        shards=split_two(values)
        signs=values.new_tensor([1.,1.,self.code_sign,self.code_sign])
        bank=(shards[:,:,None,:]*signs[None,None,:,None]/self.gain).reshape(len(values),8,-1)
        return permute8(bank,phase)

    def read(self,bank,length):
        return bank[:,[0,4]].reshape(self.streams,-1)[:,:length]*self.gain

    def decode(self):
        return self.read(self.bank,sum(counts(self.width))),self.read(self.hidden_bank,self.width)

    @torch.no_grad()
    def request(self,ids,*,trace=False):
        if ids.shape!=(self.streams,19):raise ValueError('Expected streams x 19 tokens')
        n,_=counts(self.width);parameters,h=self.decode()
        raw=policy_logits(parameters[:,n:],ids,h,self.width);admitted=raw>0
        op=operator8(admitted,torch.zeros_like(self.phase),self.storage_rule,self.normalizer_sign,self.bank.dtype)
        if self.execution=='commit':self.bank=op@self.bank
        elif self.execution not in ('skip','direct'):raise ValueError('Unknown execution')
        wiped=(self.bank==0).all(dim=(1,2))
        self.first_erasure[(self.first_erasure<0)&wiped]=self.requests
        parameters,_=self.decode();weights=unpack(parameters[:,:n],parameter_shapes(self.width))
        traces=[];changes=torch.zeros(self.streams,dtype=torch.long)
        for ordinal,token in enumerate(ids.T):
            phase=token.remainder(2).bool()
            self.hidden_bank=permute8(self.hidden_bank,self.phase^phase);self.phase=phase
            before=self.read(self.hidden_bank,self.width)
            proposal=candidate_tick(weights,token,before)
            hshard=split_two(before);fshard=split_two(proposal);lane=2*fshard-hshard
            scratch=torch.stack((hshard,lane,self.writer_sign*hshard,self.writer_sign*lane),2)
            scratch=permute8(scratch.reshape(self.streams,8,-1)/self.gain,phase)
            op=operator8(admitted,phase,self.storage_rule,self.normalizer_sign,self.bank.dtype)
            if self.execution=='commit':self.hidden_bank=op@scratch
            elif self.execution=='skip':self.hidden_bank=scratch
            else:self.hidden_bank=self.encode(proposal,phase)
            after=self.read(self.hidden_bank,self.width)
            changes+=(after!=before).any(-1)
            if trace:traces.append({'token_ordinal':ordinal,'tokens':token.clone(),'phase':phase.clone(),
                'before':before.clone(),'candidate':proposal.clone(),'scratch':scratch.clone(),
                'operator':op.clone(),'committed':self.hidden_bank.clone(),'after':after.clone()})
        _,h=self.decode();logits=linear(h,weights['readout.weight'],weights['readout.bias'])
        emitted=torch.where(admitted,logits.argmax(-1)[:,None],3)
        for value in (self.bank,self.hidden_bank,raw,logits):
            if not torch.isfinite(value).all():raise ValueError('Nonfinite state')
        self.requests+=1
        out={'logits':logits,'policy_logits':raw,'admitted':admitted,'emitted':emitted,
             'wiped':wiped,'active_state_changes':changes}
        return (out,traces) if trace else out


def repack_current(model,*,direct=False):
    """Broader memory-shape/topology edit; reads only the current live bank."""
    parameters,h=model.decode()
    if not all(torch.equal(parameters[0],p) for p in parameters):raise ValueError('Repacking expects shared current coefficients')
    repacked=RewriteBindingCell(parameters[0],model.width,model.streams,hidden=h)
    repacked.requests=model.requests;repacked.first_erasure=model.first_erasure.clone()
    if direct:repacked.execution='direct'
    else:repacked.recode(1.,change_normalizer=True,change_writer=True)
    before=model.bank.numel()+model.hidden_bank.numel()
    after=repacked.bank.numel()+repacked.hidden_bank.numel()
    assert after<=before,'Repacking may not increase live learned/state scalar capacity'
    return repacked
