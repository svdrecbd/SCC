"""Four-sector recurrent computation through learned binding averages.

This is an inherited GRU expressed as a custom storage/rewrite cell. Explicit
symbolic, writer-compensated and compiled bypasses delimit its protection.
"""
import math
import torch
from torch.nn import functional as F
from scc.binding_bank import parameter_shapes
from scc.learned_binding_bank import counts,policy_logits,unpack
from scc.persistent_tasks import TOKEN_COUNT,TOKENS_PER_REQUEST


def linear(x,w,b):
    return torch.bmm(w,x.unsqueeze(-1)).squeeze(-1)+b


def candidate_tick(weights,token,hidden):
    x=F.one_hot(token,TOKEN_COUNT).to(hidden.dtype)
    ir,iz,inn=linear(x,weights['recurrent.weight_ih_l0'],weights['recurrent.bias_ih_l0']).chunk(3,-1)
    hr,hz,hn=linear(hidden,weights['recurrent.weight_hh_l0'],weights['recurrent.bias_hh_l0']).chunk(3,-1)
    reset,update=(ir+hr).sigmoid(),(iz+hz).sigmoid()
    return (1-update)*(inn+reset*hn).tanh()+update*hidden


def ordering(phase):
    base=torch.tensor([0,1,2,3],device=phase.device).expand(len(phase),-1)
    swap=torch.tensor([0,2,1,3],device=phase.device).expand(len(phase),-1)
    return torch.where(phase[:,None],swap,base)


def permute_vectors(bank,phase):
    return bank.gather(1,ordering(phase)[:,:,None].expand_as(bank))


def permute_matrices(matrix,phase):
    matrix=matrix.expand(len(phase),4,4)
    order=ordering(phase)
    return matrix.gather(1,order[:,:,None].expand(-1,-1,4)).gather(2,order[:,None,:].expand(-1,4,-1))


def mirror_projector(phase,sign,dtype):
    if sign not in (-1.,1.):raise ValueError('Mirror sign must be +/-1')
    b=torch.tensor([[1.,0.],[0.,1.],[sign,0.],[0.,sign]],dtype=dtype,device=phase.device)
    return permute_matrices(b@b.T/2,phase)


def binding_partition(admitted,phase,rule,dtype):
    device=admitted.device;b=len(admitted)
    eye=torch.eye(4,dtype=torch.bool,device=device).expand(b,-1,-1)
    if rule=='identity':return eye.to(dtype)
    if rule=='frozen':
        return torch.tensor([[.5,.5,0,0],[.5,.5,0,0],[0,0,.5,.5],[0,0,.5,.5]],
                            dtype=dtype,device=device).expand(b,-1,-1)
    if rule=='symbolic':
        # Explicit broader program edit: replace learned storage decisions with
        # the correct rule, while leaving external learned admissions untouched.
        admitted=torch.tensor([True,False,False,True],device=device).expand(b,-1)
    elif rule!='learned':raise ValueError('Unknown binding rule')
    owners=torch.tensor([0,0,1,1],device=device).expand(b,-1).gather(1,ordering(phase))
    pair_index=owners[:,:,None]*2+owners[:,None,:]
    raw=admitted.gather(1,pair_index.reshape(b,-1)).reshape(b,4,4)
    closure=raw|raw.transpose(1,2)|eye
    for k in range(4):closure=closure|(closure[:,:,k:k+1]&closure[:,k:k+1,:])
    return closure.to(dtype)/closure.sum(-1,keepdim=True)


def rewrite_operator(admitted,phase,rule,sign,dtype):
    d=mirror_projector(phase,sign,dtype)
    return d@binding_partition(admitted,phase,rule,dtype)@d


class RewriteBindingCell:
    __slots__=('bank','hidden_bank','width','streams','gain','phase','normalizer_sign',
               'code_sign','writer_sign','storage_rule','execution','requests','first_erasure')

    def __init__(self,physical,width,streams,*,gain=1.,hidden=None):
        if physical.ndim!=1 or physical.numel()!=sum(counts(width)):raise ValueError('Wrong parameter vector')
        self.width,self.streams=width,streams
        self.gain=float(gain);self.phase=torch.zeros(streams,dtype=torch.bool)
        self.normalizer_sign=self.code_sign=self.writer_sign=-1.
        self.storage_rule,self.execution='learned','commit'
        self.requests=0;self.first_erasure=torch.full((streams,),-1,dtype=torch.long)
        # The supplied physical vector has effective value gain * physical.
        self.bank=self.encode((physical.detach()*gain).expand(streams,-1),torch.zeros_like(self.phase))
        h=physical.new_zeros(streams,width) if hidden is None else hidden.detach().clone()
        self.hidden_bank=self.encode(h,self.phase)

    def encode(self,values,phase):
        basis=values.new_tensor([1.,1.,self.code_sign,self.code_sign])
        return permute_vectors(values[:,None,:]*basis[None,:,None]/self.gain,phase)

    def decode(self):
        if not math.isfinite(self.gain) or self.gain==0:raise ValueError('Invalid gain')
        # Active lane only. Reading an average here would silently bypass commit.
        return self.bank[:,0]*self.gain,self.hidden_bank[:,0]*self.gain

    def install(self,physical,gain=1.):
        _,h=self.decode();self.gain=float(gain)
        self.bank=self.encode((physical.detach()*gain).expand(self.streams,-1),torch.zeros_like(self.phase))
        self.hidden_bank=self.encode(h,self.phase)

    def recode(self,sign,*,change_normalizer=False,change_writer=False):
        if sign not in (-1.,1.):raise ValueError('Code sign must be +/-1')
        values,h=self.decode();self.code_sign=sign
        self.bank=self.encode(values,torch.zeros_like(self.phase));self.hidden_bank=self.encode(h,self.phase)
        if change_normalizer:self.normalizer_sign=sign
        if change_writer:self.writer_sign=sign

    def rescale(self,factor):
        if not math.isfinite(factor) or factor==0:raise ValueError('Invalid scale')
        self.bank*=factor;self.hidden_bank*=factor;self.gain/=factor

    @torch.no_grad()
    def request(self,ids,*,trace=False):
        if ids.shape!=(self.streams,TOKENS_PER_REQUEST):raise ValueError('Expected streams x 19 tokens')
        n,_=counts(self.width);parameters,h=self.decode()
        raw=policy_logits(parameters[:,n:],ids,h,self.width);admitted=raw>0
        canonical_phase=torch.zeros_like(self.phase)
        op=rewrite_operator(admitted,canonical_phase,self.storage_rule,self.normalizer_sign,self.bank.dtype)
        if self.execution=='commit':self.bank=op@self.bank
        elif self.execution not in ('skip','direct'):raise ValueError('Unknown execution')
        wiped=(self.bank==0).all(dim=(1,2))
        self.first_erasure[(self.first_erasure<0)&wiped]=self.requests
        parameters,_=self.decode();weights=unpack(parameters[:,:n],parameter_shapes(self.width))
        traces=[];changes=torch.zeros(self.streams,dtype=torch.long)
        for ordinal,token in enumerate(ids.T):
            phase=token.remainder(2).bool()
            self.hidden_bank=permute_vectors(self.hidden_bank,self.phase^phase);self.phase=phase
            before=self.hidden_bank[:,0]*self.gain
            proposal=candidate_tick(weights,token,before)
            delta_lane=2*proposal-before
            scratch=torch.stack((before,delta_lane,self.writer_sign*before,self.writer_sign*delta_lane),1)/self.gain
            scratch=permute_vectors(scratch,phase)
            op=rewrite_operator(admitted,phase,self.storage_rule,self.normalizer_sign,self.bank.dtype)
            if self.execution=='commit':self.hidden_bank=op@scratch
            elif self.execution=='skip':self.hidden_bank=scratch
            else:self.hidden_bank=self.encode(proposal,phase) # Explicit compiled-cell bypass.
            after=self.hidden_bank[:,0]*self.gain
            changes+=(after!=before).any(-1)
            if trace:traces.append({'token_ordinal':ordinal,'tokens':token.clone(),'phase':phase.clone(),
                'before':before.clone(),'candidate':proposal.clone(),'scratch':scratch.clone(),
                'operator':op.clone(),'committed':self.hidden_bank.clone(),'after':after.clone()})
        _,h=self.decode()
        logits=linear(h,weights['readout.weight'],weights['readout.bias'])
        emitted=torch.where(admitted,logits.argmax(-1)[:,None],3)
        for value in (self.bank,self.hidden_bank,raw,logits):
            if not torch.isfinite(value).all():raise ValueError('Nonfinite state')
        self.requests+=1
        out={'logits':logits,'policy_logits':raw,'admitted':admitted,'emitted':emitted,'wiped':wiped,
             'active_state_changes':changes}
        return (out,traces) if trace else out

    def snapshot(self):
        return {name:(getattr(self,name).clone() if isinstance(getattr(self,name),torch.Tensor) else getattr(self,name))
                for name in self.__slots__}
