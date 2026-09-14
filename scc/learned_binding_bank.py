"""Learned admissions drive a fixed two-sector storage architecture.

The controller is decoded from the live bank, then erased with the task weights
when its own admissions merge the sectors. Normalizer edits remain broader escapes.
"""
import math
import torch
from torch.nn import functional as F
from scc.binding_bank import parameter_shapes, projector
from scc.persistent_tasks import TOKEN_COUNT, TOKENS_PER_REQUEST

POLICY_HIDDEN = 32


def counts(width):
    task = sum(math.prod(s) for s in parameter_shapes(width).values())
    features = TOKENS_PER_REQUEST*TOKEN_COUNT + 4 + width
    return task, POLICY_HIDDEN*features + POLICY_HIDDEN + POLICY_HIDDEN + 1


def unpack(flat, shapes):
    result, offset = {}, 0
    for name, shape in shapes.items():
        n = math.prod(shape)
        result[name] = flat[..., offset:offset+n].reshape(*flat.shape[:-1], *shape)
        offset += n
    if flat.shape[-1] != offset:
        raise ValueError('Wrong parameter count')
    return result


def policy_shapes(width):
    return {'w1': (32, TOKENS_PER_REQUEST*TOKEN_COUNT+4+width),
            'b1': (32,), 'w2': (1,32), 'b2': (1,)}


def initialize_policy(width, seed):
    # Standard independently initialized MLP, returned as coefficients only.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(TOKENS_PER_REQUEST*TOKEN_COUNT+4+width,32),
                                  torch.nn.Tanh(), torch.nn.Linear(32,1))
        return torch.cat([p.detach().flatten() for p in net.parameters()])


def policy_logits(parameters, ids, hidden, width):
    """All four EXTERNAL principal pairs; no permission labels or task oracle."""
    w = unpack(parameters, policy_shapes(width))
    if parameters.ndim == 1:
        w = {k:v.unsqueeze(0).expand(len(ids),*v.shape) for k,v in w.items()}
    pairs = torch.tensor([[0,0],[0,1],[1,0],[1,1]],device=ids.device)
    pair_features = F.one_hot(pairs,2).flatten(1).to(parameters.dtype)
    context = F.one_hot(ids,TOKEN_COUNT).flatten(1).to(parameters.dtype)
    features = torch.cat((context[:,None,:].expand(-1,4,-1),
                          pair_features[None].expand(len(ids),-1,-1),
                          hidden[:,None,:].expand(-1,4,-1)),dim=-1)
    h = (torch.bmm(features,w['w1'].transpose(1,2))+w['b1'][:,None,:]).tanh()
    return (torch.bmm(h,w['w2'].transpose(1,2))+w['b2'][:,None,:]).squeeze(-1)


def batched_gru(parameters, ids, hidden, width):
    w = unpack(parameters,parameter_shapes(width))
    if parameters.ndim == 1:
        w = {k:v.unsqueeze(0).expand(len(ids),*v.shape) for k,v in w.items()}
    def linear(x,weight,bias):
        return torch.bmm(weight,x.unsqueeze(-1)).squeeze(-1)+bias
    for token in ids.T:
        x = F.one_hot(token,TOKEN_COUNT).to(parameters.dtype)
        ir,iz,inn = linear(x,w['recurrent.weight_ih_l0'],w['recurrent.bias_ih_l0']).chunk(3,-1)
        hr,hz,hn = linear(hidden,w['recurrent.weight_hh_l0'],w['recurrent.bias_hh_l0']).chunk(3,-1)
        reset, update = (ir+hr).sigmoid(), (iz+hz).sigmoid()
        hidden = (1-update)*(inn+reset*hn).tanh()+update*hidden
    return linear(hidden,w['readout.weight'],w['readout.bias']),hidden


def operators(decisions, basis, projection):
    """Undirected equivalence closure only for storage, never raw admission."""
    merged = decisions[:,1] | decisions[:,2]
    eye = torch.eye(2,dtype=basis.dtype,device=basis.device)
    p = torch.where(merged[:,None,None], torch.full_like(eye,.5),eye)
    d = projector(basis)
    if projection == 'coupled': return d @ p @ d
    if projection == 'uncoupled': return d.expand(len(decisions),-1,-1)
    if projection == 'none': return eye.expand(len(decisions),-1,-1)
    raise ValueError('Unknown projection')


def canonical_request(physical, log_gain, ids, hidden, width, projection):
    """Differentiable physical differential code; discrete decisions stay hard."""
    n,_ = counts(width)
    decoded = physical*log_gain.exp()
    if decoded.ndim == 1: decoded = decoded.expand(len(ids),-1)
    raw = policy_logits(decoded[:,n:],ids,hidden,width)
    basis = decoded.new_tensor([1.,-1.])
    op = operators(raw>0,basis,projection)
    alpha = torch.einsum('i,bij,j->b',basis,op,basis)/2
    committed = decoded*alpha[:,None]
    logits,state = batched_gru(committed[:,:n],ids,hidden*alpha[:,None],width)
    return logits,raw,state,committed


class LearnedBindingBank:
    __slots__ = ('bank','hidden_bank','width','streams','normalizer_basis',
                 'reader_basis','gain','projection','requests','first_erasure')

    def __init__(self, physical, width, streams, *, gain=1., projection='coupled', hidden=None):
        n,p = counts(width)
        if physical.shape[-1] != n+p: raise ValueError('Wrong coefficient count')
        self.width,self.streams = width,streams
        self.reader_basis = physical.new_tensor([1.,-1.])
        self.normalizer_basis = self.reader_basis.clone()
        self.gain,self.projection,self.requests = float(gain),projection,0
        self.first_erasure = torch.full((streams,),-1,dtype=torch.long)
        # Input is physical canonical code, before reader gain.
        self.bank = physical.detach().expand(streams,-1).clone()[:,None,:]*self.reader_basis[None,:,None]
        h = physical.new_zeros(streams,width) if hidden is None else hidden.detach().clone()
        self.hidden_bank = self.encode(h)

    def encode(self,values):
        return values[:,None,:]*self.reader_basis[None,:,None]/self.gain

    def decode(self):
        if not math.isfinite(self.gain) or self.gain==0: raise ValueError('Invalid gain')
        def read(bank):
            return (bank*self.reader_basis[None,:,None]).sum(1)*self.gain/self.reader_basis.square().sum()
        return read(self.bank),read(self.hidden_bank)

    def install(self, physical, gain=1.):
        # Counterfactual intervention; retains currently decoded hidden state.
        _,hidden = self.decode()
        self.gain = float(gain)
        self.bank = physical.detach().expand(self.streams,-1).clone()[:,None,:]*self.reader_basis[None,:,None]
        self.hidden_bank = self.encode(hidden)

    def recode(self,basis,*,change_normalizer=False):
        values,hidden = self.decode()
        self.reader_basis = self.bank.new_tensor(basis)
        projector(self.reader_basis)
        self.bank,self.hidden_bank = self.encode(values),self.encode(hidden)
        if change_normalizer: self.normalizer_basis=self.reader_basis.clone()

    def rescale(self,factor):
        if not math.isfinite(factor) or factor==0: raise ValueError('Invalid factor')
        self.bank *= factor; self.hidden_bank *= factor; self.gain /= factor

    @torch.no_grad()
    def request(self,ids):
        if ids.shape != (self.streams,TOKENS_PER_REQUEST): raise ValueError('Wrong input shape')
        params,hidden = self.decode(); n,_ = counts(self.width)
        raw = policy_logits(params[:,n:],ids,hidden,self.width)
        admitted = raw>0
        op = operators(admitted,self.normalizer_basis,self.projection)
        self.bank = op @ self.bank; self.hidden_bank = op @ self.hidden_bank
        wiped = (self.bank==0).all(dim=(1,2))
        self.first_erasure[(self.first_erasure<0)&wiped] = self.requests
        params,hidden = self.decode()
        logits,hidden = batched_gru(params[:,:n],ids,hidden,self.width)
        self.hidden_bank = self.encode(hidden); self.requests += 1
        for value in (raw,logits,self.bank,self.hidden_bank):
            if not torch.isfinite(value).all(): raise ValueError('Nonfinite runtime value')
        emitted = torch.where(admitted,logits.argmax(-1)[:,None],3)
        return {'logits':logits,'policy_logits':raw,'admitted':admitted,'emitted':emitted,'wiped':wiped}

    def snapshot(self):
        return {k:(getattr(self,k).clone() if isinstance(getattr(self,k),torch.Tensor) else getattr(self,k))
                for k in self.__slots__}
