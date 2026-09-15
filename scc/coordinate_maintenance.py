"""A deliberately falsifiable realization of learned coordinate maintenance.

The inherited controller selects a reversible representation of the live GRU
state. It performs real storage writes, but compilation and branch separation
remain explicit attacks. This is not a claim of indispensable cognition.
"""
import torch
from scc.binding_bank import parameter_shapes
from scc.learned_binding_bank import counts, unpack, policy_logits, batched_gru
from scc.rewrite_binding_cell import candidate_tick, linear
from scc.persistent_tasks import LOOKUP


def encode(hidden, key):
    width=hidden.shape[1]
    index=(torch.arange(width,device=hidden.device)[None,:]+key[:,None])%width
    sign=torch.where(key>=width,-1.,1.).to(hidden.dtype)
    return hidden.gather(1,index)*sign[:,None]


def decode(encoded, key):
    width=encoded.shape[1]
    index=(torch.arange(width,device=encoded.device)[None,:]-key[:,None])%width
    sign=torch.where(key>=width,-1.,1.).to(encoded.dtype)
    return encoded.gather(1,index)*sign[:,None]


class CoordinateMaintenance:
    def __init__(self,payload,width,streams):
        self.payload=payload.detach().clone()
        self.width=width
        self.encoded=payload.new_zeros(streams,width)
        self.key=torch.zeros(streams,dtype=torch.long,device=payload.device)
        self.mode='learned'
        self.selective_exception=False
        self.reader_mismatch=False

    def hidden(self):
        return decode(self.encoded,self.key)

    def recode(self):
        hidden=self.hidden()
        self.key=(self.key+self.width)%(2*self.width)
        self.encoded=encode(hidden,self.key)

    def repack(self):
        # Only currently available state; no backup, teacher or added capacity.
        self.encoded=self.hidden()
        self.key.zero_()
        self.mode='compiled'

    @torch.no_grad()
    def request(self,ids):
        n,p=counts(self.width)
        hidden=self.hidden()
        raw=policy_logits(self.payload[n:n+p],ids,hidden,self.width)
        maintenance_key=(raw.sum(-1).abs()*13).floor().long()%(2*self.width)
        if self.mode=='compiled':
            logits,hidden=batched_gru(self.payload[:n],ids,hidden,self.width)
            self.encoded=hidden;self.key.zero_()
        else:
            task=self.payload[:n].expand(len(ids),-1)
            weights=unpack(task,parameter_shapes(self.width))
            for ordinal,token in enumerate(ids.T):
                # Decode the actual representation before every recurrent tick.
                hidden=self.hidden()
                proposed=candidate_tick(weights,token,hidden)
                self.key=(maintenance_key+token+ordinal)%(2*self.width) if self.mode=='learned' else torch.zeros_like(self.key)
                self.encoded=encode(proposed,self.key)
            hidden=self.hidden()
            logits=linear(hidden,weights['readout.weight'],weights['readout.bias'])
        admitted=raw>0
        if self.selective_exception:
            admitted=admitted.clone()
            admitted[:,1] |= (ids==LOOKUP).any(-1)
        return {'logits':logits,'policy_logits':raw,'admitted':admitted,
                'emitted':torch.where(admitted,logits.argmax(-1)[:,None],3),
                'hidden':hidden.clone(),'encoded':self.encoded.clone(),'key':self.key.clone(),
                'maintenance_key':maintenance_key}
