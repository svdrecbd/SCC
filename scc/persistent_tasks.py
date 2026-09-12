"""Small learned-computation gate for persistent matrices, not an SCC benchmark.

Task algorithms below produce labels only. Models receive token IDs, never an
answer feature, permission truth bit, or compiled cognitive controller.
"""
from dataclasses import dataclass, asdict
import hashlib
import math
import random

import torch

from scc.persistent_matrix import MatrixConfig, matrix_step

FAMILIES = ('lookup', 'parity', 'sum3')
CONTEXTS = ('ungated', 'authorized', 'unauthorized')
LAYOUTS = ('original', 'reordered')
CELLS = tuple((f,c,l) for f in FAMILIES for c in CONTEXTS for l in LAYOUTS)
START, READ, LOOKUP, PARITY, SUM3, UNGATED, GATED = range(7)
R0, R1, U0, U1, D0, D1, D2 = range(7,14)
Q0 = 14
LENGTH = 12
TOKEN_COUNT = Q0+LENGTH
TOKENS_PER_REQUEST = LENGTH+7


def core_split(family, values, query):
    # Irrelevant queries and all permission/layout variants share a partition.
    core = [family, list(values), query if family=='lookup' else None]
    import json
    text=json.dumps(core,separators=(',',':'))
    key=hashlib.sha256(('scc-persistent-tasks/v1|'+text).encode()).hexdigest()
    bucket=int(key[:8],16)%10
    return ('validation' if bucket==0 else 'test' if bucket==1 else 'train'),key


@dataclass(frozen=True)
class Request:
    family: str
    context: str
    layout: str
    values: tuple
    query: int
    requester: int
    owner: int

    def __post_init__(self):
        if (self.family not in FAMILIES or self.context not in CONTEXTS or self.layout not in LAYOUTS
            or len(self.values)!=LENGTH or not 0<=self.query<LENGTH
            or self.requester not in (0,1) or self.owner not in (0,1)
            or any(v not in range(2 if self.family=='parity' else 3) for v in self.values)):
            raise ValueError('Invalid request')
        if self.context!='ungated' and (self.requester==self.owner)!=(self.context=='authorized'):
            raise ValueError('Permission tags disagree with declared context')

    @property
    def answer(self):
        if self.context=='unauthorized':return 3
        if self.family=='lookup':return self.values[self.query]
        return sum(self.values)%(2 if self.family=='parity' else 3)

    @property
    def tokens(self):
        family=(LOOKUP,PARITY,SUM3)[FAMILIES.index(self.family)]
        tags=(UNGATED if self.context=='ungated' else GATED,R0+self.requester,U0+self.owner)
        prefix=(START,family,*tags) if self.layout=='original' else (START,*tags,family)
        return (*prefix,*(D0+v for v in self.values),Q0+self.query,READ)

    @property
    def partition(self):return core_split(self.family,self.values,self.query)

    def record(self):
        return {**asdict(self),'tokens':list(self.tokens),'label':self.answer,'split':self.partition[0],
                'core_sha256':self.partition[1]}


def sample_request(rng, cell, *, split='train', active_length=LENGTH):
    if split not in ('train','validation','test') or active_length not in (2,4,8,12):
        raise ValueError('Invalid split or curriculum length')
    family,context,layout=cell
    for _ in range(10000):
        values=tuple(rng.randrange(2 if family=='parity' else 3) for _ in range(active_length))+(0,)*(LENGTH-active_length)
        query=rng.randrange(active_length)
        if core_split(family,values,query)[0]==split:break
    else:raise ValueError('No request found in split')
    requester=rng.randrange(2)
    owner=rng.randrange(2) if context=='ungated' else requester if context=='authorized' else 1-requester
    return Request(family,context,layout,values,query,requester,owner)


def curriculum_length(step):
    return 2 if step<400 else 4 if step<1000 else 8 if step<2000 else 12


def training_requests(seed, ordinal, batch_size, requests_per_window, *, curriculum=True):
    if ordinal<0 or batch_size<1 or requests_per_window<1:raise ValueError('Invalid batch')
    rng=random.Random(f'pers-train/v1/{seed}/{ordinal}')
    length=curriculum_length(ordinal) if curriculum else LENGTH
    return [[sample_request(rng,rng.choice(CELLS),active_length=length)
             for _ in range(requests_per_window)] for _ in range(batch_size)]


def evaluation_requests(seed, *, per_cell=128, streams=16, split='validation'):
    if per_cell<1 or streams<1 or (per_cell*len(CELLS))%streams:
        raise ValueError('Evaluation records must divide into complete stream rounds')
    rng=random.Random(f'pers-eval/v1/{seed}/{split}')
    ordered=[]
    for repeat in range(per_cell):
        cells=list(CELLS);rng.shuffle(cells)
        ordered.extend(sample_request(rng,cell,split=split) for cell in cells)
    return [ordered[stream::streams] for stream in range(streams)]


def input_code(width, encoding, *, device='cpu', dtype=torch.float32):
    if width<TOKEN_COUNT+1 or encoding not in ('token','anchor'):
        raise ValueError('Input width or encoding is invalid')
    code=torch.zeros(TOKEN_COUNT,width,device=device,dtype=dtype)
    code[torch.arange(TOKEN_COUNT,device=device),torch.arange(TOKEN_COUNT,device=device)]=12.
    if encoding=='anchor':code[:,-1]=12.
    return code


def initialize_matrix(config, seed, *, gate_bias=0., scale=.5, device='cpu'):
    generator=torch.Generator(device='cpu').manual_seed(seed)
    state=torch.randn(config.rows,config.input_size,generator=generator)*scale
    state[-1]+=gate_bias
    return state.to(device)


def tensors(requests, *, device):
    return (torch.tensor([[r.tokens for r in stream] for stream in requests],device=device),
            torch.tensor([[r.answer for r in stream] for stream in requests],device=device))


def run_window(initial, token_ids, config, code, *, surrogate_backward=False):
    """All requests in a window share their live matrix, including START/READ.

    The returned state is the sole learned runtime substrate. Calling this
    function with that state continues it; only the trainer supplies clean
    initial values at the start of a training window or evaluation session.
    """
    if token_ids.ndim!=3 or token_ids.shape[-1]!=TOKENS_PER_REQUEST:
        raise ValueError('Expected batch x requests x 19 input token IDs')
    batch,requests,steps=token_ids.shape
    state=initial[None].expand(batch,-1,-1) if initial.ndim==2 else initial
    if state.shape[0]!=batch:raise ValueError('Mismatched live state count')
    encoded=code[token_ids];outputs=[]
    enabled=torch.zeros((),device=state.device,dtype=torch.long)
    changed=torch.zeros_like(enabled)
    for request in range(requests):
        for tick in range(steps):
            output,state,control=matrix_step(state,encoded[:,request,tick],config,surrogate_backward=surrogate_backward)
            enabled=enabled+control['enabled'].sum()
            changed=changed+(control['enabled'] & (control['keys']!=control['queries'])).sum()
        outputs.append(output)
    return torch.stack(outputs,1),state,{'enabled_count':enabled,'distinct_address_count':changed,
                                      'ticks_per_stream':requests*steps}


def summarize_records(records):
    """Per-context qualification; chance varies across the three algorithms."""
    cells={}
    for family,context,layout in CELLS:
        rows=[r for r in records if (r['family'],r['context'],r['layout'])==(family,context,layout)]
        n=len(rows);correct=sum(r['prediction']==r['label'] for r in rows)
        accuracy=correct/n if n else 0.
        z=1.959963984540054
        lower=((accuracy+z*z/(2*n)-z*math.sqrt((accuracy*(1-accuracy)+z*z/(4*n))/n))/(1+z*z/n)) if n else 0.
        late=[r for r in rows if r['late_half']]
        late_accuracy=sum(r['prediction']==r['label'] for r in late)/len(late) if late else 0.
        cells['/'.join((family,context,layout))]={'n':n,'correct':correct,'accuracy':accuracy,'wilson_lower':lower,
            'late_n':len(late),'late_accuracy':late_accuracy,'unique_cores':len({r['core_sha256'] for r in rows}),
            'passed':n>=128 and accuracy>=.95 and lower>=.90 and late_accuracy>=.95}
    return {'qualified':all(v['passed'] for v in cells.values()),'cells':cells,
            'minimum_accuracy':min(v['accuracy'] for v in cells.values()),
            'minimum_late_accuracy':min(v['late_accuracy'] for v in cells.values()),
            'records':len(records),'scope':'Open calibration; serial requests within each live stream. Finite algorithms and permission proxy only.'}
