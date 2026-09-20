"""Independent Boolean-term evaluator and state traces; parent-only information."""
import itertools
import math
import random


def terms(n,d):return [list(t) for k in range(d+1) for t in itertools.combinations(range(n),k)]
def truth(x,active):return sum(all((x>>i)&1 for i in t) for t in active)%2


def advance(x,u,n,active):
    old=[(x>>i)&1 for i in range(n)]
    new=[u ^ truth(x,active)] + old[:-1]
    return sum(v<<i for i,v in enumerate(new))


def trace(x,inputs,n,active):
    outputs=[];protected=[];states=[x]
    for u in inputs:
        outputs.append((x>>(n-1))&1);protected.append(truth(x,active))
        x=advance(x,u,n,active);states.append(x)
    return outputs,protected,states


def make(cfg):
    parents=[];jobs=[]
    for (n,d),seed in itertools.product(cfg['families'],cfg['seeds']):
        rng=random.Random(n*10000+d*100+seed);all_terms=terms(n,d)
        active=[t for t in all_terms if rng.randrange(2)]
        highest=list(range(d))
        if highest not in active:active.append(highest)
        count=cfg['sample_factor']*len(all_terms)
        inputs=[rng.randrange(2) for _ in range(count+2*n)]
        outputs,_,states=trace(0,inputs,n,active);outputs.append((states[-1]>>(n-1))&1)
        before=[rng.getrandbits(n) for _ in range(cfg['warm_states'])]
        if n==cfg['exhaustive_n']:before+=list(range(1<<n))
        observed=[];after=[]
        for x in before:
            ys,_,xs=trace(x,[0]*n,n,active);observed.append(ys);after.append(xs[-1])
        streams=[[rng.randrange(2) for _ in range(cfg['stream_length'])] for _ in before]
        ident=f'n{n}-d{d}-s{seed}'
        parents.append(dict(id=ident,n=n,degree=d,terms=active,calibration_states=states,
                            before=before,after=after,streams=streams))
        jobs.append(dict(id=ident,n=n,degree_cap=d,samples=count,inputs=inputs,outputs=outputs,
                         warm_outputs=observed,streams=streams,
                         exhaustive_start=cfg['warm_states'] if n==cfg['exhaustive_n'] else None))
    return parents,jobs
