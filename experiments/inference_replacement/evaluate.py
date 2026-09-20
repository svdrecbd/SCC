"""Registered synthetic inference replacement, exact recovery and error controls."""
from fractions import Fraction as F
from itertools import combinations,product
import json
from pathlib import Path
import sys
from repair import compile_order,partition,FactorBudget


def model(kind,n,seed,t):
    if kind=='chain':edges=[(i,i+1) for i in range(n-1)]
    elif kind=='star':edges=[(0,i) for i in range(1,n)]
    elif kind=='clique':edges=list(combinations(range(n),2))
    else:
        assert kind=='ladder' and n%2==0
        edges=[(i,i+1) for i in range(0,n,2)]+[(i,i+2) for i in range(n-2)]
    unary=[[1+(seed+3*i)%5,1+(2*seed+5*i)%7] for i in range(n)]
    for step in range(t):
        i=(seed+5*step)%n;y=(seed+step)%2
        unary[i]=[w*(3 if a==y else 1) for a,w in enumerate(unary[i])]
    factors=[((i,),u) for i,u in enumerate(unary)]
    for u,v in edges:
        factors.append(((u,v),[1+(seed+u+2*v+3*(bits&1)+(bits>>1))%5 for bits in range(4)]))
    return factors


def case(kind,n,seed,t,cfg):
    factors=model(kind,n,seed,t)
    order,checks=compile_order(n,factors);evaluations={}
    def calc(name,extra):
        result,meter=partition(factors+extra,order,cfg['factor_entry_cap'])
        evaluations[name]=dict(value=result,meter=meter)
        return result
    z=calc('base',[])
    numerators=[calc(f'node{i}',[((i,),[0,1])]) for i in range(n)]
    joint=calc('joint',[((0,),[0,1]),((n-1,),[0,1])])
    plus=calc('sensor',[((0,),[1,3])])
    plus_last=calc('sensor_last',[((0,),[1,3]),((n-1,),[0,1])])
    a,b,c=F(numerators[0],z),F(numerators[-1],z),F(plus_last,plus)
    adapter=((1+2*a)*c-b)/2
    perturb=[]
    clip=lambda v:max(F(0),min(F(1),v))
    for eps in map(F,cfg['perturbations']):
        estimates=[clip(((1+2*clip(a+s0*eps))*clip(c+s2*eps)-clip(b+s1*eps))/2)
                   for s0,s1,s2 in product((-1,1),repeat=3)]
        perturb.append(dict(epsilon=str(eps),max_error=str(max(abs(x-F(joint,z)) for x in estimates)),
                            correct_labels=sum((x>F(1,4))==(F(joint,z)>F(1,4)) for x in estimates)))
    bad=None
    if kind=='star':
        try:
            value,meter=partition(factors,list(range(n)),cfg['factor_entry_cap'])
            bad=dict(status='complete',value=value,meter=meter)
        except FactorBudget as error:
            bad=dict(status='declared_cap',detail=str(error))
    return dict(case=f'{kind}/{n}/{seed}/{t}',kind=kind,n=n,seed=seed,evidence=t,
                factors=factors,order=order,compile_pair_checks=checks,input_entries=sum(len(v) for s,v in factors),
                evidence_multiplications=2*t,evaluations=evaluations,adapter=str(adapter),
                protected_label=adapter>F(1,4),perturbations=perturb,bad_order=bad,
                constant_forecast_mean_absolute_error=str(sum(abs(F(x,z)-F(1,2)) for x in numerators)/n))


if __name__=='__main__':
    cfg=json.loads(Path(sys.argv[1]).read_text())
    with Path(sys.argv[2]).open('x') as f:
        for (kind,n),seed,t in product(cfg['models'],cfg['seeds'],cfg['evidence_prefixes']):
            f.write(json.dumps(case(kind,n,seed,t,cfg))+'\n');f.flush()
