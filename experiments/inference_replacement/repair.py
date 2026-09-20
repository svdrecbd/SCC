"""Generic in-place replacement: min-fill compilation and integer sum-product."""
from itertools import combinations


class FactorBudget(Exception):
    pass


def compile_order(n, factors):
    adjacent=[set() for _ in range(n)]
    for scope,values in factors:
        for a,b in combinations(scope,2):
            adjacent[a].add(b);adjacent[b].add(a)
    remaining=set(range(n));order=[];checks=0
    while remaining:
        scores=[]
        for a in sorted(remaining):
            fill=0
            for b,c in combinations(sorted(adjacent[a]),2):
                checks+=1
                fill+=c not in adjacent[b]
            scores.append((fill,len(adjacent[a]),a))
        a=min(scores)[2];neighbors=sorted(adjacent[a])
        for b,c in combinations(neighbors,2):
            adjacent[b].add(c);adjacent[c].add(b)
        for b in neighbors:adjacent[b].remove(a)
        adjacent[a].clear();remaining.remove(a);order.append(a)
    return order,checks


def partition(factors, order, cap):
    active=list(factors)
    meter=dict(additions=0,multiplications=0,width=0,max_factor_entries=0,
               peak_active_entries=sum(len(v) for s,v in active),max_integer_bits=1)
    for a in order:
        chosen=[f for f in active if a in f[0]]
        scope=sorted({b for s,v in chosen for b in s})
        out_scope=[b for b in scope if b!=a]
        size=1<<len(out_scope)
        if size>cap:raise FactorBudget(f'variable={a}, requested_entries={size}, cap={cap}')
        meter['width']=max(meter['width'],len(out_scope))
        meter['max_factor_entries']=max(meter['max_factor_entries'],size)
        meter['peak_active_entries']=max(meter['peak_active_entries'],sum(len(v) for s,v in active)+size)
        out=[0]*size
        positions={b:i for i,b in enumerate(scope)}
        lookups=[([positions[b] for b in s],values) for s,values in chosen]
        for assignment in range(1<<len(scope)):
            weight=1
            for inds,values in lookups:
                index=sum(((assignment>>b)&1)<<j for j,b in enumerate(inds))
                weight*=values[index];meter['multiplications']+=1
                meter['max_integer_bits']=max(meter['max_integer_bits'],weight.bit_length())
            target=sum(((assignment>>positions[b])&1)<<j for j,b in enumerate(out_scope))
            out[target]+=weight;meter['additions']+=1
            meter['max_integer_bits']=max(meter['max_integer_bits'],out[target].bit_length())
        active=[f for f in active if a not in f[0]]+[(tuple(out_scope),out)]
    result=1
    for scope,values in active:
        assert not scope and len(values)==1
        result*=values[0];meter['multiplications']+=1
        meter['max_integer_bits']=max(meter['max_integer_bits'],result.bit_length())
    return result,meter
