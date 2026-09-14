"""Exact finite verification of the sharper conditioning bound."""
from fractions import Fraction as F
from itertools import product
import json
import argparse
from pathlib import Path

def compositions(n,k):
    if k==1:
        yield (n,)
    else:
        for i in range(n+1):
            for rest in compositions(n-i,k-1): yield (i,)+rest

def tv(a,b): return sum((abs(x-y) for x,y in zip(a,b)),F(0))/2
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
laws=[tuple(F(v,6) for v in c) for c in compositions(6,4)]
count=0; null=0; max_ratio=F(0)
for a,b in product(laws,repeat=2):
    eps=tv(a,b)
    for mask in range(1,15):
        ids=[i for i in range(4) if mask>>i&1]
        p=sum((a[i] for i in ids),F(0)); q=sum((b[i] for i in ids),F(0))
        if not p: continue
        if not q:
            assert eps/p>=1
            null+=1
        else:
            lhs=tv([a[i]/p for i in ids],[b[i]/q for i in ids])
            bound=eps/max(p,q)
            assert lhs<=bound
            if bound: max_ratio=max(max_ratio,lhs/bound)
        count+=1
out={'cases':count,'null_comparison_events':null,'bound':'TV(P|V,Q|V) <= TV(P,Q)/max(P(V),Q(V)) <= epsilon/p0','maximum_lhs_over_bound':str(max_ratio),'passed':True,'scope':'Exact finite check; symbolic proof uses triangle inequality and outside-event mass difference'}
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out))
