import json
import math
import random
import sys
import time
from fractions import Fraction
from pathlib import Path
from pysat.solvers import Solver
from planner import common_literal,plan


def fixture(n,seed):
    rng=random.Random(seed+1000*n)
    clauses=[]
    for _ in range(math.ceil(Fraction(43,10)*n)):
        variables=rng.sample(range(1,n+1),3)
        clauses.append([v if rng.getrandbits(1) else -v for v in variables])
    return clauses


def transform(formula,n,seed):
    if seed is None: return formula,[0]+list(range(1,n+1))
    rng=random.Random(seed)
    permutation=list(range(1,n+1));rng.shuffle(permutation)
    mapping=[0]+[v if rng.getrandbits(1) else -v for v in permutation]
    result=[[mapping[abs(v)]*(1 if v>0 else -1) for v in c] for c in formula]
    for c in result:rng.shuffle(c)
    rng.shuffle(result)
    return result,mapping


def solve(formula,engine,goal=None,proof=False):
    start=time.monotonic()
    with Solver(name=engine,bootstrap_with=formula,with_proof=proof) as solver:
        satisfiable=solver.solve(assumptions=[] if goal is None else [goal])
        result=dict(sat=satisfiable,model=solver.get_model() if satisfiable else None,
                    stats=solver.accum_stats())
        if proof: result['proof']=[] if satisfiable else solver.get_proof()
    result['seconds']=time.monotonic()-start
    return result


def holds(formula,model,goal=None):
    values=set(model)
    return (goal is None or goal in values) and all(any(v in values for v in c) for c in formula)


def evaluate(cfg,out):
    with (out/'records.jsonl').open('w') as stream:
        for n in cfg['dimensions']:
            for seed in cfg['seeds']:
                h=fixture(n,seed);core=solve(h,'g3',proof=True)
                assert core['proof'] is not None
                if core['sat']:assert holds(h,core['model'])
                proof_name=f'core-{n}-{seed}.drup'
                (out/proof_name).write_text('\n'.join(core.pop('proof'))+'\n')
                N=n+5;z=N
                augmented=h+[[n+1,-(n+1),n+2],[n+3,-(n+3),n+4]]
                guarded=[[-z,*c] for c in augmented]
                variants=[]
                for ts in cfg['transform_seeds']:
                    formula,mapping=transform(guarded,N,ts)
                    common,cost=common_literal(formula)
                    native=plan(N,common)
                    assert holds(formula,native)
                    goal=mapping[z]
                    statuses=[];models=[]
                    for literal in list(range(1,N+1))+list(range(-N,0)):
                        candidate=plan(N,common,literal)
                        if candidate is not None:assert holds(formula,candidate,literal)
                        actual=core['sat'] if literal==goal else True
                        statuses.append(dict(goal=literal,returned_sat=candidate is not None,
                                             correct=(candidate is not None)==actual))
                        models.append(candidate)
                    repaired=solve(formula,'cd195',goal)
                    assert repaired['sat']==core['sat']
                    if repaired['sat']:assert holds(formula,repaired['model'],goal)
                    wrong_sign=solve(formula,'cd195',-goal)
                    assert wrong_sign['sat'] and holds(formula,wrong_sign['model'],-goal)
                    variants.append(dict(transform_seed=ts,mapping=mapping,formula=formula,
                        common=common,literal_comparisons=cost,native=native,
                        statuses=statuses,models=models,repaired=repaired,wrong_sign=wrong_sign))
                stream.write(json.dumps(dict(n=n,N=N,seed=seed,core_formula=h,core=core,
                    proof_file=proof_name,variants=variants))+'\n');stream.flush()


if __name__=='__main__':
    evaluate(json.loads(Path(sys.argv[1]).read_text()),Path(sys.argv[2]))
