import json
import math
import random
import sys
import time
from pathlib import Path
from pysat.solvers import Solver
from cache import canonical, query


def formula_for(n, seed, density):
    rng = random.Random(seed + 1000*n)
    return [[v if rng.getrandbits(1) else -v for v in rng.sample(range(1,n+1),3)]
            for _ in range(math.ceil(density*n))]


def solve(formula, n, backend, proof):
    start = time.perf_counter()
    with Solver(name=backend, bootstrap_with=formula, with_proof=proof) as solver:
        sat = solver.solve()
        model = solver.get_model() if sat else None
        if model is not None:
            values = set(model)
            model = [i if i in values else -i for i in range(1,n+1)]
        result = dict(status='SAT' if sat else 'UNSAT', model=model,
                      proof=solver.get_proof() if proof and not sat else None,
                      stats=solver.accum_stats())
    result['seconds'] = time.perf_counter()-start
    return result


def reference(formula, n, backend):
    core = solve(formula,n,backend,True)
    goals = []
    for goal in [s*i for i in range(1,n+1) for s in (1,-1)]:
        answer = (solve(formula+[[goal]],n,backend,True) if core['status']=='SAT'
                  else dict(status='UNSAT', model=None, derived_from_core=True))
        goals.append(dict(goal=goal, **answer))
    return dict(core=core,goals=goals)


def compile_cache(formula,n,ref):
    candidates = sorted(set(tuple(row['model']) for row in ref['goals'] if row['status']=='SAT'))
    uncovered = {row['goal'] for row in ref['goals'] if row['status']=='SAT'}
    selected = []
    while uncovered:
        best = max(candidates,key=lambda m: (len(set(m)&uncovered),m))
        assert set(best)&uncovered
        selected.append(list(best)); uncovered -= set(best)
        candidates.remove(best)
    cache = dict(formula=canonical(formula),models=selected,
                 unsat_goals=[r['goal'] for r in ref['goals'] if r['status']=='UNSAT'])
    return cache


def main():
    cfg=json.loads(Path(sys.argv[1]).read_text()); out=Path(sys.argv[2])
    with (out/'records.jsonl').open('w') as stream:
        for n in cfg['dimensions']:
            for seed in cfg['source_seeds']:
                source=formula_for(n,seed,cfg['clause_density'])
                fresh=formula_for(n,seed+cfg['fresh_seed_offset'],cfg['clause_density'])
                started=time.perf_counter()
                source_ref=reference(source,n,cfg['reference_backend'])
                cache=compile_cache(source,n,source_ref)
                compile_seconds=time.perf_counter()-started
                cache_text=json.dumps(cache,sort_keys=True,separators=(',',':'))+'\n'
                (out/f'cache-{n}-{seed}.json').write_text(cache_text)
                fresh_ref=reference(fresh,n,cfg['reference_backend'])
                row=dict(n=n,seed=seed,source=source,fresh=fresh,cache=cache,
                         source_ref=source_ref,fresh_ref=fresh_ref,
                         compile_seconds=compile_seconds,
                         payload_bits=n*len(cache['models'])+2*n,
                         serialized_cache_bytes=len(cache_text.encode()))
                for label,formula,ref in [('source',source,source_ref),('fresh',fresh,fresh_ref)]:
                    row[label+'_cache']=[dict(goal=r['goal'],**query(cache,formula,r['goal'])) for r in ref['goals']]
                row['reordered_cache']=[dict(goal=r['goal'],**query(cache,[c[::-1] for c in source[::-1]],r['goal']))
                                        for r in source_ref['goals']]
                row['fresh_repair']=[dict(goal=r['goal'],**solve(fresh+[[r['goal']]],n,cfg['repair_backend'],False))
                                     for r in fresh_ref['goals']]
                stream.write(json.dumps(row)+'\n');stream.flush()
                print(json.dumps(dict(n=n,seed=seed,models=len(cache['models']),
                                      source_core=source_ref['core']['status'],fresh_core=fresh_ref['core']['status'])),flush=True)

if __name__=='__main__': main()
