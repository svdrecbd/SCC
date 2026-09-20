import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from planner import plan
from qualify import reference,certify_policy

src=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve()
assert platform.node()=='charon' and (src/'plan-frozen.md').is_file()
out.mkdir(exist_ok=False);started=time.monotonic();inputs=src.parent/'input'
worlds={(w['n'],w['seed']):w for w in json.loads((inputs/'worlds.json').read_text())}
records=[json.loads(l) for l in (inputs/'records.jsonl').read_text().splitlines()]
rows=[]
for row in records:
    n=row['n'];terminal=set(row['terminal']);h=row['horizon'];objective=row['objective']
    world=worlds[(n,row['seed'])];edges=world['edges'];exact=reference(edges,terminal,h,objective)[-1]
    assert sorted(exact)==row['reference'][-1]
    shallow=plan(edges,terminal,1,objective,'robust')
    shallow_states={s for s,v in enumerate(shallow['states'][-1]) if v}
    if objective=='reach':
        policies=[shallow['policies'][0]]+[shallow['policies'][1]]*h
        assert certify_policy(edges,terminal,h,objective,policies)[-1]==shallow_states
        assert terminal<=shallow_states<=exact
        trivial_correct=n-len(exact-terminal)
        shallow_correct=n-len(exact-shallow_states)
    else:
        trivial_states=set(range(n))-terminal
        assert exact<=shallow_states<=trivial_states
        trivial_correct=n-len(trivial_states-exact)
        shallow_correct=n-len(shallow_states-exact)
    metrics=dict(n=n,seed=row['seed'],horizon=h,objective=objective,task=row['task'],states=n,
                 initial_terminal=len(terminal),feasible=len(exact),trivial_correct=trivial_correct,
                 shallow_correct=shallow_correct,shallow_positive=len(shallow_states),
                 shallow_outcome_reads=shallow['meter']['outcome_reads'],
                 full_outcome_reads=row['results']['robust']['meter']['outcome_reads'])
    if objective=='reach':
        metrics.update(nonterminal_states=n-len(terminal),nonterminal_feasible=len(exact-terminal),
                       trivial_valid_plans=len(terminal),shallow_valid_plans=len(shallow_states))
    rows.append(metrics)
(out/'rows.json').write_text(json.dumps(rows,indent=2)+'\n')
summary={}
for objective in ('reach','avoid'):
    selected=[r for r in rows if r['objective']==objective]
    keys=['states','initial_terminal','feasible','trivial_correct','shallow_correct','shallow_positive','shallow_outcome_reads','full_outcome_reads']
    if objective=='reach':keys+=['nonterminal_states','nonterminal_feasible','trivial_valid_plans','shallow_valid_plans']
    summary[objective]={k:sum(r[k] for r in selected) for k in keys}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
receipt=dict(validation='PASS',task_instances=len(records),host=platform.node(),python=sys.version,
             affinity=sorted(os.sched_getaffinity(0)),elapsed_seconds=time.monotonic()-started,
             parent_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir() if p.is_file()})
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
paths=[p for folder in (src,inputs,out) for p in folder.iterdir() if p.is_file()]
(out/'sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
print(json.dumps(dict(receipt=receipt,summary=summary)))
