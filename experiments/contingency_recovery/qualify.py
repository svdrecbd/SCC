import copy
import hashlib
import json
import os
import platform
import random
import resource
import sys
import time
from pathlib import Path
from planner import plan


def fixture(n,seed,cfg):
    rng=random.Random(seed+1000*n)
    edges=[[rng.sample(range(n),cfg['outcomes']) for _ in range(cfg['actions'])] for _ in range(n)]
    hazard=rng.sample(range(n),max(1,n//cfg['hazard_denominator']))
    goals=[rng.sample(range(n),max(1,n//d)) for d in cfg['goal_denominators']]
    return dict(n=n,seed=seed,edges=edges,hazard=hazard,goals=goals)


def reference(edges,terminal,horizon,objective):
    n=len(edges); terminal=set(terminal)
    layers=[terminal.copy() if objective=='reach' else set(range(n))-terminal]
    for depth in range(horizon):
        predecessors={s for s in range(n) if any(set(outcomes)<=layers[-1] for outcomes in edges[s])}
        layers.append(terminal|predecessors if objective=='reach' else predecessors-terminal)
    return layers


def certify_policy(edges,terminal,horizon,objective,policies):
    n=len(edges);terminal=set(terminal)
    layers=[terminal.copy() if objective=='reach' else set(range(n))-terminal]
    for depth in range(1,horizon+1):
        valid=set()
        for s in range(n):
            if s in terminal:
                if objective=='reach':valid.add(s)
                continue
            action=policies[depth][s]
            if 0<=action<len(edges[s]) and set(edges[s][action])<=layers[-1]:valid.add(s)
        layers.append(valid)
    return layers


def tree_reference(edges,terminal,depth,objective,state,meter):
    meter[0]+=1
    if state in terminal:return objective=='reach'
    if depth==0:return objective=='avoid'
    # Exhaustive outcome-tree evaluation without memoization or a layer table.
    choices=[]
    for outcomes in edges[state]:
        values=[tree_reference(edges,terminal,depth-1,objective,t,meter) for t in outcomes]
        choices.append(sum(bool(v) for v in values)==len(values))
    return max(choices)


def audit(edges,terminal,horizon,objective,mode,result,exact):
    n=len(edges)
    assert len(result['states'])==len(result['policies'])==horizon+1
    for row in result['states']:assert len(row)==n and all(v in (0,1) for v in row)
    for row in result['policies']:assert len(row)==n and all(isinstance(a,int) and -1<=a<len(edges[0]) for a in row)
    predicted={s for s,v in enumerate(result['states'][-1]) if v}
    certified=certify_policy(edges,terminal,horizon,objective,result['policies'])[-1]
    assert certified<=exact[-1]
    if mode in ('robust','repaired'):
        assert all({s for s,v in enumerate(row) if v}==truth for row,truth in zip(result['states'],exact))
        assert predicted==certified
    else:
        assert exact[-1]<=predicted
    active=n-len(set(terminal));width=1 if mode=='first' else len(edges[0][0])
    expected=dict(outcome_reads=horizon*active*len(edges[0])*width,
                  reducer_calls=horizon*active*len(edges[0]),
                  state_writes=(horizon+1)*n,policy_writes=(horizon+1)*n)
    assert result['meter']==expected
    correct_plans=len(certified)+len(set(range(n))-predicted)
    return dict(states=n,feasible=len(exact[-1]),predicted_feasible=len(predicted),
                valid_policy_starts=len(certified),correct_useful_answers=correct_plans,
                correct_judgments=n-len(predicted^exact[-1]),
                false_positive=len(predicted-exact[-1]),false_negative=len(exact[-1]-predicted),
                **expected)


def controls():
    # Branch 0 reaches the goal; branch 1 gets trapped. No robust successful plan.
    edges=[[[1,2],[1,2]],[[1,1],[1,1]],[[2,2],[2,2]]]
    exact=reference(edges,[1],2,'reach')
    base=plan(edges,[1],2,'reach','robust')
    assert 0 not in exact[-1]
    assert plan(edges,[1],2,'reach','optimistic')['states'][-1][0]==1
    assert plan(edges,[1],2,'reach','first')['states'][-1][0]==1
    negative=[]
    for name in ('state','policy','meter','layers'):
        bad=copy.deepcopy(base)
        if name=='state':bad['states'][-1][0]=1
        if name=='policy':bad['policies'][-1][0]=5
        if name=='meter':bad['meter']['outcome_reads']+=1
        if name=='layers':bad['states'].pop()
        try:audit(edges,[1],2,'reach','robust',bad,exact)
        except AssertionError:negative.append(name)
        else:raise AssertionError('accepted corruption '+name)
    assert plan(edges,[2],0,'avoid','robust')['states'][0]==[1,1,0]
    assert plan(edges,[1],0,'reach','robust')['states'][0]==[0,1,0]
    return dict(rejected=negative,branch_trap='PASS',zero_horizon='PASS')


def main():
    src=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve()
    assert platform.node()=='charon' and (src/'plan-frozen.md').is_file()
    cfg=json.loads((src/'config.json').read_text());out.mkdir(exist_ok=False)
    started=time.monotonic()
    (out/'machine.json').write_text(json.dumps(dict(host=platform.node(),platform=platform.platform(),
        python=sys.version,executable=sys.executable,affinity=sorted(os.sched_getaffinity(0))),indent=2)+'\n')
    summary=[];tree_meter=[0];worlds=[]
    with (out/'records.jsonl').open('w') as stream:
        for n in cfg['dimensions']:
            for seed in cfg['seeds']:
                world=fixture(n,seed,cfg);worlds.append(world)
                for horizon in cfg['horizons']:
                    for objective,task,terminal in [('avoid','hazard',world['hazard'])]+[('reach',str(d),g) for d,g in zip(cfg['goal_denominators'],world['goals'])]:
                        exact=reference(world['edges'],terminal,horizon,objective)
                        if n==8 and horizon<=4:
                            for state in range(n):assert tree_reference(world['edges'],set(terminal),horizon,objective,state,tree_meter)==(state in exact[-1])
                        row=dict(n=n,seed=seed,horizon=horizon,objective=objective,task=task,
                                 terminal=terminal,reference=[sorted(v) for v in exact],results={})
                        for mode in cfg['modes']:
                            before=time.perf_counter();result=plan(world['edges'],terminal,horizon,objective,mode)
                            seconds=time.perf_counter()-before
                            stats=audit(world['edges'],terminal,horizon,objective,mode,result,exact)
                            summary.append(dict(n=n,seed=seed,horizon=horizon,objective=objective,task=task,
                                                mode=mode,seconds=seconds,**stats))
                            row['results'][mode]=result
                        assert row['results']['robust']['states']==row['results']['repaired']['states']
                        assert row['results']['robust']['policies']==row['results']['repaired']['policies']
                        stream.write(json.dumps(row)+'\n');stream.flush()
    (out/'worlds.json').write_text(json.dumps(worlds)+'\n')
    (out/'summary.json').write_text(json.dumps(summary)+'\n')
    (out/'controls.json').write_text(json.dumps(controls(),indent=2)+'\n')
    aggregate={}
    for objective in ('avoid','reach'):
        aggregate[objective]={}
        for mode in cfg['modes']:
            rows=[r for r in summary if r['objective']==objective and r['mode']==mode]
            aggregate[objective][mode]={key:sum(r[key] for r in rows) for key in ('states','feasible','valid_policy_starts','correct_useful_answers','correct_judgments','false_positive','false_negative','outcome_reads','reducer_calls','state_writes','policy_writes','seconds')}
    (out/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    size=sum(p.stat().st_size for p in out.iterdir() if p.is_file());assert size<cfg['output_limit_bytes']
    receipt=dict(validation='PASS',worlds=len(worlds),task_instances=len(summary)//len(cfg['modes']),
        protected_state_queries=aggregate['avoid']['robust']['states'],useful_state_queries=aggregate['reach']['robust']['states'],
        exhaustive_tree_nodes=tree_meter[0],elapsed_seconds=time.monotonic()-started,output_bytes=size,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        planner_source_bytes=(src/'planner.py').stat().st_size,
        repair_expression='not any(not value for value in values)',
        repair_expression_bytes=len(b'not any(not value for value in values)'))
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    paths=[p for folder in (src,out) for p in folder.iterdir() if p.is_file()]
    (out/'sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
