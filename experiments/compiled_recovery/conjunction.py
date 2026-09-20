import copy
import hashlib
import itertools
import json
import os
import platform
import random
import resource
import subprocess
import sys
import time
from pathlib import Path
import pysat
from evaluate import solve
from audit import valid
from proof import check_unsat
from cache_conjunction import query


def goals_for(n,seed,k,count):
    if n==12 and k==2:
        return [[a*sa,b*sb] for a,b in itertools.combinations(range(1,n+1),2) for sa,sb in itertools.product((1,-1),repeat=2)]
    rng=random.Random(seed+10000*n+k);goals=set()
    while len(goals)<count:
        variables=rng.sample(range(1,n+1),k)
        g=tuple(sorted((v if rng.getrandbits(1) else -v for v in variables),key=abs))
        goals.add(g)
    return [list(g) for g in sorted(goals)]


def check_answer(f,n,g,answer,reference):
    assert answer['status'] in ('SAT','UNSAT','UNKNOWN')
    if answer['status']=='SAT':
        assert reference['status']=='SAT' and valid(f,answer['model'],n)
        assert all(v in answer['model'] for v in g)
    elif answer['status']=='UNSAT':
        assert reference['status']=='UNSAT' and answer['model'] is None
    else:
        assert answer['model'] is None


def controls():
    cache=dict(formula=[[-1,-2]],models=[[1,-2],[-1,2]],unsat_goals=[])
    assert query(cache,[[-1,-2]],[1,2])['status']=='UNKNOWN'
    assert query(cache,[[-1,-2]],[1])['status']=='SAT'
    assert query(cache,[[-1,-2]],[2])['status']=='SAT'
    cache2=dict(formula=[[1]],models=[[1,2]],unsat_goals=[-1])
    assert query(cache2,[[1]],[-1,2])['status']=='UNSAT'
    assert query(cache2,[[-1]],[-1,2])['status']=='UNKNOWN'
    ref=dict(status='UNSAT')
    corruptions=[]
    for name,ans in [('partial_witness',dict(status='SAT',model=[1,-2])),
                     ('unsupported_negative',dict(status='UNSAT',model=None)),
                     ('unknown_with_model',dict(status='UNKNOWN',model=[1,-2]))]:
        if name=='unsupported_negative': f=[[-1,-2]];g=[1];reference=dict(status='SAT')
        else:f=[[-1,-2]];g=[1,2];reference=ref
        try:check_answer(f,2,g,ans,reference)
        except AssertionError:corruptions.append(name)
        else:raise AssertionError('corruption accepted '+name)
    check_unsat([[-1,-2],[1],[2]],['0'])
    return dict(rejected=corruptions,joint_infeasibility='PASS',cross_context='PASS')


def main(src,out):
    assert platform.node()=='charon' and (src/'plan-frozen.md').exists()
    cfg=json.loads((src/'conjunction.json').read_text());out.mkdir(exist_ok=False)
    start=time.monotonic()
    (out/'machine.json').write_text(json.dumps(dict(host=platform.node(),platform=platform.platform(),python=sys.version,
        pysat=pysat.__version__,affinity=sorted(os.sched_getaffinity(0)),executable=sys.executable),indent=2)+'\n')
    parent_path=src.parent/'input/records.jsonl'
    parent=[json.loads(l) for l in parent_path.read_text().splitlines()]
    assert [(r['n'],r['seed']) for r in parent]==[(n,s) for n in cfg['dimensions'] for s in cfg['source_seeds']]
    tasks=[]
    for row in parent:
        goals=sum([goals_for(row['n'],row['seed'],k,cfg['sample_count']) for k in cfg['arities']],[])
        tasks.append(dict(cache=row['cache'],formula=row['source'],goals=goals))
    request=json.dumps(tasks)
    (out/'reader-input.json').write_text(request+'\n')
    reader_start=time.monotonic()
    process=subprocess.run([sys.executable,'-S',str(src/'cache_conjunction.py')],input=request,text=True,capture_output=True,check=True,timeout=20)
    reader_seconds=time.monotonic()-reader_start
    (out/'reader-output.json').write_text(process.stdout)
    (out/'reader.stderr').write_text(process.stderr)
    answers=json.loads(process.stdout);assert len(answers)==len(tasks)
    summary=[];proofs=additions=visits=brute_assignments=0
    with (out/'records.jsonl').open('w') as stream:
        for row,task,cached in zip(parent,tasks,answers):
            n=row['n'];f=row['source'];assert len(cached)==len(task['goals'])
            core=row['source_ref']['core']
            if core['status']=='UNSAT':
                m=check_unsat(f,core['proof']);proofs+=1;additions+=m['additions'];visits+=m['clause_visits']
            else:assert valid(f,core['model'],n)
            exact=None
            if n==12:
                exact=[]
                for bits in itertools.product((False,True),repeat=n):
                    model=[i if b else -i for i,b in enumerate(bits,1)]
                    if valid(f,model,n):exact.append(set(model))
                    brute_assignments+=1
            totals={k:dict(n=n,seed=row['seed'],arity=k,core=core['status'],cases=0,feasible=0,
                           cached_sat=0,cached_unsat=0,unknown_sat=0,unknown_unsat=0,repair_seconds=0,repair_conflicts=0)
                    for k in cfg['arities']}
            for goals,answer in zip(task['goals'],cached):
                augmented=f+[[g] for g in goals]
                reference=(solve(augmented,n,'g3',True) if core['status']=='SAT'
                           else dict(status='UNSAT',model=None,derived_from_core=True))
                if reference['status']=='SAT':
                    assert valid(augmented,reference['model'],n)
                elif not reference.get('derived_from_core'):
                    m=check_unsat(augmented,reference['proof']);proofs+=1;additions+=m['additions'];visits+=m['clause_visits']
                if exact is not None:
                    assert (reference['status']=='SAT')==any(set(goals)<=m for m in exact)
                check_answer(f,n,goals,answer,reference)
                assert answer==query(task['cache'],f,goals)
                repair=solve(augmented,n,'cd195',False)
                assert repair['status']==reference['status']
                check_answer(f,n,goals,repair,reference)
                t=totals[len(goals)];t['cases']+=1;t['feasible']+=reference['status']=='SAT'
                if answer['status']=='UNKNOWN':t['unknown_'+reference['status'].lower()]+=1
                else:t['cached_'+answer['status'].lower()]+=1
                t['repair_seconds']+=repair['seconds'];t['repair_conflicts']+=repair['stats']['conflicts']
                stream.write(json.dumps(dict(n=n,seed=row['seed'],goals=goals,cache=answer,reference=reference,repair=repair))+'\n')
            summary.extend(totals.values());stream.flush()
    negative=controls()
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (out/'controls.json').write_text(json.dumps(negative,indent=2)+'\n')
    size=sum(p.stat().st_size for p in out.iterdir() if p.is_file());assert size<cfg['output_limit_bytes']
    receipt=dict(validation='PASS',cases=sum(r['cases'] for r in summary),
        cached_sat=sum(r['cached_sat'] for r in summary),cached_unsat=sum(r['cached_unsat'] for r in summary),
        unknown_sat=sum(r['unknown_sat'] for r in summary),unknown_unsat=sum(r['unknown_unsat'] for r in summary),
        drup_proofs=proofs,proof_additions=additions,clause_visits=visits,brute_assignments=brute_assignments,
        reader_isolation='separate Python -S process; only cache, formula and goals supplied',
        reader_seconds=reader_seconds,elapsed_seconds=time.monotonic()-start,output_bytes=size,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        reader_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        parent_records_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest())
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    paths=[p for folder in (src,src.parent/'input',out) for p in folder.iterdir() if p.is_file()]
    (out/'sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':
    src=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve()
    try:main(src,out)
    except Exception as error:
        out.mkdir(exist_ok=True);(out/'failure.json').write_text(json.dumps(dict(error=repr(error)))+'\n');raise
