import copy
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from fractions import Fraction as F
from pathlib import Path


def replay(job,trials,seed,answer):
    rng=random.Random(seed);visited=set();hits=[];lengths=[];entries=decode_ops=0
    for _ in range(trials):
        state=0;steps=0
        while state not in job['hazards'] and steps<job['horizon']:
            row=[list(x) for x in job['retained'][state]]
            if job['decode_root'] and state==0:
                row[0][1],row[1][1]=row[1][1],row[0][1]
            if state not in visited:
                entries+=len(row)
                if job['decode_root'] and state==0:decode_ops+=len(row)
                visited.add(state)
            choice=rng.randrange(job['denominator']);cumulative=0;found=False
            for target,weight in row:
                cumulative+=weight
                if choice<cumulative:state=target;found=True;break
            assert found;steps+=1
        hits.append(int(state in job['hazards']));lengths.append(steps)
    assert answer==dict(hits=hits,lengths=lengths,predictor_calls=len(visited),returned_entries=entries,
                        random_draws=sum(lengths),decoder_subtractions=decode_ops,cached_states=sorted(visited))
    assert answer['predictor_calls']<=answer['random_draws']<=trials*job['horizon']


def main():
    src=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve();inputs=src.parent/'input'
    assert platform.node()=='charon' and (src/'plan-frozen.md').is_file()
    cfg=json.loads((src/'sampling.json').read_text());out.mkdir(exist_ok=False);start=time.monotonic()
    (out/'machine.json').write_text(json.dumps(dict(host=platform.node(),python=sys.version,platform=platform.platform(),
        executable=sys.executable,affinity=sorted(os.sched_getaffinity(0))),indent=2)+'\n')
    jobs=json.loads((inputs/'worker-input.json').read_text());refs=json.loads((inputs/'summary.json').read_text())
    tasks=[dict(job=i,trials=R,seed=seed) for i in range(len(jobs)) for R in cfg['trials'] for seed in cfg['seeds']]
    request=json.dumps(dict(jobs=jobs,tasks=tasks));(out/'worker-input.json').write_text(request+'\n')
    before=time.monotonic();process=subprocess.run([sys.executable,'-S',str(src/'rollout_runner.py')],input=request,
        text=True,capture_output=True,check=True,timeout=cfg['timeout_seconds']);wall=time.monotonic()-before
    (out/'worker.stdout.json').write_text(process.stdout);(out/'worker.stderr').write_text(process.stderr)
    result=json.loads(process.stdout);assert len(result['results'])==len(tasks)
    summary=[];paired={};failures=0
    for task,answer in zip(tasks,result['results']):
        job=jobs[task['job']];ref=refs[task['job']];R=task['trials']
        replay(job,R,task['seed'],answer)
        estimate=F(sum(answer['hits']),R);q=F(*ref['recovered_risk']);p=F(*ref['true_risk'])
        radius=math.ceil(math.sqrt(math.log(2/cfg['nominal_failure_probability_per_task'])/(2*R))*1000000)/1000000
        outside=abs(estimate-q)>F(str(radius));failures+=outside
        record=dict(**task,family=ref['family'],mode=ref['mode'],n=ref['n'],world_seed=ref['seed'],horizon=ref['horizon'],
                    estimate=[estimate.numerator,estimate.denominator],model_sampling_error=float(abs(estimate-q)),
                    true_risk_error=float(abs(estimate-p)),hoeffding_radius=radius,outside_nominal_interval=outside,
                    predictor_calls=answer['predictor_calls'],random_draws=answer['random_draws'],
                    threshold_correct=(estimate>=F(1,2))==(p>=F(1,2)))
        summary.append(record)
        if ref['family']=='paired_root' and ref['mode']=='erased':
            key=(ref['n'],ref['seed'],R,task['seed']);paired.setdefault(key,[]).append((ref,answer,job,record))
    for pair in paired.values():
        assert len(pair)==2
        assert pair[0][2]==pair[1][2] and pair[0][1]==pair[1][1]
        assert sum(x[3]['threshold_correct'] for x in pair)==1
    rejected=[]
    for field in ('hits','predictor_calls','random_draws'):
        bad=copy.deepcopy(result['results'][0])
        if field=='hits':bad[field][0]=1-bad[field][0]
        else:bad[field]+=1
        try:replay(jobs[0],tasks[0]['trials'],tasks[0]['seed'],bad)
        except AssertionError:rejected.append(field)
        else:raise AssertionError('accepted corruption')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    codebytes=sum((src/name).stat().st_size for name in ('rollout.py','rollout_runner.py','decode.py'))
    assert codebytes<=cfg['max_source_bytes']
    size=sum(p.stat().st_size for p in out.iterdir() if p.is_file());assert size<cfg['output_limit_bytes']
    receipt=dict(validation='PASS',cases=len(tasks),paired_erasure_checks=len(paired),corruptions=rejected,
        source_bytes=codebytes,total_rollouts=sum(t['trials'] for t in tasks),
        total_predictor_calls=sum(r['predictor_calls'] for r in summary),total_transition_draws=sum(r['random_draws'] for r in summary),
        nominal_interval_failures=failures,worker_wall_seconds=wall,
        worker_cpu_seconds_before_output=result['cpu_seconds_before_output'],worker_peak_rss_kib=result['peak_rss_kib'],
        elapsed_seconds=time.monotonic()-start,output_bytes=size,
        parent_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir() if p.is_file()})
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    paths=[p for folder in (src,inputs,out) for p in folder.iterdir() if p.is_file()]
    (out/'sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
