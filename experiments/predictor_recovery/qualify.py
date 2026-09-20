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


def encode(q):return [q.numerator,q.denominator]

def exact_values(kernel,hazards,horizon,D):
    n=len(kernel);hazards=set(hazards);v=[F(s in hazards) for s in range(n)]
    for _ in range(horizon):
        v=[F(1) if s in hazards else sum((F(w,D)*v[t] for t,w in row),F(0)) for s,row in enumerate(kernel)]
    return v


def occupancy(kernel,horizon,D):
    n=len(kernel);p=[F(0)]*n;p[0]=F(1);total=[F(0)]*n
    for _ in range(horizon):
        total=[x+y for x,y in zip(total,p)];nextp=[F(0)]*n
        for s,row in enumerate(kernel):
            for t,w in row:nextp[t]+=p[s]*F(w,D)
        p=nextp
    return [v/horizon for v in total], p


def tv_rows(p,q,D):
    result=[]
    for a,b in zip(p,q):
        aa={t:w for t,w in a};bb={t:w for t,w in b}
        result.append(F(sum(abs(aa.get(t,0)-bb.get(t,0)) for t in aa.keys()|bb.keys()),2*D))
    return result


def worlds(cfg):
    D=cfg['denominator'];cases=[]
    for n in cfg['dimensions']:
        for seed in cfg['seeds']:
            rng=random.Random(seed+1000*n)
            hazards=sorted(rng.sample(range(1,n),max(1,n//16)))
            base=[]
            for s in range(n):
                if s in hazards:base.append([[s,D]])
                else:
                    a,b=rng.sample(range(n),2);w=rng.choice([16,32,48]);base.append([[a,w],[b,D-w]])
            for h in cfg['horizons']:
                for mode in ('exact','small_bias'):
                    retained=copy.deepcopy(base)
                    if mode=='small_bias':
                        for s in range(n):
                            if s not in hazards:retained[s][0][1]+=1;retained[s][1][1]-=1
                    cases.append(dict(family='random',n=n,seed=seed,horizon=h,mode=mode,truth=base,retained=retained,hazards=hazards))
            # Pair differs only in one world's root transition probability.
            tail=[[[0,D]],[[1,D]],[[2,D]]]
            for s in range(3,n):
                a,b=rng.sample(range(1,n),2);tail.append([[a,32],[b,32]])
            for theta,w in [(0,16),(1,48)]:
                truth=copy.deepcopy(tail);truth[0]=[[2,w],[1,D-w]]
                for mode in ('exact','erased','recoded','recoded_repaired'):
                    retained=copy.deepcopy(truth)
                    if mode=='erased':retained[0]=[[2,32],[1,32]]
                    if mode in ('recoded','recoded_repaired'):
                        retained[0]=[[2,D-w],[1,w]]
                    cases.append(dict(family='paired_root',theta=theta,n=n,seed=seed,horizon=12,mode=mode,
                                      truth=truth,retained=retained,hazards=[2]))
    return cases


def job_for(c,D):return {k:c[k] for k in ('n','horizon','retained','hazards')}|dict(denominator=D,decode_root=c['mode']=='recoded_repaired')


def check_answer(case,answer,D):
    n=case['n'];h=case['horizon'];hazards=case['hazards'];retained=copy.deepcopy(case['retained'])
    decoded=case['mode']=='recoded_repaired'
    if decoded:
        a,b=retained[0];retained[0]=[[a[0],b[1]],[b[0],a[1]]]  # Independent inverse: exchange weights.
    q=exact_values(retained,hazards,h,D)
    assert answer['denominator']==D**h
    assert [F(v,answer['denominator']) for v in answer['numerators']]==q
    entries=sum(len(row) for s,row in enumerate(retained) if s not in hazards)
    assert answer['meter']==dict(predictor_calls=n-len(hazards),returned_entries=entries,multiply_adds=h*entries,
                                 state_writes=n*(h+1),cached_rows=n-len(hazards),peak_value_entries=2*n,
                                 decoder_weight_subtractions=2*int(decoded),decoder_entry_writes=2*int(decoded))
    p=exact_values(case['truth'],hazards,h,D);nu,p_end=occupancy(case['truth'],h,D)
    _,q_end=occupancy(retained,h,D)
    assert p[0]==sum((p_end[s] for s in hazards),F(0))
    assert q[0]==sum((q_end[s] for s in hazards),F(0))
    eps=tv_rows(case['truth'],retained,D);mean=sum(eps,F(0))/n
    path_bound=min(F(1),h*sum((a*b for a,b in zip(nu,eps)),F(0)))
    C=n*max(nu);uniform_bound=min(F(1),h*C*mean)
    error=abs(p[0]-q[0]);assert error<=path_bound<=uniform_bound
    assert sum(nu)==1
    return dict(**{k:case[k] for k in ('family','n','seed','horizon','mode')},
        theta=case.get('theta'),true_risk=encode(p[0]),recovered_risk=encode(q[0]),risk_error=encode(error),
        raw_uniform_mean_tv=encode(sum(tv_rows(case['truth'],case['retained'],D),F(0))/n),
        uniform_mean_tv=encode(mean),path_bound=encode(path_bound),coverage_C=encode(C),uniform_bound=encode(uniform_bound),
        root_occupancy=encode(nu[0]),threshold_correct=(p[0]>=F(1,2))==(q[0]>=F(1,2)),
        **answer['meter'])


def binomial_frontier(budgets):
    rows=[]
    for k in budgets:
        p=[F(math.comb(k,j)*3**(k-j),4**k) for j in range(k+1)]
        q=[F(math.comb(k,j)*3**j,4**k) for j in range(k+1)]
        assert sum(p)==sum(q)==1
        optimal=sum((min(a,b) for a,b in zip(p,q)),F(0))/2
        errors=[]
        for theta,law in enumerate((p,q)):
            err=F(0)
            for j,mass in enumerate(law):
                decision=F(1) if 2*j>k else F(0) if 2*j<k else F(1,2)
                err+=mass*(decision if theta==0 else 1-decision)
            errors.append(err)
        assert errors==[optimal,optimal]
        if k<=4:
            for theta,law in enumerate((p,q)):
                counts=[F(0)]*(k+1)
                for sequence in range(1<<k):
                    j=sequence.bit_count();a=F(1,4) if theta==0 else F(3,4)
                    counts[j]+=a**j*(1-a)**(k-j)
                assert counts==law
        rows.append(dict(observations=k,error=encode(optimal),accuracy=encode(1-optimal),
                         count_values=k+1,count_bits=math.ceil(math.log2(k+1)) if k else 0,
                         pmf_low=[encode(v) for v in p],pmf_high=[encode(v) for v in q],parent_bit_error=[0,1]))
    return rows


def controls(case,answer,D):
    rejected=[]
    for name in ('risk','calls','normalizer','entry_count','decoder_work'):
        bad=copy.deepcopy(answer)
        if name=='risk':bad['numerators'][0]+=1
        if name=='calls':bad['meter']['predictor_calls']+=1
        if name=='normalizer':bad['denominator']+=1
        if name=='entry_count':bad['meter']['returned_entries']+=1
        if name=='decoder_work':bad['meter']['decoder_weight_subtractions']+=1
        try:check_answer(case,bad,D)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('accepted '+name)
    return rejected


def main():
    src=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve()
    assert platform.node()=='charon' and (src/'plan-frozen.md').is_file()
    cfg=json.loads((src/'config.json').read_text());D=cfg['denominator'];out.mkdir(exist_ok=False);start=time.monotonic()
    (out/'machine.json').write_text(json.dumps(dict(host=platform.node(),platform=platform.platform(),python=sys.version,
        executable=sys.executable,affinity=sorted(os.sched_getaffinity(0))),indent=2)+'\n')
    cases=worlds(cfg);jobs=[job_for(c,D) for c in cases]
    request=json.dumps(jobs);(out/'worker-input.json').write_text(request+'\n')
    before=time.monotonic()
    process=subprocess.run([sys.executable,'-S',str(src/'runner.py')],input=request,text=True,capture_output=True,
        check=True,timeout=cfg['timeout_seconds'])
    elapsed=time.monotonic()-before
    (out/'worker.stdout.json').write_text(process.stdout);(out/'worker.stderr').write_text(process.stderr)
    result=json.loads(process.stdout);assert len(result['answers'])==len(cases)
    summary=[check_answer(c,a,D) for c,a in zip(cases,result['answers'])]
    paired=[]
    for n in cfg['dimensions']:
        for seed in cfg['seeds']:
            pair=[c for c in cases if c['family']=='paired_root' and c['mode']=='erased' and c['n']==n and c['seed']==seed]
            assert len(pair)==2 and pair[0]['theta']!=pair[1]['theta']
            assert json.dumps(job_for(pair[0],D),sort_keys=True)==json.dumps(job_for(pair[1],D),sort_keys=True)
            for c in pair:
                eps=tv_rows(c['truth'],c['retained'],D)
                assert eps[0]==F(1,4) and all(e==0 for e in eps[1:])
            paired.append(dict(n=n,seed=seed,observable_input_identical=True,uniform_mean_tv=encode(F(1,4*n)),
                               excluded_root_mean_tv=[0,1],singular_root_occupancy=[1,12]))
    frontier=binomial_frontier(cfg['observation_budgets'])
    negative=controls(cases[0],result['answers'][0],D)
    codebytes=sum((src/name).stat().st_size for name in ('adapter.py','runner.py','decode.py'))
    assert codebytes<=cfg['max_adapter_and_runner_bytes']
    assert max(a['meter']['predictor_calls'] for a in result['answers'])<=cfg['max_predictor_calls_per_job']
    for name,value in [('cases',cases),('summary',summary),('erasure_pairs',paired),('observation_frontier',frontier),('controls',negative)]:
        (out/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    size=sum(p.stat().st_size for p in out.iterdir() if p.is_file());assert size<cfg['output_limit_bytes']
    receipt=dict(validation='PASS',cases=len(cases),paired_erasures=len(paired),corruptions=len(negative),
                 adapter_and_runner_bytes=codebytes,worker_input_bytes=len(request.encode()),
                 max_predictor_calls=max(r['predictor_calls'] for r in summary),
                 total_predictor_calls=sum(r['predictor_calls'] for r in summary),
                 decoder_subtractions=sum(r['decoder_weight_subtractions'] for r in summary),
                 worker_wall_seconds=elapsed,worker_cpu_seconds_before_output=result['cpu_seconds_before_output'],
                 worker_peak_rss_kib=result['peak_rss_kib'],elapsed_seconds=time.monotonic()-start,output_bytes=size)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    paths=[p for folder in (src,out) for p in folder.iterdir() if p.is_file()]
    (out/'sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
    print(json.dumps(receipt))

if __name__=='__main__':main()
