import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
from reference import make, terms, truth, trace

HERE=Path(__file__).resolve().parent
def dump(p,value):p.write_text(json.dumps(value,indent=2)+'\n')


def equal(actual,expected,label):
    assert actual==expected,label


def sample_rows(job,degree):
    n=job['n'];basis=terms(n,degree);rows=[];targets=[];states=[]
    for t in range(n,n+job['samples']):
        bits=list(reversed(job['outputs'][t:t+n]))
        states.append(sum(v*(2**i) for i,v in enumerate(bits)))
        rows.append([int(all(bits[i] for i in term)) for term in basis])
        targets.append(job['inputs'][t]^job['outputs'][t+n])
    return np.array(rows,dtype=np.uint8),np.array(targets,dtype=np.uint8),states


def check_fit(job,attempt):
    matrix,target,states=sample_rows(job,attempt['degree']);processed=attempt['processed'];m=matrix.shape[1]
    assert 0<processed<=len(matrix)
    basis={};chosen=[];xor_count=0;contradiction=None
    for i in range(processed):
        row=matrix[i].copy();rhs=int(target[i])
        while row.any():
            p=int(np.flatnonzero(row)[-1])
            if p not in basis:
                basis[p]=(row.copy(),rhs);chosen.append(i);break
            br,by=basis[p];row ^= br;rhs ^= by;xor_count+=1
        if not row.any() and rhs:
            contradiction=i;break
    assert chosen==attempt['independent_rows'] and len(basis)==attempt['rank']
    assert attempt['row_xors']==attempt['provenance_xors']==xor_count
    assert attempt['feature_tests']==processed*m
    if attempt['status']=='inconsistent':
        assert contradiction==processed-1
        witness=attempt['contradiction'];assert witness and witness>>processed==0
        ix=[i for i in range(processed) if (witness>>i)&1]
        assert not np.bitwise_xor.reduce(matrix[ix],axis=0).any()
        assert int(np.bitwise_xor.reduce(target[ix]))==1
    elif attempt['status']=='full_rank':
        assert contradiction is None and processed==len(matrix) and len(basis)==m
        coeff=attempt['coefficients'];assert 0<=coeff<(1<<m)
        vector=np.array([(coeff>>i)&1 for i in range(m)],dtype=np.uint8)
        assert np.array_equal((matrix@vector)&1,target)
    else:
        assert attempt['status']=='underidentified' and contradiction is None and len(basis)<m
    return matrix.shape,states


def polynomial_certificate(parent,ans):
    assert ans['degree']==parent['degree'];basis=terms(parent['n'],ans['degree'])
    masks=[sum(1<<i for i in t) for t in basis]
    values={x:truth(x,parent['terms']) for x in masks};coefs=[]
    for mask in masks:
        sub=mask;value=0
        while True:
            value ^= values[sub]
            if sub==0:break
            sub=(sub-1)&mask
        coefs.append(value)
    assert sum(v<<i for i,v in enumerate(coefs))==ans['coefficients'],'polynomial coefficient mismatch'
    assert any(v and len(t)==parent['degree'] for t,v in zip(basis,coefs))
    return dict(monomial_masks=masks,parent_values=[values[x] for x in masks],coefficients=coefs)


def check_meter(job,ans):
    n=job['n'];a=ans['attempts'];m=len(terms(n,ans['degree']));warm=len(job['warm_outputs'])
    normal=sum(map(len,job['streams']));exhaust=4*len(ans['exhaustive'])
    expected=dict(calibration_parent_steps=len(job['inputs']),recorded_input_bits=len(job['inputs']),
        observed_calibration_bits=len(job['outputs']),warm_observed_bits=n*warm,warm_parent_steps=n*warm,
        warm_successor_steps=n*warm,decoded_sample_payload_bits=job['samples']*(n+1),coefficient_payload_bits=m,
        running_state_bits=n,fit_feature_tests=sum(x['feature_tests'] for x in a),warm_feature_tests=n*warm*m,
        row_xors=sum(x['row_xors'] for x in a),
        provenance_xors=sum(x['provenance_xors'] for x in a),largest_feature_row_bits=max(len(terms(n,x['degree'])) for x in a),
        provenance_width_bound_bits=job['samples'],evaluation_feedback_calls=5*normal+exhaust,evaluation_update_steps=4*normal+exhaust,
        evaluation_feature_tests=(4*normal+exhaust)*m)
    expected['feature_tests']=sum(expected[name] for name in ('fit_feature_tests','warm_feature_tests','evaluation_feature_tests'))
    assert ans['meter']==expected,'meter mismatch'


def verify(parents,jobs,answers,cfg):
    pp,jj=make(cfg);equal(parents,pp,'parent substitution');equal(jobs,jj,'history or input substitution')
    assert len(parents)==len(jobs)==len(answers)==10
    summary=[];certs=[]
    for p,j,a in zip(parents,jobs,answers):
        assert p['id']==j['id']==a['id'];n=j['n']
        assert [x['degree'] for x in a['attempts']]==list(range(1,a['degree']+1))
        for attempt in a['attempts']:
            shape,states=check_fit(j,attempt)
            assert states==p['calibration_states'][n:n+j['samples']]
        assert all(x['status']=='inconsistent' for x in a['attempts'][:-1])
        assert a['attempts'][-1]['status']=='full_rank' and a['coefficients']==a['attempts'][-1]['coefficients']
        certs.append(polynomial_certificate(p,a))
        equal((a['decoded_initial'],a['current_states']),(p['before'],p['after']),'state reconstruction')
        assert len(a['streams'])==len(p['streams'])
        total=frozen_correct=restart_correct=positive=restart_synced=zero_correct=zero_tail_correct=zero_tail_total=recall_bits=0
        for state,inputs,record in zip(p['after'],p['streams'],a['streams']):
            ys,fs,xs=trace(state,inputs,n,p['terms']);expected=dict(outputs=ys,protected=fs,states=xs)
            equal(record['intact'],expected,'stream record')
            assert record['head_removed']==dict(outputs=ys,protected=[0]*len(fs),states=xs)
            assert record['frozen']==dict(outputs=[(state>>(n-1))&1]*len(inputs),
                protected=[truth(state,p['terms'])]*len(inputs),states=[state]*(len(inputs)+1))
            ry,rf,rx=trace(0,inputs,n,p['terms'])
            assert record['restart']==dict(outputs=ry,protected=rf,states=rx)
            zy,zf,zx=trace(state,inputs,n,[])
            equal(record['function_zero'],dict(outputs=zy,protected=zf,states=zx),'function removal')
            zero_correct+=sum(x==y for x,y in zip(ys,zy))
            zero_tail_correct+=sum(x==y for x,y in zip(ys[n:],zy[n:]));zero_tail_total+=len(ys[n:])
            for k in range(n,len(inputs)+1):
                remembered=sum(inputs[k-1-i]<<i for i in range(n))
                equal(zx[k],remembered,'surviving fresh-input memory');recall_bits+=n
            total+=len(ys);positive+=sum(fs)
            frozen_correct+=sum(x==y for x,y in zip(ys,record['frozen']['outputs']))
            restart_correct+=sum(x==y for x,y in zip(ys,ry));restart_synced+=int(rx[-1]==xs[-1])
        start=j['exhaustive_start'];expected_exhaust=[]
        if start is not None:
            for initial in p['before'][start:]:
                for word in itertools.product((0,1),repeat=cfg['exhaustive_length']):
                    ys,fs,xs=trace(initial,word,n,p['terms'])
                    expected_exhaust.append(dict(outputs=ys,protected=fs,states=xs))
        assert a['exhaustive']==expected_exhaust
        check_meter(j,a)
        summary.append(dict(id=p['id'],n=n,degree=a['degree'],coefficients=shape[1],
            rejected_degrees=[x['degree'] for x in a['attempts'][:-1]],full_rank=shape[1],
            stream_steps=total,exact_useful=total,exact_protected=total,
            frozen_correct=frozen_correct,restart_correct=restart_correct,restart_final_state_matches=restart_synced,
            stream_instances=len(p['streams']),constant_zero_protected_correct=total-positive,
            function_zero_correct=zero_correct,function_zero_tail_correct=zero_tail_correct,
            function_zero_tail_total=zero_tail_total,fresh_recall_bits=recall_bits,
            constant_one_protected_correct=positive,exhaustive_steps=4*len(expected_exhaust),meter=a['meter']))
    corruption=[]
    bad=copy.deepcopy(answers[0]);bad['coefficients']^=1
    try:polynomial_certificate(parents[0],bad)
    except AssertionError:corruption.append('coefficients')
    else:raise AssertionError('accepted changed polynomial')
    bad=copy.deepcopy(answers[0]['attempts'][0]);bad['contradiction']=1
    try:check_fit(jobs[0],bad)
    except AssertionError:corruption.append('contradiction_witness')
    else:raise AssertionError('accepted changed contradiction')
    bad=copy.deepcopy(answers[0]['attempts'][-1]);bad['independent_rows'][0]=bad['independent_rows'][1]
    try:check_fit(jobs[0],bad)
    except AssertionError:corruption.append('basis_row')
    else:raise AssertionError('accepted duplicate basis row')
    bad=copy.deepcopy(answers[0]);bad['meter']['row_xors']+=1
    try:check_meter(jobs[0],bad)
    except AssertionError:corruption.append('work_counter')
    else:raise AssertionError('accepted work counter')
    bad=copy.deepcopy(answers[0]);bad['current_states'][0]^=1
    try:equal((bad['decoded_initial'],bad['current_states']),(parents[0]['before'],parents[0]['after']),'state reconstruction')
    except AssertionError:corruption.append('current_state')
    else:raise AssertionError('accepted current state')
    p=parents[0];ys,fs,xs=trace(p['after'][0],p['streams'][0],p['n'],p['terms'])
    bad=copy.deepcopy(answers[0]['streams'][0]['intact']);bad['outputs'][0]^=1
    try:equal(bad,dict(outputs=ys,protected=fs,states=xs),'stream record')
    except AssertionError:corruption.append('stream_output')
    else:raise AssertionError('accepted stream output')
    bad=copy.deepcopy(jobs);bad[0]['outputs'][0]^=1
    try:equal(bad,jj,'history or input substitution')
    except AssertionError:corruption.append('calibration_history')
    else:raise AssertionError('accepted changed history')
    return dict(models=summary,corruptions=corruption),certs


def main():
    out=Path(sys.argv[1]).resolve();out.mkdir(exist_ok=False);start=time.perf_counter()
    assert platform.node()=='charon' and (HERE/'plan-frozen.md').exists()
    cfg=json.loads((HERE/'config.json').read_text())
    dump(out/'machine.json',dict(host=platform.node(),platform=platform.platform(),python=sys.version,
        numpy=np.__version__,affinity=sorted(os.sched_getaffinity(0))))
    try:
        parents,jobs=make(cfg);dump(out/'parents.json',parents);dump(out/'jobs.json',jobs)
        before=time.perf_counter()
        with (out/'worker.stdout').open('w') as stdout,(out/'worker.stderr').open('w') as stderr:
            p=subprocess.run([sys.executable,'-S',str(HERE/'runner.py')],input=json.dumps(jobs),text=True,
                stdout=stdout,stderr=stderr,timeout=cfg['worker_timeout_seconds'])
        elapsed=time.perf_counter()-before
        dump(out/'worker.receipt.json',dict(returncode=p.returncode,wall_seconds=elapsed,timeout_seconds=cfg['worker_timeout_seconds']))
        assert p.returncode==0
        result=json.loads((out/'worker.stdout').read_text())
        summary,certs=verify(parents,jobs,result['answers'],cfg)
        dump(out/'summary.json',summary);dump(out/'certificates.json',certs)
        size=sum(p.stat().st_size for p in out.iterdir() if p.is_file());assert size<=cfg['output_limit_bytes']
        receipt=dict(qualification='PASS',models=len(parents),stream_steps=sum(x['stream_steps'] for x in summary['models']),
            exhaustive_steps=sum(x['exhaustive_steps'] for x in summary['models']),
            inconsistent_model_certificates=sum(len(x['rejected_degrees']) for x in summary['models']),
            corruptions=summary['corruptions'],worker_source_bytes=sum((HERE/name).stat().st_size for name in ('repair.py','runner.py')),
            worker_input_bytes=len(json.dumps(jobs).encode()),worker_wall_seconds=elapsed,
            worker_cpu_seconds_before_output=result['cpu_seconds_before_output'],worker_peak_rss_kib=result['peak_rss_kib'],
            total_seconds=time.perf_counter()-start,output_bytes=size)
        dump(out/'receipt.json',receipt)
        dump(out/'sha256.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (HERE,out) for p in folder.iterdir() if p.is_file()})
        print(json.dumps(receipt))
    except BaseException as error:dump(out/'failure.json',dict(error=repr(error)));raise


if __name__=='__main__':main()
