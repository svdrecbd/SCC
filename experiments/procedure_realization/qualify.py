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
from reference import bits, pack, mul, make, certificate, parent_step

HERE=Path(__file__).resolve().parent


def dump(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n')


def dense_cost(matrix,columns):
    a=matrix.copy();k=0;tests=swaps=xors=scans=0
    for j in range(columns):
        found=np.flatnonzero(a[k:,j])
        tests+=int(found[0])+1 if len(found) else len(a)-k
        if not len(found):continue
        i=k+int(found[0]);swaps+=int(i!=k);a[[i,k]]=a[[k,i]]
        for i in range(len(a)):
            scans+=1
            if i!=k and a[i,j]:a[i]^=a[k];xors+=1
        k+=1
    return dict(pivot_bit_tests=tests,elimination_bit_tests=scans,row_swaps=swaps,row_xors=xors,pivots=k)


def check_meter(job,answer):
    n=job['upper_order'];r=answer['order'];m=job['impulse_outputs'];meter=answer['meter']
    h=np.array([[m[i+j] for j in range(n)] for i in range(n)],dtype=np.uint8)
    aug=np.concatenate([h[:r,:r],np.eye(r,dtype=np.uint8)],axis=1)
    p=dense_cost(h,n);q=dense_cost(aug,r)
    expected={k:p[k]+q[k] for k in p}
    count=len(job['warm_outputs'])
    expected.update(max_row_width_bits=max(n,2*r),largest_elimination_matrix_bits=max(n*n,2*r*r),
        coefficient_dot_products=r,warm_dot_products=count*r,warm_successor_steps=count*r,
        calibration_output_bits=2*n,calibration_parent_steps=2*n,warm_output_bits=count*r,
        warm_parent_steps=count*r,coefficient_payload_bits=2*r,per_stream_state_bits=r,warm_states_returned=count,
        stream_steps=sum(len(u) for u in job['inputs']),warning_steps=sum(len(u)//8*4 for u in job['inputs']),
        exhaustive_steps=(count-job['exhaustive_start'])*256*8 if job['exhaustive_start'] is not None else 0,
        restart_steps=sum(len(u) for u in job['inputs']))
    expected.update(head_patch_stream_steps=sum(len(u) for u in job['inputs']),head_patch_warning_steps=0)
    assert meter==expected, (meter,expected)


def verify(parents,jobs,answers,cfg):
    pp,jj=make(cfg);assert pp==parents and jj==jobs,'input substitution'
    assert len(parents)==len(jobs)==len(answers)==24
    summary=[];certificates=[]
    for p,job,ans in zip(parents,jobs,answers):
        assert p['id']==job['id']==ans['id']
        cert=certificate(p,ans);certificates.append(cert);check_meter(job,ans)
        a=np.array(p['matrix'],dtype=np.uint8);b=np.array(p['input'],dtype=np.uint8);c=np.array(p['output'],dtype=np.uint8)
        def trajectory(state,inputs):
            x=bits(state,p['n']);out=[]
            for u in inputs:
                x=mul(a,x) ^ (b if u else 0);out.append(int(mul(c,x)))
            return out
        assert len(ans['streams'])==len(p['streams'])
        total=dead=restart=warn_total=warn_true=0
        for initial,inputs,record in zip(p['warm_after'],p['streams'],ans['streams']):
            x=bits(initial,p['n']);expected=[];warning=[]
            for k,u in enumerate(inputs):
                if k%8==0:warning.append(int(any(trajectory(pack(x),[1-v for v in inputs[k:k+4]]))))
                if k==0:
                    # Scalar row-wise check of dense parent arithmetic.
                    assert parent_step(p,x,u)==(mul(a,x) ^ (b if u else 0)).tolist()
                x=mul(a,x) ^ (b if u else 0);expected.append(int(mul(c,x)))
            assert record['outputs']==expected
            assert record['warnings']==warning and record['parity']==sum(expected)%2
            assert record['patched_outputs']==expected and record['patched_warnings']==[0]*len(warning)
            frozen=int(mul(c,bits(initial,p['n'])))
            assert record['dead_outputs']==[frozen]*len(inputs)
            assert record['restart_outputs']==trajectory(0,inputs)
            dead+=sum(x==y for x,y in zip(expected,record['dead_outputs']))
            restart+=sum(x==y for x,y in zip(expected,record['restart_outputs']))
            total+=len(inputs);warn_total+=len(warning);warn_true+=sum(warning)
        start=job['exhaustive_start'];exhaust=[]
        if start is not None:
            for initial in p['warm_after'][start:]:
                for word in itertools.product((0,1),repeat=cfg['exhaustive_length']):
                    exhaust.append(trajectory(initial,word))
        assert ans['exhaustive']==exhaust
        summary.append(dict(id=p['id'],ambient_bits=p['n'],order=ans['order'],
            dense_parent_coefficient_bits=p['n']**2+2*p['n'],replacement_coefficient_bits=2*ans['order'],
            per_stream_state_bits=ans['order'],stream_output_bits=total,repaired_correct=total,
            frozen_correct=dead,zero_restart_correct=restart,warning_cases=warn_total,warning_ones=warn_true,
            constant_zero_warning_correct=warn_total-warn_true,exhaustive_output_bits=sum(map(len,exhaust)),
            constant_one_warning_correct=warn_true,patched_useful_correct=total,
            private_unobservable_pair=p['private'] is not None,meter=ans['meter']))
    corruption=[]
    for kind in ('coefficients','readout','warm_state','row_xors'):
        bad=copy.deepcopy(answers[0])
        if kind in ('coefficients','readout'):bad[kind]^=1
        if kind=='warm_state':bad['warm_states'][0]^=1
        if kind=='row_xors':bad['meter']['row_xors']+=1
        try:certificate(parents[0],bad);check_meter(jobs[0],bad)
        except AssertionError:corruption.append(kind)
        else:raise AssertionError('accepted corruption '+kind)
    return dict(models=summary,corruptions=corruption),certificates


def main():
    out=Path(sys.argv[1]).resolve();out.mkdir(exist_ok=False);start=time.perf_counter()
    assert platform.node()=='charon' and (HERE/'plan-frozen.md').exists()
    cfg=json.loads((HERE/'config.json').read_text())
    dump(out/'machine.json',dict(host=platform.node(),platform=platform.platform(),python=sys.version,
        numpy=np.__version__,affinity=sorted(os.sched_getaffinity(0))))
    try:
        parents,jobs=make(cfg);dump(out/'parents.json',parents);dump(out/'jobs.json',jobs)
        begin=time.perf_counter()
        with (out/'worker.stdout').open('w') as stdout,(out/'worker.stderr').open('w') as stderr:
            proc=subprocess.run([sys.executable,'-S',str(HERE/'runner.py')],input=json.dumps(jobs),text=True,
                stdout=stdout,stderr=stderr,timeout=cfg['worker_timeout_seconds'])
        elapsed=time.perf_counter()-begin
        dump(out/'worker.receipt.json',dict(returncode=proc.returncode,wall_seconds=elapsed,timeout=cfg['worker_timeout_seconds']))
        assert proc.returncode==0
        worker=json.loads((out/'worker.stdout').read_text())
        summary,cert=verify(parents,jobs,worker['answers'],cfg)
        dump(out/'summary.json',summary);dump(out/'certificates.json',cert)
        size=sum(p.stat().st_size for p in out.iterdir() if p.is_file());assert size<cfg['output_limit_bytes']
        receipt=dict(qualification='PASS',systems=len(parents),certificates=len(cert),
            stream_output_bits=sum(r['stream_output_bits'] for r in summary['models']),
            exhaustive_output_bits=sum(r['exhaustive_output_bits'] for r in summary['models']),
            warning_cases=sum(r['warning_cases'] for r in summary['models']),corruptions=summary['corruptions'],
            worker_source_bytes=sum((HERE/name).stat().st_size for name in ('repair.py','runner.py')),
            worker_input_bytes=len(json.dumps(jobs).encode()),worker_wall_seconds=elapsed,
            worker_cpu_seconds_before_output=worker['cpu_seconds_before_output'],worker_peak_rss_kib=worker['peak_rss_kib'],
            total_seconds=time.perf_counter()-start,output_bytes=size)
        dump(out/'receipt.json',receipt)
        dump(out/'sha256.json',{str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in (HERE,out) for p in folder.iterdir() if p.is_file()})
        print(json.dumps(receipt))
    except BaseException as e:dump(out/'failure.json',dict(error=repr(e)));raise


if __name__=='__main__':main()
