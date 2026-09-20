"""Exact small group certificate for the coordinate-to-linear-query bridge."""
import copy
import hashlib
import itertools
import json
import platform
import sys
import time
from fractions import Fraction
from pathlib import Path


def scalar(rows,x):
    bits=[(x >> j)&1 for j in range(3)]
    return sum((sum(((row >> j)&1)*bits[j] for j in range(3))%2)*2**i
               for i,row in enumerate(rows))


def prediction(mode,index,i,truth,original,b):
    if mode=='correct': return truth
    if mode=='first_bad': return truth ^ int(i==0)
    if mode=='table_errors': return truth ^ int((3*index+i+b)%8<3)
    if mode=='native_only': return truth if index==original else 0
    raise ValueError(mode)


def evaluate(cfg):
    group=[];perms=[]
    for rows in itertools.product(range(8),repeat=3):
        perm=tuple(sum(((x & row).bit_count()%2)*2**i for i,row in enumerate(rows)) for x in range(8))
        if len(set(perm))==8:
            group.append(rows);perms.append(perm)
    assert len(group)==168
    lookup={perm:i for i,perm in enumerate(perms)}
    inverses=[tuple(perm.index(y) for y in range(8)) for perm in perms]
    records=[];wrong_inverse_failures=0
    for original,b in itertools.product(cfg['original_indices'],cfg['commands']):
        x=inverses[original][b]
        counts={mode:[0]*8 for mode in cfg['modes']}; fibers=[0]*8;seen=set()
        for r,perm in enumerate(perms):
            transformed=lookup[tuple(perms[original][inverses[r][z]] for z in range(8))]
            changed_solution=perm[x]
            assert perms[transformed][changed_solution]==b
            if perms[original][perm[changed_solution]]!=b:
                wrong_inverse_failures+=1
            for i,q in enumerate(group[r]):
                seen.add((transformed,i));fibers[q]+=1
                truth=(changed_solution >> i)&1
                assert truth==((x & q).bit_count()%2)
                for mode in cfg['modes']:
                    answer=prediction(mode,transformed,i,truth,original,b)
                    counts[mode][q]+=int(answer==truth)
        assert len(seen)==504 and fibers==[0]+[72]*7
        for mode in cfg['modes']:
            u=Fraction(sum(counts[mode]),504)
            related=(1+sum(Fraction(counts[mode][q],72) for q in range(1,8)))/8
            assert related==Fraction(1,8)+Fraction(7,8)*u
            native=sum(prediction(mode,original,i,(x >> i)&1,original,b)==((x >> i)&1) for i in range(3))
            records.append(dict(original=original,b=b,solution=x,mode=mode,
                                fiber_counts=fibers,correct_by_query=counts[mode],
                                native_accuracy=str(Fraction(native,3)),
                                representation_accuracy=str(u),related_accuracy=str(related)))
    return dict(group=group,records=records,wrong_inverse_failures=wrong_inverse_failures)


def audit(data,cfg):
    # Scalar finite-state enumeration, independent of evaluate's permutation maps.
    group=[tuple(rows) for rows in data['group']]
    expected=[rows for rows in itertools.product(range(8),repeat=3)
              if len({scalar(rows,x) for x in range(8)})==8]
    assert group==expected and len(group)==168
    index={rows:j for j,rows in enumerate(group)}
    wanted=list(itertools.product(cfg['original_indices'],cfg['commands'],cfg['modes']))
    assert [(r['original'],r['b'],r['mode']) for r in data['records']]==wanted
    cached={};inverse_failures=0;checks=0
    for original,b in itertools.product(cfg['original_indices'],cfg['commands']):
        x=next(x for x in range(8) if scalar(group[original],x)==b)
        counts={mode:[0]*8 for mode in cfg['modes']};fibers=[0]*8;seen=set()
        for rows in group:
            # Solve R v=e_j separately to obtain columns of R^-1.
            inverse_cols=[next(v for v in range(8) if scalar(rows,v)==1<<j) for j in range(3)]
            composed_cols=[scalar(group[original],v) for v in inverse_cols]
            composed=tuple(sum(((composed_cols[j] >> i)&1)*2**j for j in range(3)) for i in range(3))
            j=index[composed]
            changed=next(v for v in range(8) if scalar(composed,v)==b)
            assert changed==scalar(rows,x)
            inverse_failures+=int(scalar(group[original],scalar(rows,changed))!=b)
            for i,q in enumerate(rows):
                fibers[q]+=1;seen.add((j,i));truth=bin(q & x).count('1')%2
                for mode in cfg['modes']:
                    if mode=='native_only': answer=truth if j==original else 0
                    else:
                        error=(mode=='first_bad' and i==0) or (mode=='table_errors' and (3*j+i+b)%8 in (0,1,2))
                        answer=truth ^ int(error)
                    counts[mode][q]+=int(answer==truth)
                checks+=1
        assert fibers==[0]+[72]*7 and len(seen)==504
        cached[original,b]=(x,counts)
    for r in data['records']:
        x,counts=cached[r['original'],r['b']]
        assert r['solution']==x and r['fiber_counts']==[0]+[72]*7
        assert r['correct_by_query']==counts[r['mode']]
        u=Fraction(sum(r['correct_by_query']),504)
        assert Fraction(r['representation_accuracy'])==u
        assert Fraction(r['related_accuracy'])==Fraction(1,8)+Fraction(7,8)*u
        native_correct=0
        for i in range(3):
            error=(r['mode']=='first_bad' and i==0) or (r['mode']=='table_errors' and (3*r['original']+i+r['b'])%8 in (0,1,2))
            native_correct+=int(not error)
        assert Fraction(r['native_accuracy'])==Fraction(native_correct,3)
    assert data['wrong_inverse_failures']==inverse_failures>0
    return dict(worlds=len(cached),conditions=len(data['records']),coordinate_transform_checks=checks,
                wrong_inverse_failures=inverse_failures)


def corruptions(data,cfg):
    passed=[]
    for name,change in [
        ('fiber',lambda d:d['records'][0]['fiber_counts'].__setitem__(1,71)),
        ('coverage',lambda d:d['records'].pop()),
        ('transfer',lambda d:d['records'][0].__setitem__('related_accuracy','1/2')),
        ('wrong_inverse_control',lambda d:d.__setitem__('wrong_inverse_failures',0))]:
        bad=copy.deepcopy(data);change(bad)
        try: audit(bad,cfg)
        except AssertionError: passed.append(name)
        else: raise AssertionError('accepted corruption '+name)
    return passed


if __name__=='__main__':
    src=Path(__file__).resolve().parent;out=Path(sys.argv[1]).resolve()
    assert platform.node()=='charon' and (src/'plan-frozen.md').is_file()
    cfg=json.loads((src/'symmetry_config.json').read_text());assert cfg['n']==3
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    (out/'machine.json').write_text(json.dumps(dict(host=platform.node(),python=sys.version,
        platform=platform.platform(),executable=sys.executable),indent=2)+'\n')
    try:
        data=evaluate(cfg);summary=audit(data,cfg);negative=corruptions(data,cfg)
        (out/'records.json').write_text(json.dumps(data,indent=2)+'\n')
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (out/'corruptions.json').write_text(json.dumps(negative,indent=2)+'\n')
        size=sum(p.stat().st_size for p in out.iterdir() if p.is_file());assert size<cfg['output_limit_bytes']
        receipt=dict(validation='PASS',**summary,corruptions=len(negative),
                     elapsed_seconds=time.monotonic()-start,output_bytes=size)
        (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        paths=[p for folder in (src,out) for p in folder.iterdir() if p.is_file()]
        (out/'sha256.json').write_text(json.dumps({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},indent=2)+'\n')
        print(json.dumps(receipt))
    except Exception as error:
        (out/'failure.json').write_text(json.dumps(dict(error=repr(error)))+'\n');raise
