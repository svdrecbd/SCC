"""Audit every response and candidate; direct coefficient checks on small cases."""
import copy
import random
from fractions import Fraction
import numpy as np


def parity(words):
    result = np.array(words, dtype=np.uint64, copy=True)
    for shift in [32,16,8,4,2,1]:
        result ^= result >> np.uint64(shift)
    return (result & 1).astype(np.uint8)


def rank(rows):
    pivots = {}
    for word in rows:
        while word:
            col = word.bit_length()-1
            if col in pivots:
                word ^= pivots[col]
            else:
                pivots[col] = word
                break
    return len(pivots)


def masks_from_record(d):
    rng = random.Random(d['decoder_seed'])
    assert d['basis'] == [rng.getrandbits(d['n']) for _ in range(d['k'])]
    masks = [0]
    for s in range(1,2**d['k']):
        low = s & -s
        masks.append(masks[s ^ low] ^ d['basis'][low.bit_length()-1])
    return np.array(masks,dtype=np.uint64)


def reference_transform(v):
    # Descending tensor axes rather than the decoder's ascending butterflies.
    v = np.array(v,dtype=np.int64,copy=True)
    half = len(v)//2
    while half:
        blocks = v.reshape(-1,2,half)
        total = blocks.sum(axis=1)
        difference = blocks[:,0,:]-blocks[:,1,:]
        blocks[:,0,:], blocks[:,1,:] = total, difference
        half //= 2
    return v


def expected_oracle(spec, q):
    result = parity(q & np.uint64(spec['target']))
    projection = np.zeros(len(q),dtype=np.uint16)
    for j,row in enumerate(spec['projection']):
        projection += parity(q & np.uint64(row)).astype(np.uint16) * (2**j)
    if spec['kind']=='lookup':
        result ^= np.asarray(spec['table'],dtype=np.uint8)[projection]
    else:
        for j in range(spec['parameter']):
            result ^= (((projection >> (2*j)) & 3)==3).astype(np.uint8)
    return result


def verify_decode(d, trace, expected, direct=False):
    n,k = d['n'],d['k']; length = 2**k
    masks = masks_from_record(d)
    packed, saved = trace['responses'],trace['candidates']
    assert packed.dtype==np.uint8 and packed.shape==(n,(length-1+7)//8)
    assert saved.dtype==np.uint64 and saved.shape==(length,)
    assert np.all((packed[:,-1] & 128)==0)
    candidates = np.zeros(length,dtype=np.uint64)
    direct_checks = 0
    for i in range(n):
        observed = np.unpackbits(packed[i],bitorder='little')[:length-1]
        assert np.array_equal(observed,expected(masks[1:] ^ np.uint64(1 << i)))
        votes = np.zeros(length,dtype=np.int64)
        votes[1:] = 1-2*observed.astype(np.int64)
        scores = reference_transform(votes)
        if direct:
            for guess in range(length):
                value = sum(int(votes[s]) * (-1 if bin(s & guess).count('1') % 2 else 1)
                            for s in range(length))
                assert int(scores[guess])==value
                direct_checks += 1
        assert np.all(scores!=0)
        candidates += (scores<0).astype(np.uint64) * np.uint64(1 << i)
    assert np.array_equal(candidates,saved)
    assert d['candidates']==sorted(set(map(int,candidates)))
    expected_meter=dict(logical_oracle_queries=n*(length-1),oracle_batches=n,
        transform_add_sub=n*length*k,candidate_bit_placements=n*length,
        subset_xors=length-1,transcript_payload_bytes=n*((length-1+7)//8),
        candidate_array_bytes=8*length,mask_array_bytes=8*length,
        transform_array_bytes=8*length,raw_candidate_slots=length)
    for name,value in expected_meter.items():
        assert d[name]==value,(name,d[name],value)
    return direct_checks


def verify_selection(selected,candidates,rows,rhs):
    matches=[]
    for x in candidates:
        bits = [sum(int(bit) for bit in bin(x & row)[2:]) % 2 for row in rows]
        if sum(bit*2**i for i,bit in enumerate(bits))==rhs:
            matches.append(x)
    assert selected==dict(matches=matches,candidate_checks=len(candidates),
                          verifier_row_parities=len(rows)*len(candidates))


def verify(records,controls,cfg,out):
    wanted=[(n,kind,p,seed) for n,kind,p in cfg['main'] for seed in cfg['seeds']]
    wanted += [(cfg['small_n'],'quadratic',2,seed) for seed in cfg['seeds']]
    assert [(r['spec']['n'],r['spec']['kind'],r['spec']['parameter'],r['spec']['seed']) for r in records]==wanted
    response_count=0; direct_checks=0; summary=[]
    for r in records:
        s,d=r['spec'],r['decoded']; n=s['n']; k=d['k']
        assert n==d['n'] and rank(s['rows'])==n
        assert rank(s['projection'])==len(s['projection'])
        assert all(0<=v<2**n for v in [s['target'],*s['rows'],*s['projection']])
        assert sum((bin(row & s['target']).count('1') % 2)*2**i for i,row in enumerate(s['rows']))==s['rhs']
        if s['kind']=='quadratic':
            width=2*s['parameter']
            assert len(s['projection'])==width
            errors=sum(sum(((x >> (2*j)) & 3)==3 for j in range(s['parameter']))%2 for x in range(2**width))
        else:
            width=s['parameter']; assert len(s['table'])==2**width and set(s['table'])=={0,1}
            errors=sum(s['table'])
        accuracy=1-Fraction(errors,2**width)
        epsilon=accuracy-Fraction(1,2)
        assert Fraction(r['epsilon'])==epsilon and Fraction(r['uniform_accuracy'])==accuracy
        beta=Fraction(n,4*(2**k-1)*epsilon**2)
        assert Fraction(r['failure_bound'])==beta
        if r['group']=='main':
            assert beta<=Fraction(cfg['failure_target'])
            assert k==1 or Fraction(n,4*(2**(k-1)-1)*epsilon**2)>Fraction(cfg['failure_target'])
        else:
            assert k==cfg['small_k']
        with np.load(out/(r['name']+'.npz')) as z:
            direct_checks+=verify_decode(d,z,lambda q:expected_oracle(s,q),r['group']=='small')
        response_count+=d['logical_oracle_queries']
        verify_selection(r['selected'],d['candidates'],s['rows'],s['rhs'])
        assert r['target_found']==(s['target'] in d['candidates'])
        assert r['selected']['matches']==([s['target']] if r['target_found'] else [])
        assert r['candidate_protected_bits']==sorted({x&1 for x in d['candidates']})
        assert r['oracle_projection_parities']==d['logical_oracle_queries']*len(s['projection'])
        assert r['oracle_target_parities']==d['logical_oracle_queries']
        assert r['oracle_projection_bits']==n*len(s['projection'])
        assert r['oracle_table_bits']==len(s['table']) and r['verifier_matrix_bits']==n*n
        summary.append(dict(name=r['name'],accuracy=str(accuracy),k=k,beta=str(beta),
            found=r['target_found'],unique_candidates=len(d['candidates']),
            protected_bits=r['candidate_protected_bits'],calls=d['logical_oracle_queries'],
            add_sub=d['transform_add_sub'],decode_seconds=r['decode_seconds']))
    a=controls['ambiguity']
    assert a['candidates']==[0,1]
    assert a['oracle']==[int(bool(q&1) and bool(q&2)) ^ int(bool(q&4) and bool(q&8)) for q in range(16)]
    assert a['agreements']==[sum(value==(bin(q & target).count('1')%2) for q,value in enumerate(a['oracle'])) for target in [0,1]]==[10,10]
    verify_selection(a['full0'],[0,1],[1,2,4,8],0)
    verify_selection(a['full1'],[0,1],[1,2,4,8],1)
    verify_selection(a['partial'],[0,1],[2,4,8],0)
    p=controls['poison']; n=p['n']; masks=masks_from_record(p['known'])
    poison=np.unique(np.concatenate([masks[1:] ^ np.uint64(1<<i) for i in range(n)]))
    assert np.array_equal(np.load(out/'poison_queries.npy'),poison)
    errors=int(parity(poison & np.uint64(p['target'] ^ p['wrong'])).sum())
    assert p['error_count']==errors and p['query_set_size']==len(poison)
    assert Fraction(p['uniform_accuracy'])==1-Fraction(errors,2**n)
    def poisoned(q):
        base=parity(q & np.uint64(p['target']))
        base ^= np.isin(q,poison).astype(np.uint8) * parity(q & np.uint64(p['target'] ^ p['wrong']))
        return base
    for label in ['known','fresh']:
        with np.load(out/('poison_'+label+'.npz')) as z:
            verify_decode(p[label],z,poisoned)
        assert p[label+'_found']==(p['target'] in p[label]['candidates'])
        response_count+=p[label]['logical_oracle_queries']
    assert not p['known_found']
    c=controls['coordinate']
    assert c['target']!=0 and c['coordinate_accuracy']=='1'
    assert Fraction(c['uniform_parity_accuracy'])==Fraction(1,2)+Fraction(bin(c['target']).count('1'),2**c['n'])
    assert controls['invalid']==['zero_advantage','dimension','nonbits']
    return dict(cases=summary,logical_answers_audited=response_count,direct_coefficients_checked=direct_checks,
                main_recovered=sum(r['target_found'] for r in records if r['group']=='main'),
                main_cases=sum(r['group']=='main' for r in records),
                poison=dict(accuracy=p['uniform_accuracy'],known_found=p['known_found'],fresh_found=p['fresh_found']))


def corruptions(record,out):
    d=record['decoded']; s=record['spec']
    with np.load(out/(record['name']+'.npz')) as z:
        trace={name:z[name].copy() for name in z.files}
    mutations=[('response',lambda x,t:t['responses'].__setitem__((0,0),t['responses'][0,0]^1)),
               ('candidate_word',lambda x,t:t['candidates'].__setitem__(0,int(t['candidates'][0])^1)),
               ('candidate_list',lambda x,t:x.__setitem__('candidates',[])),
               ('query_count',lambda x,t:x.__setitem__('logical_oracle_queries',0)),
               ('work_count',lambda x,t:x.__setitem__('transform_add_sub',0)),
               ('basis',lambda x,t:x['basis'].__setitem__(0,x['basis'][0]^1)),
               ('padding',lambda x,t:t['responses'].__setitem__((0,-1),t['responses'][0,-1]|128))]
    passed=[]
    for name,edit in mutations:
        bad=copy.deepcopy(d); t={key:value.copy() for key,value in trace.items()}
        edit(bad,t)
        try:
            verify_decode(bad,t,lambda q:expected_oracle(s,q),True)
        except AssertionError:
            passed.append(name)
        else:
            raise AssertionError('corruption survived: '+name)
    return passed
