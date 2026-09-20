import copy
import itertools
import json
import math
import random
from cache import query
from proof import check_unsat, conflict, masks


def valid(formula, model, n, goal=None):
    return (isinstance(model,list) and len(model)==n and
            all(abs(x)==i for i,x in enumerate(model,1)) and
            (goal is None or goal in model) and
            all(set(clause).intersection(model) for clause in formula))


def check_row(row, cfg):
    n=row['n']; goals=[s*i for i in range(1,n+1) for s in (1,-1)]
    summary=dict(n=n,seed=row['seed'],proofs=0,proof_additions=0,proof_clause_visits=0,
                 brute_assignments=0,source_correct=0,fresh_known=0,fresh_unknown=0,
                 fresh_repaired=0,goals_per_context=2*n)
    for label,seed in [('source',row['seed']),('fresh',row['seed']+cfg['fresh_seed_offset'])]:
        f=row[label];rng=random.Random(seed+1000*n)
        expected=[]
        for _ in range(math.ceil(cfg['clause_density']*n)):
            vs=rng.sample(range(1,n+1),3)
            expected.append([v if rng.getrandbits(1) else -v for v in vs])
        assert f==expected
        assert not set.intersection(*(set(c) for c in f)), 'common guard survived'
        ref=row[label+'_ref'];assert [r['goal'] for r in ref['goals']]==goals
        core=ref['core'];assert core['status'] in ('SAT','UNSAT')
        summary[label+'_core']=core['status']
        for answer,formula in [(core,f)]+[(r,f+[[r['goal']]]) for r in ref['goals']]:
            if answer['status']=='SAT':
                assert valid(formula,answer['model'],n)
            else:
                assert answer['status']=='UNSAT' and answer['model'] is None
                if answer.get('derived_from_core'):
                    assert core['status']=='UNSAT'
                else:
                    meter=check_unsat(formula,answer['proof'])
                    summary['proofs']+=1;summary['proof_additions']+=meter['additions']
                    summary['proof_clause_visits']+=meter['clause_visits']
        if n==12:
            feasible=set(); sat=False
            for bits in itertools.product((False,True),repeat=n):
                model=[i if b else -i for i,b in enumerate(bits,1)]
                if valid(f,model,n): feasible.update(model);sat=True
                summary['brute_assignments']+=1
            assert sat==(core['status']=='SAT')
            assert feasible=={r['goal'] for r in ref['goals'] if r['status']=='SAT'}
        summary[label+'_feasible_goals']=sum(r['status']=='SAT' for r in ref['goals'])
        summary[label+'_protected']=ref['goals'][0]['status']
    cache=row['cache'];assert cache['formula']==sorted(sorted(c) for c in row['source'])
    assert len(cache['models'])==len(set(tuple(m) for m in cache['models']))
    for model in cache['models']:assert valid(row['source'],model,n)
    assert cache['unsat_goals']==[r['goal'] for r in row['source_ref']['goals'] if r['status']=='UNSAT']
    assert row['payload_bits']==n*len(cache['models'])+2*n
    assert row['serialized_cache_bytes']==len((json.dumps(cache,sort_keys=True,separators=(',',':'))+'\n').encode())
    for label in ('source','fresh','reordered'):
        f=row['fresh'] if label=='fresh' else row['source']
        reference=row['fresh_ref' if label=='fresh' else 'source_ref']['goals']
        results=row[label+'_cache'];assert [r['goal'] for r in results]==goals
        for answer,ref in zip(results,reference):
            if answer['status']=='UNKNOWN':
                assert label=='fresh' and answer['model'] is None
                summary['fresh_unknown']+=1
            else:
                assert answer['status']==ref['status']
                if answer['status']=='SAT':assert valid(f,answer['model'],n,answer['goal'])
                else:assert answer['model'] is None
                if label=='source':summary['source_correct']+=1
                if label=='fresh':summary['fresh_known']+=1
            execution_formula=[c[::-1] for c in f[::-1]] if label=='reordered' else f
            assert answer==dict(goal=answer['goal'],**query(cache,execution_formula,answer['goal']))
    assert [r['goal'] for r in row['fresh_repair']]==goals
    for answer,ref in zip(row['fresh_repair'],row['fresh_ref']['goals']):
        assert answer['status']==ref['status']
        if answer['status']=='SAT':assert valid(row['fresh'],answer['model'],n,answer['goal'])
        summary['fresh_repaired']+=1
    summary.update(models=len(cache['models']),payload_bits=row['payload_bits'],
                   serialized_cache_bytes=row['serialized_cache_bytes'],compile_seconds=row['compile_seconds'],
                   reference_calls=2+sum(len(row[k+'_ref']['goals']) for k in ('source','fresh') if row[k+'_ref']['core']['status']=='SAT'),
                   fresh_repair_seconds=sum(r['seconds'] for r in row['fresh_repair']),
                   fresh_repair_conflicts=sum(r['stats']['conflicts'] for r in row['fresh_repair']),
                   fresh_protected_cache=row['fresh_cache'][0]['status'],
                   source_literal_checks=sum(r['meter']['formula_literal_checks'] for r in row['source_cache']),
                   fresh_literal_checks=sum(r['meter']['formula_literal_checks'] for r in row['fresh_cache']))
    return summary


def controls(row,cfg):
    rejected=[]
    for name in ('model','unsat_fact','coverage','context','payload','source_answer','fresh_answer'):
        bad=copy.deepcopy(row)
        if name=='model':bad['cache']['models'][0][0]=0
        if name=='unsat_fact':bad['cache']['unsat_goals'].append(next(r['goal'] for r in row['source_ref']['goals'] if r['status']=='SAT'))
        if name=='coverage':bad['source_ref']['goals'].pop()
        if name=='context':bad['cache']['formula'].pop()
        if name=='payload':bad['payload_bits']+=1
        if name=='source_answer':bad['source_cache'][0]['status']='UNKNOWN'
        if name=='fresh_answer':bad['fresh_cache'][0]['status']='SAT';bad['fresh_cache'][0]['model']=None
        try:check_row(bad,cfg)
        except (AssertionError,KeyError,IndexError,TypeError):rejected.append(name)
        else:raise AssertionError('accepted corruption '+name)
    nontrivial=[[1,2],[1,-2],[-1,2],[-1,-2]]
    check_unsat(nontrivial,['1 0','0'])
    for name,proof in [('invalid_proof',['3 0','0']),('missing_proof_step',['0'])]:
        try:check_unsat(nontrivial,proof)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('accepted '+name)
    assert not conflict([masks([1,-1])],[],dict(clause_visits=0))
    assert conflict([], [1,-1],dict(clause_visits=0))
    cache=dict(formula=[[1]],models=[[1]],unsat_goals=[-1])
    assert query(cache,[[1]],-1)['status']=='UNSAT'
    assert query(cache,[[-1]],-1)['status']=='UNKNOWN', 'cross-context UNSAT leak'
    assert query(cache,[[1, -1]],1)['status']=='SAT', 'valid cross-context witness rejected'
    assert query(cache,[[1, -1]],-1)['status']=='UNKNOWN'
    return dict(rejected=rejected,context_replay_control='PASS',proof_controls='PASS')
