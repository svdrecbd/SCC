import copy
from fractions import Fraction
from proof import check_unsat,conflict,masks


def verify_model(formula,model,goal=None):
    assert model is not None and all(type(v) is int and v!=0 for v in model)
    assignment={abs(v):v>0 for v in model}
    assert len(assignment)==len(model)
    assert all(any(assignment.get(abs(v))==(v>0) for v in clause) for clause in formula)
    if goal is not None:assert assignment.get(abs(goal))==(goal>0)


def verify_variant(row,v):
    n,N=row['n'],row['N'];mapping=v['mapping'];formula=v['formula']
    assert len(mapping)==N+1 and mapping[0]==0
    assert sorted(map(abs,mapping[1:]))==list(range(1,N+1))
    augmented=row['core_formula']+[[n+1,-(n+1),n+2],[n+3,-(n+3),n+4]]
    expected=[[-N,*c] for c in augmented]
    expected=[sorted((mapping[abs(l)] if l>0 else -mapping[abs(l)]) for l in c) for c in expected]
    assert sorted(expected)==sorted(map(sorted,formula))
    shared=set.intersection(*(set(c) for c in formula))
    assert shared=={-mapping[N]}=={v['common']}
    candidates=list(formula[0]);comparisons=0
    for clause in formula[1:]:
        for l in list(candidates):
            comparisons+=clause.index(l)+1 if l in clause else len(clause)
        candidates=[l for l in candidates if l in clause]
    assert v['literal_comparisons']==comparisons
    verify_model(formula,v['native'],-mapping[N])
    goals=list(range(1,N+1))+list(range(-N,0))
    assert [s['goal'] for s in v['statuses']]==goals and len(v['models'])==2*N
    for status,model in zip(v['statuses'],v['models']):
        goal=status['goal'];present=model is not None
        assert status['returned_sat']==present
        if present:verify_model(formula,model,goal)
        assert present==(goal!=mapping[N])
        actual=row['core']['sat'] if goal==mapping[N] else True
        assert status['correct']==(present==actual)
    assert sum(s['correct'] for s in v['statuses'])==2*N-int(row['core']['sat'])
    assert v['repaired']['sat']==row['core']['sat']
    if v['repaired']['sat']:verify_model(formula,v['repaired']['model'],mapping[N])
    else:assert v['repaired']['model'] is None
    assert v['wrong_sign']['sat']
    verify_model(formula,v['wrong_sign']['model'],-mapping[N])
    for result in [v['repaired'],v['wrong_sign']]:
        assert result['seconds']>=0 and all(value>=0 for value in result['stats'].values())


def verify(rows,cfg,out):
    assert [(r['n'],r['seed']) for r in rows]==[(n,s) for n in cfg['dimensions'] for s in cfg['seeds']]
    summary=[];proofs=[];brute_checks=0
    for row in rows:
        n,N=row['n'],row['N'];formula=row['core_formula']
        assert N==n+5 and len(formula)==(43*n+9)//10
        assert all(len(c)==3 and len({abs(v) for v in c})==3 and all(1<=abs(v)<=n for v in c) for c in formula)
        if row['core']['sat']:verify_model(formula,row['core']['model'])
        else:
            assert row['core']['model'] is None
            proof=check_unsat(formula,(out/row['proof_file']).read_text().splitlines())
            proofs.append(dict(n=n,seed=row['seed'],**proof))
        if n==12:
            witnesses=0
            for word in range(1<<n):
                witnesses+=all(any(bool((word >> (abs(v)-1))&1)==(v>0) for v in c) for c in formula)
                brute_checks+=1
            assert row['core']['sat']==(witnesses>0)
        assert [v['transform_seed'] for v in row['variants']]==cfg['transform_seeds']
        for v in row['variants']:verify_variant(row,v)
        v=row['variants'][0]
        summary.append(dict(n=n,N=N,seed=row['seed'],hazard_feasible=row['core']['sat'],
            shortcut_goal_accuracy=str(Fraction(2*N-int(row['core']['sat']),2*N)),
            shortcut_literal_comparisons=[w['literal_comparisons'] for w in row['variants']],
            repair_seconds=[w['repaired']['seconds'] for w in row['variants']],
            repair_conflicts=[w['repaired']['stats']['conflicts'] for w in row['variants']],
            repair_decisions=[w['repaired']['stats']['decisions'] for w in row['variants']]))
    return dict(cases=summary,unsafe_sat=sum(r['core']['sat'] for r in rows),
                unsafe_unsat=sum(not r['core']['sat'] for r in rows),
                natural_variants=sum(len(r['variants']) for r in rows),
                goal_cases=sum(2*r['N']*len(r['variants']) for r in rows),
                brute_assignments=brute_checks,proof_checks=proofs,
                balanced_weight_examples={str(w):str(1-w/2) for w in [Fraction(1,100),Fraction(1,10),Fraction(1,2)]})


def controls(row):
    v=row['variants'][0];passed=[]
    edits=[('goal_coverage',lambda x:x['statuses'].pop()),
           ('plan',lambda x:x['native'].__setitem__(abs(x['common'])-1,-x['native'][abs(x['common'])-1])),
           ('mapping',lambda x:x['mapping'].__setitem__(1,-x['mapping'][1])),
           ('work_count',lambda x:x.__setitem__('literal_comparisons',0)),
           ('hazard_result',lambda x:x['repaired'].__setitem__('sat',not x['repaired']['sat']))]
    for name,edit in edits:
        bad=copy.deepcopy(v);edit(bad)
        try:verify_variant(row,bad)
        except AssertionError:passed.append(name)
        else:raise AssertionError('accepted corruption '+name)
    hard=[[1,2],[1,-2],[-1,2],[-1,-2]]
    check_unsat(hard,['1 0','0'])
    for name,proof in [('invalid_drup_addition',['3 0','0']),('missing_drup_step',['0'])]:
        try:check_unsat(hard,proof)
        except AssertionError:passed.append(name)
        else:raise AssertionError('invalid proof accepted')
    # Tautologies and inconsistent assumptions are unit-propagation edge cases.
    assert not conflict([masks([1,-1])],[],{'clause_visits':0})
    assert conflict([],[-1,1],{'clause_visits':0})
    return passed
