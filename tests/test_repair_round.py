import copy

import pytest
import torch

from scc.generation_guard import GenerationGuard,guard_rows,compare_generation
from scc.numerical_diagnosis import profiled_objective,derivative_sweep
from scc.selective_coupling import recovered_objective
from scc.varied_coupling import optimize_varied
from tests.test_learned_bottleneck import tiny
from tests.test_selective_coupling import fixture


def test_generation_guard_uses_train_cores_all_contexts_and_preserves_casewise_answers():
    rows,ids=guard_rows(1021731,3)
    assert len(rows)==54 and len(ids)==9
    assert len({r['guard_domain'] for r in rows})==18
    assert all(r['split']=='train' for r in rows)
    other,other_ids=guard_rows(1021731,3,ids)
    assert not ids & other_ids and other==guard_rows(1021731,3,ids)[0]
    reference=[{'text':r['target'],'terminated':True} for r in rows]
    assert compare_generation(rows,reference,reference)['passed']
    changed=copy.deepcopy(reference);changed[0]['terminated']=False
    assert not compare_generation(rows,reference,changed)['passed']  # Correct digits without EOS do not pass.
    changed=copy.deepcopy(reference)
    unauth=next(i for i,r in enumerate(rows) if r['category']=='unauthorized')
    changed[unauth]['text']=rows[unauth]['underlying_answer']
    assert not compare_generation(rows,reference,changed)['passed']
    # An equal aggregate score cannot conceal a newly broken original example.
    reference[1]={'text':'wrong','terminated':True}
    changed=copy.deepcopy(reference);changed[1]={'text':rows[1]['target'],'terminated':True}
    changed[0]={'text':'wrong','terminated':True}
    result=compare_generation(rows,reference,changed)
    assert not result['passed'] and result['regression_indices']==[0]


def test_fresh_guard_reuses_within_opportunity_but_changes_across_opportunities(tmp_path,monkeypatch):
    calls=[]
    def predict(model,rows):
        calls.append({r['latent_id'] for r in rows})
        return [{'text':r['target'],'terminated':True} for r in rows]
    monkeypatch.setattr('scc.generation_guard.predictions',predict)
    rows,ids=guard_rows(1021731,2)
    guard=GenerationGuard(object(),tmp_path/'guard',rows,ids,fresh_size=32)
    episode={'excluded_task_ids':ids}
    guard(object(),0,[episode],.01)
    first={r['latent_id'] for r in guard.fresh[0]}
    count=len(calls)
    guard(object(),0,[episode],.001)
    assert len(calls)==count+2 and first=={r['latent_id'] for r in guard.fresh[0]}
    guard(object(),1,[episode],.01)
    second={r['latent_id'] for r in guard.fresh[0]}
    assert len(first)==len(second)==96
    assert len(first & second)<len(first)//10
    assert not ids & (first | second)


def test_guard_rejection_rolls_back_actual_descent_and_observe_control_accepts():
    _,c,data=fixture();model=tiny();origin=copy.deepcopy(model.state_dict())
    calls=[]
    def reject(current,iteration,episodes,radius):
        assert any(not torch.equal(p,origin[n]) for n,p in current.state_dict().items())
        calls.append((iteration,radius));return {'passed':False,'reason':'fixture behavioral rejection'}
    kw=dict(iterations=1,radii=(.001,),minimum_decrease=1e-8,trial_guard=reject)
    args=(model,c,lambda i:[data],data['queries'],lambda i:(data['queries'],{}))
    rejected,fit=optimize_varied(*args,**kw)
    trial=fit['history'][0]['trials'][0]
    assert trial['base_admissible'] and not trial['admissible'] and fit['accepted_steps']==0
    assert all(torch.equal(p,origin[n]) for n,p in rejected.state_dict().items())
    accepted,control=optimize_varied(*args,enforce_guard=False,**kw)
    trial=control['history'][0]['trials'][0]
    assert trial['admissible'] and not trial['behavior_guard']['passed'] and control['accepted_steps']==1
    assert len(calls)==2 and any(not torch.equal(p,origin[n]) for n,p in accepted.state_dict().items())


def test_profiled_unroll_matches_objective_and_full_gradient_and_sweep_restores_parent():
    _,c,data=fixture();c.update(inner_scope='core',repair_scope='all')
    model=tiny();origin=copy.deepcopy(model.state_dict())
    v,_=recovered_objective(model,c,data);g=torch.autograd.grad(v,tuple(model.parameters()))
    p,profile=profiled_objective(model,c,data);pg=torch.autograd.grad(p,tuple(model.parameters()))
    torch.testing.assert_close(v,p,atol=0,rtol=0)
    for a,b in zip(g,pg,strict=True):torch.testing.assert_close(a,b,atol=0,rtol=0)
    assert len(profile['trace']['modification'])==len(profile['trace']['repair'])==2
    result=derivative_sweep(model,c,data,epsilons=(1e-5,3e-6))
    assert len(result['checks'])==4
    assert all(x['relative_error']<.001 for x in result['checks'])
    assert all(torch.equal(p,origin[n]) for n,p in model.state_dict().items())
    for check in result['checks']:
        for side in check['sides']:
            assert abs(side['realized_l2']-check['epsilon_l2']) < 1e-12


def test_sweep_restores_parent_after_callback_failure():
    _,c,data=fixture();model=tiny();origin=copy.deepcopy(model.state_dict())
    def stop(_):raise RuntimeError('fixture interruption')
    with pytest.raises(RuntimeError,match='fixture interruption'):
        derivative_sweep(model,c,data,epsilons=(1e-5,),callback=stop)
    assert all(torch.equal(p,origin[n]) for n,p in model.state_dict().items())
