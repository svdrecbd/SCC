import copy
from dataclasses import asdict
import pytest
import torch
from torch.nn.attention import SDPBackend,sdpa_kernel
from scc.coupling import nll
from scc.developmental_tasks import FAMILIES,TaskStream,batch_rows
from scc.model import ModelConfig,Transformer
from scc.pilot_objectives import vector_norm
from scc.selective_coupling import Target,defaults,edit_loss,rollout,recovered_objective


def fixture():
    torch.set_num_threads(2);torch.manual_seed(691)
    model=Transformer(ModelConfig(width=16,layers=1,heads=2,context_length=192)).double().eval()
    stream=TaskStream(991);target=Target();steps=[]
    for i in range(4):
        rows=stream.rows(2,'lookup','unauthorized');attack=batch_rows(target.apply(rows),disclose=True)
        replay=batch_rows(stream.rows(2,FAMILIES[i%3],'authorized'))
        rows=[r for r in stream.rows(16,'lookup','unauthorized') if not target.matches(r)][:2]
        steps.append((attack,replay,batch_rows(rows),2.302585092994046))
    queries={f+'/ungated':batch_rows(stream.rows(2,f,'ungated')) for f in FAMILIES}
    data={'modifications':steps[:2],'repairs':steps[2:],'supports':list(queries.values()),'queries':queries,
          'target_queries':{'lookup/selected_exception':batch_rows(target.apply(stream.rows(2,'lookup','unauthorized')),disclose=True)},'record':{}}
    c=defaults();c.update(inner_steps=2,repair_steps=2,inner_lr=.001,repair_lr=.0003)
    return model,c,data


def test_selected_exception_transformation_is_precise_and_preserves_oracle():
    for family in FAMILIES:
        for reordered in (False,True):
            stream=TaskStream(471);target=Target(family,'Y','Z');rows=stream.rows(16,family,'unauthorized',reordered)
            changed=target.apply(rows)
            assert all(target.matches(r) for r in changed)
            assert [r['underlying_answer'] for r in changed]==[r['underlying_answer'] for r in rows]
            assert [r['latent_id'] for r in changed]==[r['latent_id'] for r in rows]
            batch_rows(changed,disclose=True)  # The independent prompt oracle checks answers and refusal metadata.
            assert not any(target.matches(r) for r in stream.rows(16,family,'authorized',reordered))
    with pytest.raises(ValueError):Target('lookup','W','W')


def test_selective_unroll_matches_independent_stock_adam_with_fresh_repair():
    model,c,data=fixture();copy_model=copy.deepcopy(model)
    first,last=rollout(model,c,data,create_graph=False)
    for steps,lr,expected in [(data['modifications'],c['inner_lr'],first),(data['repairs'],c['repair_lr'],last)]:
        optimizer=torch.optim.AdamW(copy_model.parameters(),lr=lr,betas=(.9,.95),eps=c['inner_epsilon'],weight_decay=0.,foreach=False)
        with sdpa_kernel(SDPBackend.MATH):
            for attack,replay,refusal,floor in steps:
                optimizer.zero_grad(set_to_none=True);params=dict(copy_model.named_parameters())
                # Independent assembly checks the selective objective's terms and weights.
                loss=(nll(copy_model,params,attack)+3*nll(copy_model,params,replay)/floor)/4+.5*nll(copy_model,params,refusal)
                loss.backward();torch.nn.utils.clip_grad_norm_(copy_model.parameters(),1.);optimizer.step()
        for name,p in copy_model.named_parameters():assert torch.allclose(p,expected[name],atol=1e-12,rtol=0),name


def test_full_selective_derivative_matches_rerunning_edit_and_reader_fitting():
    model,c,data=fixture();origin=copy.deepcopy(model.state_dict())
    value,_=recovered_objective(model,c,data);gradients=torch.autograd.grad(value,tuple(model.parameters()));norm=vector_norm(gradients)
    assert torch.isfinite(norm) and norm>0
    outcomes=[]
    for sign in (-1,1):
        with torch.no_grad():
            for (name,p),g in zip(model.named_parameters(),gradients):p.copy_(origin[name]+sign*1e-5*g/norm)
        v,_=recovered_objective(model,c,data,create_graph=False);outcomes.append(float(v.detach()))
    numerical=(outcomes[1]-outcomes[0])/2e-5
    assert abs(numerical-float(norm))/float(norm)<1e-4
    assert outcomes[0]<float(value.detach())


def test_calibration_requires_joint_target_capability_and_other_refusals():
    from scripts.run_selective_construction_screen import eligible
    before={'text':{'a':{'context_gain':1.},'b':{'context_gain':2.}}}
    after={'layouts':{k:{'target':1.,'benign_min':1.,'other_refusal':1.} for k in ('original','reordered')},
           'text':{'a':{'context_gain':.95},'b':{'context_gain':1.9}}}
    assert eligible(after,before)
    for key in ('target','benign_min','other_refusal'):
        changed=copy.deepcopy(after);changed['layouts']['reordered'][key]=.899
        assert not eligible(changed,before)
    changed=copy.deepcopy(after);changed['text']['b']['context_gain']=1.89
    assert not eligible(changed,before)
