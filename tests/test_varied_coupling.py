import copy
import time

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from scc.coupling import nll
from scc.learned_bottleneck import editable_names
from scc.selective_coupling import rollout, recovered_objective
from scc.varied_coupling import optimize_varied, pool_score
from tests.test_learned_bottleneck import tiny
from tests.test_selective_coupling import fixture
from tests.test_developmental import bank


@pytest.mark.parametrize('repair_scope', ['core', 'all'])
def test_masked_trajectory_matches_stock_adam_and_preserves_unedited_tensors(repair_scope):
    _, c, data = fixture()
    c.update(inner_scope='core', repair_scope=repair_scope)
    model = tiny(); independent = copy.deepcopy(model)
    changed, repaired = rollout(model, c, data, create_graph=False)
    for name, p in model.named_parameters():
        if name not in editable_names(model, 'core'):
            assert changed[name] is p  # Preserves the outer autograd connection.
            if repair_scope == 'core':
                assert repaired[name] is p
    for steps, lr, scope, expected in [(data['modifications'], c['inner_lr'], 'core', changed),
                                       (data['repairs'], c['repair_lr'], repair_scope, repaired)]:
        names = editable_names(independent, scope)
        params = [p for name, p in independent.named_parameters() if name in names]
        optimizer = torch.optim.AdamW(params, lr=lr, betas=(.9, .95), eps=c['inner_epsilon'],
                                      weight_decay=0., foreach=False)
        with sdpa_kernel(SDPBackend.MATH):
            for attack, replay, refusal, floor in steps:
                independent.zero_grad(set_to_none=True)
                parameters = dict(independent.named_parameters())
                value = ((nll(independent, parameters, attack) + 3*nll(independent, parameters, replay)/floor)/4
                         + .5*nll(independent, parameters, refusal))
                value.backward(); torch.nn.utils.clip_grad_norm_(params, 1.); optimizer.step()
        for name, p in independent.named_parameters():
            torch.testing.assert_close(p, expected[name], atol=1e-12, rtol=0)


@pytest.mark.parametrize('outer_group', ['core', 'frozen_in_inner'])
def test_full_outer_derivative_includes_parameters_frozen_during_inner_edit(outer_group):
    _, c, data = fixture(); c.update(inner_scope='core', repair_scope='core')
    model = tiny(); origin = copy.deepcopy(model.state_dict())
    value, _ = recovered_objective(model, c, data)
    gradients = torch.autograd.grad(value, tuple(model.parameters()))
    selected = [g if (name.startswith('cells.') == (outer_group == 'core')) else torch.zeros_like(g)
                for (name, _), g in zip(model.named_parameters(), gradients, strict=True)]
    norm = torch.stack([g.square().sum() for g in selected]).sum().sqrt()
    assert norm > 0
    outcomes = []
    for sign in (-1, 1):
        with torch.no_grad():
            for (name, p), g in zip(model.named_parameters(), selected, strict=True):
                p.copy_(origin[name] + sign*1e-6*g/norm)
        score, _ = recovered_objective(model, c, data, create_graph=False)
        outcomes.append(float(score.detach()))
    numerical = (outcomes[1] - outcomes[0])/2e-6
    assert abs(numerical-float(norm))/float(norm) < 1e-4


def test_varied_minibatch_descent_rollback_and_fixed_parent_anchors():
    _, c, data = fixture(); c.update(inner_scope='core', repair_scope='all')
    model = tiny(); origin = copy.deepcopy(model.state_dict())
    # Distinct real target queries change the second episode's objective.
    other = dict(data, target_queries={'lookup/selected_exception': data['queries']['lookup/ungated']})
    calls = []
    def episodes(i):
        calls.append(i)
        return [data, other] if i % 2 == 0 else [other, data]
    fresh = lambda i: (data['queries'], {'ordinal': i})
    score, records, pooled = pool_score(model, c, [data, other], gradient=True)
    separate = [pool_score(model, c, [d], gradient=True) for d in [data, other]]
    assert score == sum(r[0] for r in separate)/2
    for g, a, b in zip(pooled, separate[0][2], separate[1][2], strict=True):
        torch.testing.assert_close(g, (a+b)/2, atol=1e-12, rtol=1e-12)
    fitted, record = optimize_varied(model, c, episodes, data['queries'], fresh,
                                     iterations=2, radii=(.001, .0003), minimum_decrease=1e-8)
    assert calls == [0, 1] and record['accepted_steps'] == 2
    assert all(r['after'] < r['before'] for r in record['history'])
    assert all(torch.equal(p, origin[n]) for n, p in model.state_dict().items())
    assert any(not torch.equal(p, origin[n]) for n, p in fitted.state_dict().items())
    rejected, failure = optimize_varied(model, c, episodes, data['queries'], fresh,
                                        iterations=2, radii=(.001,), minimum_decrease=100)
    assert failure['accepted_steps'] == 0 and failure['completed_iterations'] == 2
    assert all(torch.equal(p, origin[n]) for n, p in rejected.state_dict().items())
    interrupted, limited = optimize_varied(model, c, episodes, data['queries'], fresh,
                                           iterations=2, deadline=time.monotonic()-1)
    assert limited['stop_reason'] == 'training_wall_limit' and limited['completed_iterations'] == 0
    assert all(torch.equal(p, origin[n]) for n, p in interrupted.state_dict().items())


def test_reserved_examples_stay_excluded_across_all_phase_boundaries(bank):
    from scc.selective_coupling import defaults, make_episode
    from scripts.run_varied_coupling import exclusions, anchors
    c = defaults(); c.update(inner_steps=2, repair_steps=2, inner_scope='core', repair_scope='all')
    monitor = make_episode(bank, c, 95000, 'cpu')
    reserved = exclusions([monitor])
    training = make_episode(bank, c, 97000, 'cpu', **reserved)
    repeat = make_episode(bank, c, 97000, 'cpu', **reserved)
    assert training['record'] == repeat['record']
    assert training['record']['external_exclusions_verified']
    assert not training['excluded_task_ids'] & reserved['excluded_task_ids']
    assert all(not training['excluded_text_ids'][s] & ids for s, ids in reserved['excluded_text_ids'].items())
    _, record, used = anchors(bank, 'cpu', 993731, 2, exclusions([monitor, training]))
    assert record['reserved_exclusions_verified']
    assert not used['excluded_task_ids'] & (monitor['excluded_task_ids'] | training['excluded_task_ids'])
    assert all(not used['excluded_text_ids'][s] & (monitor['excluded_text_ids'][s] | training['excluded_text_ids'][s])
               for s in used['excluded_text_ids'])
