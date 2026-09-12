import copy
from dataclasses import asdict

import pytest
import torch
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel

from scc.checkpoint import load_checkpoint, save_checkpoint
from scc.coupling import nll
from scc.developmental_tasks import TaskStream, batch_rows
from scc.learned_bottleneck import BottleneckConfig, LearnedBottleneck, editable_names, from_checkpoint
from scc.fixed_episode_optimization import within_anchor


def tiny(shared=True):
    torch.set_num_threads(2); torch.manual_seed(172)
    return LearnedBottleneck(BottleneckConfig(width=16, heads=2, layers=3,
                                             bottleneck_width=8, shared_core=shared)).double().eval()


@pytest.mark.parametrize('shared', [True, False])
def test_causal_state_flow_has_no_bottleneck_bypass_and_checkpoint_round_trips(shared, tmp_path):
    model = tiny(shared)
    tokens = torch.randint(0, 260, (2, 12))
    changed = tokens.clone(); changed[:, 7:] = (changed[:, 7:] + 13) % 260
    with torch.no_grad():
        before = model(tokens)
        torch.testing.assert_close(before[:, :7], model(changed)[:, :7], atol=1e-12, rtol=0)
        torch.testing.assert_close(before, functional_call(model, dict(model.named_parameters()), (tokens,), strict=True))
    assert len(model.cells) == (1 if shared else 3)
    visits = []
    hooks = [c.register_forward_hook(lambda *_: visits.append(1)) for c in model.cells]
    model(tokens)
    for hook in hooks:
        hook.remove()
    assert len(visits) == 3
    lesioned = copy.deepcopy(model)
    with torch.no_grad():
        lesioned.cells[0].compress.weight.zero_()
        # The first state projection really is mandatory for token information.
        torch.testing.assert_close(lesioned(tokens), lesioned(changed), atol=1e-12, rtol=0)
        assert not torch.allclose(lesioned.with_bypass()(tokens), lesioned.with_bypass()(changed))
    bypassed = model.with_bypass()
    state = {'schema_version': 1, 'model': bypassed.state_dict(),
             'configuration': {'architecture': 'learned-bottleneck/v1', 'model': asdict(bypassed.config)}}
    save_checkpoint(tmp_path / 'model.pt', state)
    restored = from_checkpoint(load_checkpoint(tmp_path / 'model.pt')).double()
    # Loading constructs FP32 by default; convert the fixture to FP32 before
    # comparison so the test reflects the actual stored inference precision.
    restored = restored.float(); bypassed = bypassed.float()
    torch.testing.assert_close(restored(tokens), bypassed(tokens), atol=0, rtol=0)
    assert restored.config.bypass_bottleneck
    assert not model.config.bypass_bottleneck


def test_shared_core_accumulates_the_gradient_of_all_recurrent_uses():
    shared = tiny(True)
    untied = tiny(False)
    state = untied.state_dict()
    for name, value in shared.state_dict().items():
        if name.startswith('cells.0.'):
            for index in range(3):
                state[name.replace('cells.0.', f'cells.{index}.')] = value.clone()
        else:
            state[name] = value.clone()
    untied.load_state_dict(state)
    rows = TaskStream(154).rows(2, 'lookup', 'authorized')
    batch = batch_rows(rows)
    with sdpa_kernel(SDPBackend.MATH):
        a = nll(shared, dict(shared.named_parameters()), batch)
        b = nll(untied, dict(untied.named_parameters()), batch)
        a.backward(); b.backward()
    torch.testing.assert_close(a, b, atol=1e-12, rtol=0)
    for name, p in shared.named_parameters():
        expected = (sum(dict(untied.named_parameters())[name.replace('cells.0.', f'cells.{i}.')].grad for i in range(3))
                    if name.startswith('cells.0.') else dict(untied.named_parameters())[name].grad)
        torch.testing.assert_close(p.grad, expected, atol=1e-10, rtol=1e-10)


def test_reader_scope_excludes_embeddings_and_core_and_anchor_checks_every_domain():
    model = tiny()
    reader = editable_names(model, 'reader'); core = editable_names(model, 'core')
    assert reader and core and not set(reader) & set(core)
    assert all(name.startswith(('norm.', 'output.')) for name in reader)
    assert 'tokens.weight' not in reader and 'positions.weight' not in reader
    assert set(editable_names(model, 'all')) == {name for name, _ in model.named_parameters()}
    assert within_anchor({'a': 1.04, 'b': 2.04}, {'a': 1., 'b': 2.})
    assert not within_anchor({'a': 1., 'b': 2.051}, {'a': 1., 'b': 2.})
    assert not within_anchor({'a': 1.}, {'a': 1., 'b': 2.})
    assert not within_anchor({'a': float('nan')}, {'a': 1.})


def test_fixed_episode_steps_descend_or_rollback_and_preserve_parent(tmp_path):
    from tests.test_selective_coupling import fixture
    from scc.fixed_episode_optimization import directional_check, optimize
    from scc.selective_coupling import recovered_objective
    # Use the existing independently checked unroll fixture with the new
    # recurrent architecture. No fake loss stands in for the coupling objective.
    _, c, data = fixture()
    model = tiny(); origin = copy.deepcopy(model.state_dict())
    anchors = data['queries']
    gate = directional_check(model, c, data, epsilons=(1e-5, 3e-6, 1e-6))
    assert gate['passed']
    fitted, record = optimize(model, c, data, anchors, iterations=2, radii=(.001, .0003, .0001), minimum_decrease=1e-8)
    assert record['accepted_steps'] == 2
    assert all(r['after'] < r['before'] for r in record['history'])
    assert all(torch.equal(v, origin[k]) for k, v in model.state_dict().items())
    for p in model.parameters():
        assert p.grad is None
    rolled_back, failed = optimize(model, c, data, anchors, iterations=1, radii=(.001,), minimum_decrease=100.)
    assert failed['accepted_steps'] == 0
    assert all(torch.equal(v, origin[k]) for k, v in rolled_back.state_dict().items())
