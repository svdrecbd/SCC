import argparse
import importlib.util
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from scc.persistent_reference import PersistentGRU
from scc.persistent_tasks import training_requests, tensors


def fixture():
    torch.manual_seed(107)
    model = PersistentGRU(8).double()
    ids, labels = tensors(training_requests(308, 2001, 2, 4), device='cpu')
    return model, ids, labels


def test_continuation_and_explicit_tick_match_without_request_resets():
    model, ids, _ = fixture()
    with torch.no_grad():
        full, final = model(ids)
        first, middle = model(ids[:, :2])
        rest, continued = model(ids[:, 2:], middle)
        assert torch.allclose(full, torch.cat([first, rest], 1), atol=1e-12, rtol=1e-12)
        assert torch.allclose(final, continued, atol=1e-12, rtol=1e-12)
        h = torch.zeros(2, 8, dtype=torch.float64)
        outputs = []
        for request in range(4):
            for token in range(19):
                y, h = model.manual_tick(ids[:, request, token], h)
            outputs.append(y)
        assert torch.allclose(full, torch.stack(outputs, 1), atol=1e-12, rtol=1e-12)
        assert torch.allclose(final[0], h, atol=1e-12, rtol=1e-12)
        reset, _ = model(ids[:, 2:])
        assert not torch.equal(reset, rest)


@pytest.mark.parametrize('name,index', [
    ('recurrent.weight_ih_l0', (20, 11)),
    ('recurrent.weight_hh_l0', (22, 3)),
    ('readout.weight', (2, 4)),
])
def test_full_window_gradients_match_centered_finite_difference(name, index):
    model, ids, labels = fixture()
    def objective():
        outputs, _ = model(ids)
        return F.cross_entropy(outputs.reshape(-1, 4), labels.reshape(-1))
    objective().backward()
    parameter = dict(model.named_parameters())[name]
    exact = float(parameter.grad[index])
    epsilon = 1e-5
    with torch.no_grad():
        original = float(parameter[index])
        parameter[index] = original+epsilon
        upper = float(objective())
        parameter[index] = original-epsilon
        lower = float(objective())
        parameter[index] = original
    assert abs(exact) > 1e-9
    assert exact == pytest.approx((upper-lower)/(2*epsilon), rel=2e-5, abs=1e-9)


def test_checkpoint_reload_preserves_continuation(tmp_path):
    model, ids, labels = fixture()
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    y, _ = model(ids)
    F.cross_entropy(y.reshape(-1, 4), labels.reshape(-1)).backward()
    optimizer.step()
    with torch.no_grad():
        _, middle = model(ids[:, :2])
        expected, expected_state = model(ids[:, 2:], middle)
        torch.save({'model': model.state_dict(), 'state': middle}, tmp_path/'state.pt')
        saved = torch.load(tmp_path/'state.pt', weights_only=True)
        other = PersistentGRU(8).double()
        other.load_state_dict(saved['model'])
        actual, actual_state = other(ids[:, 2:], saved['state'])
    assert torch.equal(actual, expected) and torch.equal(actual_state, expected_state)


def test_fixture_and_changed_resources_cannot_claim_declared_reference():
    path = Path(__file__).resolve().parents[1]/'scripts/run_persistent_reference.py'
    spec = importlib.util.spec_from_file_location('reference_runner', path)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    args = argparse.Namespace(device='cpu', width=128, seed=17, data_seed=24017,
                              steps=6000, batch=32, window=4, lr=.003,
                              eval_per_cell=128, eval_streams=16, wall_seconds=540,
                              total_seconds=600, fixture=False)
    assert runner.is_declared(args)
    assert runner.is_declared(argparse.Namespace(**{**vars(args), 'steps': 12000, 'extension': True}))
    assert runner.is_declared(argparse.Namespace(**{**vars(args), 'steps': 12000, 'extension': True, 'stabilize': True}))
    assert not runner.is_declared(argparse.Namespace(**{**vars(args), 'extension': True}))
    assert not runner.is_declared(argparse.Namespace(**{**vars(args), 'stabilize': True}))
    for field, value in [('fixture', True), ('steps', 5999), ('device', 'cuda'),
                         ('window', 1), ('eval_per_cell', 127), ('wall_seconds', 600)]:
        assert not runner.is_declared(argparse.Namespace(**{**vars(args), field: value}))


def test_matrix_control_learning_rate_changes_after_exactly_6000_updates():
    path = Path(__file__).resolve().parents[1]/'scripts/run_persistent_learnability.py'
    spec = importlib.util.spec_from_file_location('matrix_runner', path)
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    assert runner.reference_learning_rate(0) == .003
    assert runner.reference_learning_rate(5999) == .003
    assert runner.reference_learning_rate(6000) == .0003
    assert runner.reference_learning_rate(11999) == .0003
