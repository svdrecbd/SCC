"""Failures must preserve evidence and never grant learnability qualification."""
from dataclasses import asdict
import json
import sys

import pytest
import torch

from scripts import run_persistent_feedback as runner
from scc.persistent_matrix import MatrixConfig
from scc.persistent_tasks import input_code
from scc.persistent_feedback import feedback_wiring
from scc.provenance import file_digest


@pytest.fixture
def two_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def inject_live_failure(monkeypatch, kind):
    original = runner.LiveFeedbackMatrix.tick

    def tick(self, inputs):
        y, controls = original(self, inputs)
        if kind == 'transition':
            raise ValueError('Injected rejected transition')
        if kind == 'offset':
            y = y + .01  # All decisions can match while the numerical gate fails.
        elif kind == 'decision':
            y = y.clone()
            y[1] += 1.
        elif kind == 'nonfinite':
            y = y * float('nan')
        return y, controls

    monkeypatch.setattr(runner.LiveFeedbackMatrix, 'tick', tick)


@pytest.mark.parametrize('kind', ['none', 'offset', 'decision', 'nonfinite', 'transition'])
def test_evaluator_finishes_diagnostics_and_preserves_failed_replay(tmp_path, monkeypatch, two_threads, kind):
    inject_live_failure(monkeypatch, kind)
    config = MatrixConfig(32, 4, 'soft')
    weights = torch.zeros(config.rows, config.input_size)
    folder = tmp_path/'evaluation'
    result = runner.evaluate(weights, config, input_code(32, 'anchor'), feedback_wiring(config),
                             folder, per_cell=8, streams=8)
    assert json.loads((folder/'result.json').read_text()) == result
    assert set(result) == {'continuous', 'reset-1', 'reset-4'}
    for mode in result:
        assert len((folder/(mode+'.jsonl')).read_text().splitlines()) == 144
        assert file_digest(folder/(mode+'.jsonl')) == result[mode]['prediction_sha256']
    continuous = result['continuous']
    assert continuous['numerical_validation_passed'] == (kind == 'none')
    assert continuous['numerical_logit_tolerance'] == 1e-4
    rows = [json.loads(line) for line in (folder/'single-stream.jsonl').read_text().splitlines()]
    assert len(rows) == continuous['single_stream_requests_completed']
    assert file_digest(folder/'single-stream.jsonl') == continuous['single_stream_prediction_sha256']
    if kind in ('nonfinite', 'transition'):
        assert continuous['single_stream_failure'] is not None
        assert not continuous['single_stream_decisions_match']
    elif kind == 'offset':
        assert continuous['single_stream_decisions_match']
        assert continuous['single_stream_maximum_logit_error'] == pytest.approx(.01)
    elif kind == 'decision':
        assert continuous['single_stream_decision_mismatches'] == len(rows)
    saved = torch.load(folder/'final-live-states.pt', weights_only=True)
    assert saved['current_weights'].shape == (8, config.rows, 32)
    assert torch.count_nonzero(weights) == 0


def test_training_reports_failure_only_after_saving_all_results(tmp_path, monkeypatch, two_threads):
    inject_live_failure(monkeypatch, 'offset')
    config = MatrixConfig(32, 4, 'soft')
    conditions = tmp_path/'initial.json'
    conditions.write_text(json.dumps({'schema': 'persistent-feedback-initial-conditions/v1',
        'configuration': asdict(config), 'initial_seed': 17, 'feedback_seed': 130913,
        'feedback_strength': 1., 'initial_weights': torch.zeros(config.rows, 32).tolist(),
        'fixed_wiring': feedback_wiring(config).tolist()}))
    plan = tmp_path/'plan.md'
    plan.write_text('Injected validation-failure regression fixture; no scientific evidence.')
    output = tmp_path/'run'
    inline = tmp_path/'provider-result.json'
    monkeypatch.setenv('GMN_RESULT_PATH', str(inline))
    monkeypatch.setattr(sys, 'argv', ['runner', '--output', str(output), '--plan', str(plan),
        '--initial-conditions', str(conditions), '--width', '32', '--steps', '1',
        '--batch', '2', '--window', '1', '--eval-per-cell', '8', '--eval-streams', '8',
        '--wall-seconds', '30', '--total-seconds', '60'])
    # Isolate result handling from startup checks and scientific qualification.
    monkeypatch.setattr(runner, 'numerical_gate', lambda *args: {'passed': True})
    monkeypatch.setattr(runner, 'declared', lambda args: True)
    summarize = runner.summarize_records
    monkeypatch.setattr(runner, 'summarize_records', lambda rows: {**summarize(rows), 'qualified': True})
    with pytest.raises(ValueError, match='all evaluation records saved'):
        runner.main()
    result = json.loads((output/'result.json').read_text())
    assert result['training_complete'] and result['completed_steps'] == 1
    assert result['status'] == 'evaluation_validation_failed'
    assert not result['qualified_learnability'] and not result['evaluation_validation_passed']
    assert set(result['evaluation']) == {'continuous', 'reset-1', 'reset-4'}
    assert (output/'trained.pt').is_file() and (output/'failure.json').is_file()
    assert not json.loads(inline.read_text())['ok']
