import pytest
import torch
from torch.nn import functional as F
from scc.provenance import atomic_json
from scc.sharded_repair import functional_window
from scripts.run_curriculum_repair import (configuration, active_length, learning_rate,
    make_training, make_evaluation, objective, calibration_passes, write_data)
from scripts.audit_curriculum_repair import audit_data
from test_sharded_repair import example


@pytest.mark.parametrize('ordinal,length,lr', [
    (0, 2, .003), (399, 2, .003), (400, 4, .003), (999, 4, .003),
    (1000, 8, .003), (1999, 8, .003), (2000, 12, .003),
    (5999, 12, .003), (6000, 12, .0003), (11999, 12, .0003)])
def test_curriculum_and_lr_boundaries(ordinal, length, lr):
    config = configuration()
    assert active_length(ordinal, config) == length
    assert learning_rate(ordinal, config) == lr


def test_heldout_excludes_calibration_cores_at_full_panel_size():
    dev = make_evaluation(501, 128)
    cores = {r['core_sha256'] for s in dev for r in s}
    heldout = make_evaluation(502, 128, cores)
    assert not cores & {r['core_sha256'] for s in heldout for r in s}
    assert sum(map(len, heldout)) == 768


def test_data_audit_rejects_wrong_curriculum_and_reproduces_matched_schedule(tmp_path):
    config = configuration(True)
    config['pool_per_length'] = 12
    config['train_probe_size'] = 12
    atomic_json(tmp_path / 'configuration.json', config)
    pool, schedule, _ = write_data(tmp_path / 'data', config, config['calibration_seeds'])
    again = make_training(config, *config['calibration_seeds'])
    assert pool == again[0] and torch.equal(schedule, again[1])
    atomic_json(tmp_path / 'evaluation.json', make_evaluation(99, config['evaluation_per_cell']))
    audit_data(tmp_path, tmp_path / 'data', tmp_path / 'evaluation.json')
    schedule[0, 0, 0] = next(i for i, r in enumerate(pool) if r['active_length'] == 12)
    torch.save(schedule, tmp_path / 'data/schedule.pt')
    with pytest.raises(AssertionError):
        audit_data(tmp_path, tmp_path / 'data', tmp_path / 'evaluation.json')


@pytest.mark.parametrize('rule', ['learned', 'symbolic'])
@pytest.mark.parametrize('mask', [0, 9, 11, 15])
def test_four_requests_preserve_actual_commit_and_history(rule, mask):
    model, ids = example(mask)
    model.storage_rule = rule
    ids = torch.cat((ids, ids.flip(-1)), 1)
    payload = model.bank[0, [0, 4]].flatten().clone()
    out, shards, hidden = functional_window(payload, model.decode()[1], ids, 4, rule)
    for j in range(4):
        actual = model.request(ids[:, j])
        for key in ('logits', 'policy_logits'):
            torch.testing.assert_close(actual[key], out[key][:, j], atol=1e-11, rtol=1e-11)
        assert torch.equal(actual['admitted'], out['admitted'][:, j])
    torch.testing.assert_close(shards, model.bank[:, [0, 4]], atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(hidden, model.decode()[1], atol=1e-11, rtol=1e-11)


def test_objective_has_exact_per_request_weighting():
    gen = torch.Generator().manual_seed(89)
    logits = torch.randn(32, 4, 4, generator=gen, dtype=torch.float64, requires_grad=True)
    raw = torch.randn(32, 4, 4, generator=gen, dtype=torch.float64, requires_grad=True)
    labels = torch.arange(128).remainder(3).reshape(32, 4)
    target = torch.arange(32)[:, None].expand(32, 4) < 16
    actual, _ = objective({'logits': logits, 'policy_logits': raw}, labels, target)
    desired = torch.tensor([1., 0., 0., 1.], dtype=raw.dtype).expand_as(raw).clone()
    desired[:, :, 1] = target.double()
    mask = torch.zeros_like(raw, dtype=torch.bool)
    mask[:, :, 1] = target
    expected = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
    expected += .5 * F.binary_cross_entropy_with_logits(raw[mask], desired[mask])
    expected += .5 * F.binary_cross_entropy_with_logits(raw[~mask], desired[~mask])
    torch.testing.assert_close(actual, expected)
    a = torch.autograd.grad(actual, (logits, raw), retain_graph=True)
    b = torch.autograd.grad(expected, (logits, raw))
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y)


def passing_summary():
    return {'status': 'complete', 'precision_agreement': {'passed': True}, 'results': [
        {'endpoint': 'final', 'panel': 'validation', 'precision': p, 'recovery_qualified': True}
        for p in ('fp32', 'fp64')]}


@pytest.mark.parametrize('failure', ['gate', 'audit', 'precision', 'missing-view', 'duplicate-view', 'status'])
def test_dispatch_requires_both_precisions_science_and_integrity(failure):
    summary = passing_summary()
    audit = {'passed': True}
    assert calibration_passes(summary, audit)
    if failure == 'gate':
        summary['results'][1]['recovery_qualified'] = False
    elif failure == 'audit':
        audit['passed'] = False
    elif failure == 'precision':
        summary['precision_agreement']['passed'] = False
    elif failure == 'missing-view':
        summary['results'].pop()
    elif failure == 'duplicate-view':
        summary['results'][1]['precision'] = 'fp32'
    else:
        summary['status'] = 'numerical-validation-failed'
    assert not calibration_passes(summary, audit)


@pytest.mark.parametrize('qualifies', [False, True])
def test_execute_only_opens_comparison_after_gate_and_runs_all_pairs(tmp_path, monkeypatch, qualifies):
    import scripts.run_curriculum_repair as runner
    import scripts.audit_curriculum_repair as auditor
    from scripts.run_sharded_rewrite_cell import pack_state
    model, ids = example(15)
    out = model.request(ids[:, 0])
    torch.save({'out': out, 'post_event_state': pack_state(model)}, tmp_path / 'merged-start-parent.pt')
    events = []
    config = configuration()

    def evaluation(seed, per_cell, excluded=()):
        events.append(('evaluation', seed))
        if seed == config['comparison_evaluation_seed']:
            assert qualifies and ('case', 'calibration') in events
            assert set(excluded) == {'development-core'}
        return [[{'core_sha256': 'development-core'}] for _ in range(4)]

    def dataset(folder, config, seeds):
        events.append(('data', tuple(seeds)))

    def case(root, name, arm, dataset, panel, *args):
        events.append(('case', name))
        if name != 'calibration':
            assert qualifies and panel.name == 'comparison-evaluation.json'
        summary = passing_summary()
        if not qualifies:
            summary['results'][0]['recovery_qualified'] = False
        return summary, {'passed': True}

    monkeypatch.setattr(runner, 'make_evaluation', evaluation)
    monkeypatch.setattr(runner, 'write_data', dataset)
    monkeypatch.setattr(runner, 'run_case', case)
    monkeypatch.setattr(auditor, 'audit_batch', lambda root: {'passed': True})
    result = runner.execute(tmp_path, config)
    assert result['matched_comparison_started'] == qualifies
    assert len(result['completed_cases']) == (7 if qualifies else 1)
    assert (tmp_path / 'comparison-evaluation.json').exists() == qualifies
    if qualifies:
        assert [event[1] for event in events if event[0] == 'data'] == [
            tuple(config['calibration_seeds']), *(tuple(s) for s in config['paired_seeds'])]
        assert result['completed_cases'][1:] == [
            f'pair-{i}-{arm}' for i in (1, 2, 3) for arm in runner.ARMS]
