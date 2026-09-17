"""Fail-closed evidence audits and a useful, state-preserving forward attack."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_wide_trajectory as runner
from scripts.audit_wide_trajectory import audit


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value))


def rehash(directory):
    manifest = read(directory / 'sha256.json')
    for name in manifest:
        path = directory / name
        if path.is_file():
            manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(directory / 'sha256.json', manifest)


@pytest.fixture(scope='module')
def complete_run(tmp_path_factory):
    root = tmp_path_factory.mktemp('wide-audit')
    plan = root / 'plan.md'
    plan.write_text('Generated implementation fixture, not scientific evidence.\n')
    out = root / 'run'
    cmd = [sys.executable, str(ROOT/'scripts/run_wide_trajectory.py'), '--out', str(out),
           '--plan', str(plan), '--widths', '8', '--rounds', '1,2', '--contexts', '2',
           '--samples', '3', '--burn-in', '1', '--horizon', '2', '--sat-instances', '2',
           '--half-instances', '4', '--wall-seconds', '30']
    if not runner.HAVE_SAT:
        cmd += ['--skip-sat']
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=40)
    return out


@pytest.fixture
def run_copy(complete_run, tmp_path):
    out = tmp_path / 'run'
    shutil.copytree(complete_run, out)
    return out


def test_complete_run_independently_replays_forward_attack(complete_run):
    report = audit(complete_run)
    assert report['passed'], report
    assert report['fiber_coverage'] == 'full'
    assert len(report['checks']) == 2
    results = read(complete_run/'results.json')
    for res in results:
        attack = res['panels']['forward_only_owner_then_honest']
        assert attack['owner_answer_correct_at_attack'] == 1.0
        assert attack['answer_agreement_after_attack_mean'] == 1.0
        assert attack['state_equality_at_steps_1_16_64_last'] == [1.0]*4


def test_empty_run_is_rejected(tmp_path):
    write(tmp_path/'results.json', [])
    write(tmp_path/'sha256.json', {'results.json':hashlib.sha256((tmp_path/'results.json').read_bytes()).hexdigest()})
    assert not audit(tmp_path)['passed']


@pytest.mark.parametrize('mutation', ['empty', 'missing', 'duplicate', 'unexpected', 'missing_panel',
    'false_panel', 'missing_sat', 'missing_fiber', 'false_fiber_mean', 'false_half', 'incomplete_receipt',
    'wrong_condition_receipt', 'old_schema'])
def test_semantic_corruption_is_rejected_even_with_matching_hashes(run_copy, mutation):
    results = read(run_copy/'results.json')
    receipt = read(run_copy/'receipt.json')
    config = read(run_copy/'config.json')
    if mutation == 'empty': results = []
    elif mutation == 'missing': results.pop()
    elif mutation == 'duplicate': results[1] = results[0]
    elif mutation == 'unexpected': results[1]['R'] = 3
    elif mutation == 'missing_panel': del results[0]['panels']['forward_only_owner_then_honest']
    elif mutation == 'false_panel': results[0]['panels']['forward_only_owner_then_honest']['owner_answer_correct_at_attack'] = 0.0
    elif mutation == 'missing_sat':
        config['sat_required'] = config['have_sat'] = True
        results[0].pop('sat', None)
    elif mutation == 'missing_fiber': results[0]['fibers']['contexts'].pop()
    elif mutation == 'false_fiber_mean': results[0]['fibers']['contexts'][0]['mean_witness_bits'] += .1
    elif mutation == 'false_half': results[0]['half_enumeration']['rows'][0]['found'] = False
    elif mutation == 'incomplete_receipt': receipt['completed'] = False
    elif mutation == 'wrong_condition_receipt': receipt['conditions'] += 1
    elif mutation == 'old_schema': config['schema_version'] = 1
    write(run_copy/'results.json',results); write(run_copy/'config.json',config)
    write(run_copy/'receipt.json',receipt); rehash(run_copy)
    assert not audit(run_copy)['passed'], mutation


@pytest.mark.parametrize('name', ['receipt.json', 'config.json', 'results.json', 'plan.md',
                                 'source/run_wide_trajectory.py','source/audit_wide_trajectory.py'])
def test_manifest_must_cover_required_evidence(run_copy, name):
    manifest = read(run_copy/'sha256.json'); del manifest[name]
    write(run_copy/'sha256.json',manifest)
    assert not audit(run_copy)['passed']


def test_missing_receipt_and_changed_source_are_rejected(run_copy):
    (run_copy/'receipt.json').unlink()
    (run_copy/'source/run_wide_trajectory.py').write_text('changed')
    result = audit(run_copy)
    assert not result['passed']
    assert set(result['hash_mismatches']) == {'receipt.json','source/run_wide_trajectory.py'}


@pytest.mark.skipif(not runner.HAVE_SAT, reason='SAT required')
@pytest.mark.parametrize('mutation', ['wrong_solution','budget_stop','missing_row','duplicate_solution'])
def test_sat_is_replayed_not_trusted(run_copy, mutation):
    results = read(run_copy/'results.json'); sat = results[0]['sat']; row = sat['rows'][0]
    if mutation == 'wrong_solution': row['solutions'][0] ^= 1
    elif mutation == 'budget_stop': row['termination'] = 'budget_exhausted'
    elif mutation == 'missing_row': sat['rows'].pop()
    elif mutation == 'duplicate_solution': row['solutions'].append(row['solutions'][0])
    # The old success flags intentionally remain true.
    write(run_copy/'results.json',results); rehash(run_copy)
    assert not audit(run_copy)['passed']


def test_audit_cli_fails_with_receipt_on_bad_evidence(tmp_path):
    output = tmp_path/'failed.json'
    proc = subprocess.run([sys.executable,str(ROOT/'scripts/audit_wide_trajectory.py'),str(tmp_path),
                           '--out',str(output)],capture_output=True,text=True)
    assert proc.returncode == 1
    assert read(output)['passed'] is False


def test_partial_coverage_is_explicit_and_zero_is_rejected(complete_run):
    report = audit(complete_run,1)
    assert report['passed'] and report['fiber_coverage'] == 'partial'
    assert all(c['contexts_replayed'] == 1 and c['contexts_available'] == 2 for c in report['checks'])
    assert not audit(complete_run,0)['passed']


def test_sat_budget_exhaustion_is_not_unsat(monkeypatch):
    class BudgetStop:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def conf_budget(self, budget): pass
        def solve_limited(self): return None
        def accum_stats(self): return {'conflicts':1}
    monkeypatch.setattr(runner,'Solver',BudgetStop,raising=False)
    p=runner.params(8)
    result=runner.sat_preimage(2,p,runner.constants(2,4),0,0,0,0,enumerate_all=True)
    assert result['termination'] == 'budget_exhausted'
    assert result['solutions'] == []
