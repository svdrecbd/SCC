"""Status checks must not silently monitor jobs or mix historical account jobs."""
import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location('scc_status', Path(__file__).resolve().parents[1]/'scripts/scc_status.py')
status = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(status)


def test_saved_default_never_contacts_provider(tmp_path, monkeypatch, capsys):
    path = tmp_path/'state.json'
    path.write_text(json.dumps({'new_runs': [{'job_id': 'job-a', 'label': 'candidate', 'last_observed_status': 'running'}]}))
    before = path.read_bytes()
    monkeypatch.setattr(status, 'refresh_unfinished', lambda *a: (_ for _ in ()).throw(AssertionError('Unexpected live check')))
    assert status.main(['--state', str(path)]) == 0
    assert 'Saved observations only' in capsys.readouterr().out
    assert path.read_bytes() == before


def test_live_checks_only_registered_unfinished_once():
    rows = [{'job_id': f'job-{i}', 'label': str(i), 'status': s, 'queried_now': False}
            for i, s in enumerate(['succeeded', 'failed', 'canceled', 'cancelled', 'running', 'queued'])]
    calls = []
    def fetch(cli, job):
        calls.append(job)
        return {'job_id': job, 'status': 'succeeded', 'artifact': {'download': {'url': 'SECRET'}}}
    result = status.refresh_unfinished(rows, 'fixture-cli', fetch)
    assert sorted(calls) == ['job-4', 'job-5']
    assert len(calls) == 2
    assert 'SECRET' not in json.dumps(result)
    assert [r['status'] for r in rows][-2:] == ['running', 'queued']


def test_failure_is_unknown_and_does_not_expose_provider_error():
    def failed(*args):
        raise RuntimeError('SECRET_URL')
    result = status.refresh_unfinished([{'job_id': 'job-a', 'status': 'running'}], 'fixture-cli', failed)
    assert result[0]['status'] == 'query_error'
    assert result[0]['previous_status'] == 'running'
    assert 'SECRET_URL' not in json.dumps(result)


def test_duplicate_job_ids_are_rejected():
    import pytest
    row = {'job_id': 'job-a', 'label': 'candidate'}
    with pytest.raises(ValueError, match='Duplicate'):
        status.registered_rows({'new_runs': [row], 'learnability_calibration_runs': [row]})
