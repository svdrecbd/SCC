import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from scc_persistent_context import native_files, job_request, upload_complete_context
from submit_scc_experiment import require_embedded_parents


def test_persistent_submission_has_embedded_code_and_no_expiring_runtime_reference(tmp_path):
    content = b'print("fixture")\n'
    archive = tmp_path / 'source.tar'
    with tarfile.open(archive, 'w') as tar:
        info = tarfile.TarInfo('scripts/run.py')
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    files = native_files(archive, {'scripts/run.py': hashlib.sha256(content).hexdigest()})
    request = job_request(files, runner='scripts/run.py', arguments=['--fixture', 'value with space'],
                          output_name='check', key='fixture', depends_on='job-fixture')
    assert request['context']['scripts/run.py']['content'].encode() == content
    assert request['context']['scc-empty-parents.json']['content'] == '{}\n'
    assert request['env'] == {}
    assert 'http' not in request['command'] and 'SCC_OVERLAY_URL' not in json.dumps(request)
    assert request['depends_on'] == {'job': 'job-fixture', 'require': 'result.ok == true'}
    uploaded = job_request(files, runner='scripts/run.py', arguments=[], output_name='check', key='fixture',
                           uploaded_context_id='ctx-fixture')
    assert uploaded['context_id'] == 'ctx-fixture' and 'context' not in uploaded
    assert len(json.dumps(uploaded)) < 1000 and uploaded['env'] == {}
    with pytest.raises(ValueError):
        native_files(archive, {'scripts/run.py': 'incorrect hash'})


def test_expiring_parent_attachment_path_is_rejected_before_submission():
    require_embedded_parents([])
    with pytest.raises(ValueError, match='Package verified parent checkpoints'):
        require_embedded_parents([{'artifact_id': 'unresolved'}])


def test_complete_build_context_is_zstd_and_preserves_source_and_data(tmp_path, monkeypatch):
    import scc_persistent_context as transport
    base = tmp_path / 'base'
    (base / 'data').mkdir(parents=True)
    (base / 'data/tokens.bin').write_bytes(b'\x01\x00\x02\x00')
    (base / 'Dockerfile').write_text('FROM fixture\nCOPY . /workspace\n')
    def api(method, path, data=None):
        if path == '/contexts':
            return {'context_id': 'ctx-test', 'upload': {'url': 'https://upload.invalid/fixture', 'headers': {}, 'method': 'PUT'}}
        assert path == '/contexts/ctx-test/finalize'
        return {}
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self): return b''
    def upload(request, timeout):
        # Independent decoder rejects the plain-tar regression.
        with tarfile.open(fileobj=io.BytesIO(request.data), mode='r:zst') as tar:
            assert tar.extractfile('data/tokens.bin').read() == b'\x01\x00\x02\x00'
            assert tar.extractfile('scripts/run.py').read() == b'pass\n'
        return Response()
    monkeypatch.setattr(transport, 'api', api)
    monkeypatch.setattr(transport.urllib.request, 'urlopen', upload)
    result = upload_complete_context({'scripts/run.py': {'content': 'pass\n'}}, tmp_path / 'cache', base)
    assert result['compression'] == 'zstd' and result['archive_roundtrip_verified']
