"""Persistent build contexts: never vend runtime source download URLs."""
import hashlib
import json
from pathlib import PurePosixPath
import re
import shlex
import tarfile
import io
from pathlib import Path
import subprocess
import urllib.request

BASE_CONTEXT = 'ctx-983c199b'


def native_files(archive, expected):
    files = {}
    with tarfile.open(archive) as tar:
        for member in tar:
            name = member.name
            path = PurePosixPath(name)
            if not member.isfile() or path.is_absolute() or '..' in path.parts or name in files:
                raise ValueError('Invalid frozen source member')
            data = tar.extractfile(member).read()
            if hashlib.sha256(data).hexdigest() != expected.get(name):
                raise ValueError('Frozen source digest mismatch: ' + name)
            files[name] = {'content': data.decode('utf-8')}
    if set(files) != set(expected):
        raise ValueError('Frozen source manifest and archive differ')
    files['scc-empty-parents.json'] = {'content': '{}\n'}
    return files


def job_request(files, *, runner, arguments, output_name, key, minutes=120, depends_on=None, uploaded_context_id=None):
    """Set a per-attempt runtime budget; queue lifetime is a separate provider limit.

    A longer budget also needs compatible runner deadlines and a declared protocol.
    """
    if not re.fullmatch('[a-z0-9-]+', output_name) or not 1 <= minutes <= 720:
        raise ValueError('Invalid output name or duration')
    if runner not in files or not all(isinstance(a, str) for a in arguments):
        raise ValueError('Missing runner or invalid arguments')
    if not uploaded_context_id and len(json.dumps(files).encode()) > 4 * 1024**2:
        raise ValueError('Native inline overlay exceeds 4 MiB; use a complete uploaded build context')
    command = shlex.join(['python', '/workspace/' + runner, '--output', '/output/' + output_name,
                         '--data', '/workspace/data', '--parents', '/workspace/scc-empty-parents.json',
                         '--device', 'cuda', *arguments])
    result = {'chip': 'h100', 'chip_count': 1, 'context_id': uploaded_context_id or BASE_CONTEXT,
              'command': command, 'env': {},
              'max_duration_minutes': minutes, 'build_timeout_minutes': 30,
              'mission': 'scc-developmental-coupling', 'idempotency_key': key, 'label': key,
              'resume': 'none', 'max_restarts': 0,
              'checks': {'success': {'check': {'type': 'file_exists', 'path': '/output/' + output_name + '/result.json'}}}}
    if not uploaded_context_id:
        result['context'] = files
    if depends_on:
        result['depends_on'] = {'job': depends_on, 'require': 'result.ok == true'}
    return result


def api(method, path, data=None):
    command = ['/Users/svdr/.local/bin/gman', 'api', method, path, '--json']
    if data is not None:
        command += ['--data', json.dumps(data)]
    process = subprocess.run(command, capture_output=True, text=True)
    if process.returncode:
        raise RuntimeError(process.stderr)
    return json.loads(process.stdout)


def upload_complete_context(files, cache_root, base_root):
    """Upload source, data and Dockerfile before queueing; cache identical builds.

    The upload URL is used immediately by this local process. No URL or account
    credential enters the job. Queued work uses the platform-owned context ID.
    """
    cache_root, base_root = Path(cache_root), Path(base_root)
    metadata = {n: hashlib.sha256(v['content'].encode()).hexdigest() for n, v in files.items()}
    extras = {'Dockerfile': base_root / 'Dockerfile'}
    extras.update({str(p.relative_to(base_root)): p for p in sorted((base_root / 'data').rglob('*')) if p.is_file()})
    if set(extras) & set(files):
        raise ValueError('Source overlay must not replace dataset or Dockerfile')
    metadata.update({n: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest() for n, p in extras.items()})
    key = hashlib.sha256(json.dumps({'format': 'tar.zst/v1', 'files': metadata}, sort_keys=True).encode()).hexdigest()
    folder = cache_root / key
    receipt = folder / 'context.json'
    if receipt.exists():
        return json.loads(receipt.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / 'build-context.tar.zst'
    with tarfile.open(archive, 'w:zst') as tar:
        for name in sorted(metadata):
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Invalid build member')
            info = tarfile.TarInfo(name)
            info.mode = 0o644
            if name in files:
                data = files[name]['content'].encode()
                assert hashlib.sha256(data).hexdigest() == metadata[name]
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            else:
                info.size = extras[name].stat().st_size
                with extras[name].open('rb') as stream:
                    tar.addfile(info, stream)
    (folder / 'files.json').write_text(json.dumps(metadata, indent=2) + '\n')
    # Verify the actual compressed archive, not only the intended inputs.
    with tarfile.open(archive, 'r:zst') as tar:
        observed = {member.name: hashlib.file_digest(tar.extractfile(member), 'sha256').hexdigest() for member in tar}
    assert observed == metadata
    sha = hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()
    context = api('post', '/contexts', {'sha256': sha, 'size_bytes': archive.stat().st_size})
    transfer = context['upload']
    with urllib.request.urlopen(urllib.request.Request(transfer['url'], data=archive.read_bytes(),
                                headers=transfer['headers'], method=transfer['method']), timeout=60) as response:
        response.read()
    api('post', '/contexts/' + context['context_id'] + '/finalize')
    result = {'context_id': context['context_id'], 'sha256': sha, 'bytes': archive.stat().st_size,
              'files_manifest': str(folder / 'files.json'), 'local_archive': str(archive),
              'transport': 'complete-persistent-build-context/v2', 'compression': 'zstd',
              'archive_roundtrip_verified': True, 'runtime_source_urls': False}
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    return result
