"""Submit SCC with persistent build contexts, without expiring runtime URLs."""
import argparse, hashlib, io, json, subprocess, tarfile
from pathlib import Path
from scc_persistent_context import native_files, job_request, upload_complete_context

ROOT = Path(__file__).resolve().parents[1]

def require_embedded_parents(specs):
    if specs:
        raise ValueError('Presigned runtime parent downloads are disabled. Package verified parent checkpoints in a complete persistent build context before submission.')

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parents', type=Path, required=True)
    p.add_argument('--runner', required=True)
    p.add_argument('--protocol', required=True)
    p.add_argument('--include', nargs='*', default=[])
    p.add_argument('--arguments', default='[]')
    p.add_argument('--output-name', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--key', required=True)
    p.add_argument('--minutes', type=int, default=30)
    a = p.parse_args()
    specs = json.loads(a.parents.read_text())
    require_embedded_parents(specs)
    arguments = json.loads(a.arguments)
    assert isinstance(arguments, list) and all(isinstance(x, str) for x in arguments)
    paths = sorted((ROOT / 'scc').glob('*.py'))
    paths += [ROOT / name for name in ('pyproject.toml', 'uv.lock', '.python-version', a.runner, a.protocol, *a.include)]
    entries = {str(path.resolve().relative_to(ROOT)): path.resolve() for path in paths}
    assert all(path.is_file() for path in entries.values())
    a.output.mkdir(parents=True, exist_ok=False)
    def save(name, value):
        (a.output / name).write_text(json.dumps(value, indent=2) + '\n')
    save('parents.json', specs)
    archive, manifest = a.output / 'source-overlay.tar', {}
    with tarfile.open(archive, 'w') as tar:
        for name, path in sorted(entries.items()):
            data = path.read_bytes()
            manifest[name] = hashlib.sha256(data).hexdigest()
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o644
            tar.addfile(info, io.BytesIO(data))
    save('overlay-manifest.json', manifest)
    files = native_files(archive, manifest)
    context = upload_complete_context(files, ROOT / 'artifacts/gman-persistent-contexts',
                                      ROOT / 'artifacts/developmental-gpu-20260910-v5/context')
    save('persistent-context.json', context)
    save('overlay-context.json', {'context_id': context['context_id'],
         'sha256': context['sha256'], 'bytes': context['bytes'],
         'source_overlay_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
         'source_overlay_bytes': archive.stat().st_size, 'transport': context['transport']})
    request = job_request(files, runner=a.runner, arguments=arguments,
                          output_name=a.output_name, key=a.key, minutes=a.minutes,
                          uploaded_context_id=context['context_id'])
    save('request.json', request)
    process = subprocess.run(['/Users/svdr/.local/bin/gman', 'api', 'post', '/jobs', '--data', json.dumps(request), '--json'],
                             capture_output=True, text=True)
    if process.returncode:
        (a.output / 'submission-error.txt').write_text(process.stderr)
        raise RuntimeError('Submission rejected; see saved submission-error.txt')
    receipt = json.loads(process.stdout)
    save('submit.json', receipt)
    print(json.dumps({key: receipt.get(key) for key in ('job_id', 'status', 'max_cost_usd')}))

if __name__ == '__main__':
    main()
