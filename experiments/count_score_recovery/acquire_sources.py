"""Preserve selected direct-estimation sources and checkpoint availability records."""
import hashlib
import json
from pathlib import Path
import sys
import time
import urllib.request


def main(directory):
    configuration = json.loads((directory / 'acquisition_config.json').read_text())
    total_bytes = 0
    receipt = []
    started = time.monotonic()

    def obtain(address, destination, blob=None):
        nonlocal total_bytes
        request = urllib.request.Request(address, headers={'User-Agent': 'SCC-research-source-preservation'})
        with urllib.request.urlopen(request, timeout=15) as response:
            data = response.read(configuration['maximum_bytes'] - total_bytes + 1)
        total_bytes += len(data)
        assert total_bytes <= configuration['maximum_bytes']
        assert time.monotonic() - started < configuration['wall_seconds']
        if blob is not None:
            prefix = f'blob {len(data)}\0'.encode()
            assert hashlib.sha1(prefix + data).hexdigest() == blob
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        receipt.append({'url': address, 'path': str(destination.relative_to(directory)),
                        'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                        'git_blob_sha1': blob})
        return data

    inventories = []
    for specification in configuration['repositories']:
        repository = specification['repository']
        revision = specification['revision']
        location = directory / 'upstream' / repository
        address = f'https://api.github.com/repos/{repository}'
        tree = json.loads(obtain(address + f'/git/trees/{revision}?recursive=1', location / 'tree.json'))
        assert tree['sha'] == revision and not tree['truncated']
        records = {record['path']: record for record in tree['tree']}
        candidates = [path for path in records if path.endswith(('.pt', '.pth', '.ckpt', '.safetensors'))]
        inventories.append({'repository': repository, 'revision': revision,
                            'entries': len(records), 'checkpoint_candidates': candidates})
        for path in specification['files']:
            record = records[path]
            assert record['type'] == 'blob'
            obtain(f'https://raw.githubusercontent.com/{repository}/{revision}/{path}',
                   location / path, record['sha'])
        obtain(address + '/releases', location / 'releases.json')
        if specification['issue'] is not None:
            issue_path = f'/issues/{specification["issue"]}'
            obtain(address + issue_path, location / 'checkpoint_issue.json')
            obtain(address + issue_path + '/comments', location / 'checkpoint_comments.json')
    result = {'files': receipt, 'inventories': inventories, 'bytes': total_bytes,
              'seconds': time.monotonic() - started, 'upstream_executed': False, 'training': False}
    (directory / 'acquisition.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'files': len(receipt), 'bytes': total_bytes, 'inventories': inventories}), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
