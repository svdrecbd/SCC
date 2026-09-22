"""Retain pinned upstream source as data, without importing or executing it."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request


def main(directory):
    configuration = json.loads((directory / 'source_config.json').read_text())
    total = 0
    records = []
    for relative in configuration['files']:
        url = configuration['base_url'] + '/' + relative
        with urllib.request.urlopen(url, timeout=20) as response:
            payload = response.read(configuration['maximum_total_bytes'] - total + 1)
        total += len(payload)
        if total > configuration['maximum_total_bytes']:
            raise RuntimeError('Source acquisition exceeded declared size bound')
        destination = directory / 'upstream' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as output:
            output.write(payload)
        records.append({'path': str(destination.relative_to(directory)), 'url': url,
                        'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})
    receipt = {'revision': configuration['revision'], 'bytes': total, 'files': records,
               'upstream_executed': False, 'weights_acquired': False, 'training': False}
    (directory / 'acquisition.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'files': len(records), 'bytes': total, 'upstream_executed': False}))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
