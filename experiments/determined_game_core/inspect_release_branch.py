"""Retain selected files and trees from an immutable upstream revision."""
from pathlib import Path
import json
import sys
import time
import urllib.parse
import urllib.error
from acquire_release_inventory import BoundedAcquisition


def main():
    directory = Path(sys.argv[1])
    configuration = json.loads((directory / 'config.json').read_text())
    acquisition = BoundedAcquisition(directory, configuration)
    base = f"https://gitlab.com/api/v4/projects/{configuration['project']}/repository"
    revision = configuration['revision']
    started = time.perf_counter()
    results = {'revision': revision, 'trees': {}, 'files': {}, 'executed_upstream_source': False}
    for path in configuration.get('history_paths', []):
        query = urllib.parse.urlencode({'ref_name': revision, 'path': path, 'per_page': 100})
        history = json.loads(acquisition.acquire(base + '/commits?' + query))
        results.setdefault('histories', {})[path] = history
        if len(history) == 100:
            results.setdefault('possibly_truncated_histories', []).append(path)
    for path in configuration['source_directories']:
        query = urllib.parse.urlencode({'ref': revision, 'path': path, 'per_page': 100})
        results['trees'][path] = json.loads(acquisition.acquire(base + '/tree?' + query))
        if len(results['trees'][path]) == 100:
            results.setdefault('possibly_truncated_trees', []).append(path)
    for path in configuration['source_files']:
        try:
            data = acquisition.acquire(base + '/files/' + urllib.parse.quote(path, safe='') + '/raw?ref=' + revision)
        except urllib.error.HTTPError as error:
            if error.code != 404:
                raise
            results['files'][path] = 'not present'
            continue
        output = directory / 'selected_source' / path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
        results['files'][path] = len(data)
    results.update(seconds=time.perf_counter() - started, downloaded_bytes=acquisition.downloaded)
    (directory / 'inspection.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
