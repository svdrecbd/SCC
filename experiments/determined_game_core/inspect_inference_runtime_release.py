"""Resolve a compatible public CPython runtime without altering the host."""
from pathlib import Path
import json
import sys
from acquire_release_inventory import BoundedAcquisition


def main():
    directory = Path(sys.argv[1])
    configuration = json.loads((directory / 'config.json').read_text())
    acquisition = BoundedAcquisition(directory, configuration)
    release = json.loads(acquisition.acquire(configuration['release_url']))
    candidates = [asset for asset in release['assets']
        if asset['name'].startswith('cpython-3.11.')
        and asset['name'].endswith('x86_64-unknown-linux-gnu-install_only_stripped.tar.gz')]
    if len(candidates) != 1:
        raise ValueError('Runtime asset is missing or ambiguous.')
    selected = dict(release_id=release['id'], release_tag=release['tag_name'], asset=candidates[0])
    (directory / 'selected_runtime.json').write_text(json.dumps(selected, indent=2)+'\n')
    print(json.dumps(selected, indent=2))


if __name__ == '__main__':
    main()
