"""Inspect immutable neural synthesis sources without model installation."""
from pathlib import Path
import json
import sys
import time
import urllib.error
import zipfile
from acquire_release_inventory import BoundedAcquisition, RemoteArchive


def main():
    directory = Path(sys.argv[1])
    configuration = json.loads((directory / 'config.json').read_text())
    acquisition = BoundedAcquisition(directory, configuration)
    records = {}
    started = time.perf_counter()
    for repository, files in configuration['repositories'].items():
        base = 'https://api.github.com/repos/' + repository
        reference = configuration.get('revisions', {}).get(repository, 'HEAD')
        commit = json.loads(acquisition.acquire(base + '/commits/' + reference))
        revision = commit['sha']
        tree = json.loads(acquisition.acquire(base + '/git/trees/' + commit['commit']['tree']['sha'] + '?recursive=1'))
        if tree.get('truncated'):
            raise ValueError('Source tree was truncated.')
        record = dict(revision=revision, tree=tree['tree'], files={})
        records[repository] = record
        (directory / 'inspection.json').write_text(json.dumps(records, indent=2) + '\n')
        for name in files:
            data = acquisition.acquire('https://raw.githubusercontent.com/' + repository + '/' + revision + '/' + name)
            output = directory / 'selected_source' / repository.split('/')[-1] / name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
            record['files'][name] = len(data)
        (directory / 'inspection.json').write_text(json.dumps(records, indent=2) + '\n')
    for name, url in configuration.get('resources', {}).items():
        try:
            data = acquisition.acquire(url)
        except urllib.error.HTTPError:
            if configuration.get('continue_on_http_error', False):
                continue
            raise
        output = directory / 'selected_resources' / name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(data)
    if configuration.get('archive_url'):
        with zipfile.ZipFile(RemoteArchive(acquisition)) as archive:
            members = [dict(path=item.filename, bytes=item.file_size, compressed_bytes=item.compress_size,
                            crc32=item.CRC, header_offset=item.header_offset) for item in archive.infolist()]
        (directory / 'archive_inventory.json').write_text(json.dumps(members, indent=2) + '\n')
    print(json.dumps(dict(seconds=time.perf_counter()-started, downloaded_bytes=acquisition.downloaded,
        repositories={name:dict(revision=record['revision'], tree_entries=len(record['tree']), files=record['files'])
                      for name,record in records.items()}), indent=2))


if __name__ == '__main__':
    main()
