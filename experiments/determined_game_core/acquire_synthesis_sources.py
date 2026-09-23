"""Acquire selected build, model and release documents from pinned sources."""
from pathlib import Path
import json
import sys
import time
import urllib.parse
import zipfile
from acquire_release_inventory import BoundedAcquisition, RemoteArchive


def main():
    directory = Path(sys.argv[1])
    configuration = json.loads((directory/'config.json').read_text())
    acquisition = BoundedAcquisition(directory, configuration)
    started = time.perf_counter()
    base = f"https://gitlab.com/api/v4/projects/{configuration['project']}/repository"
    source_root = directory/'source_documents'
    source_root.mkdir()
    for name in configuration['source_files']:
        body = acquisition.acquire(base+'/files/'+urllib.parse.quote(name, safe='')+'/raw?ref='+configuration['revision'])
        (source_root/name).write_bytes(body)
    trees = {}
    for name in configuration['source_directories']:
        trees[name] = json.loads(acquisition.acquire(base+'/tree?'+urllib.parse.urlencode(
            {'ref': configuration['revision'], 'path': name, 'per_page': 100})))
    (directory/'source_trees.json').write_text(json.dumps(trees, indent=2)+'\n')
    archive_root = directory/'archive_documents'
    archive_root.mkdir()
    with zipfile.ZipFile(RemoteArchive(acquisition)) as archive:
        for name in configuration['archive_members']:
            member = archive.getinfo(name)
            if member.file_size > 1048576:
                raise ValueError('Selected document exceeds the extraction allowance.')
            body = archive.read(member)
            output = archive_root/name
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(body)
    result = {'revision': configuration['revision'], 'source_files': configuration['source_files'],
              'source_directories': trees, 'archive_documents': configuration['archive_members'],
              'downloaded_bytes': acquisition.downloaded, 'seconds': time.perf_counter()-started,
              'executed_upstream_source': False, 'full_archive_checksum_verified': False}
    (directory/'inspection.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
