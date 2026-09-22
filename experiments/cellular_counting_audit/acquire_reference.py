"""Acquire and verify one pinned official model-counter release."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import urllib.request


def main(directory):
    configuration = json.loads((directory/'reference_config.json').read_text())
    with urllib.request.urlopen(configuration['release_api'], timeout=20) as response:
        metadata = response.read()
    (directory/'release.json').write_bytes(metadata)
    release = json.loads(metadata)
    assert release['tag_name'] == configuration['release_tag']
    asset = next(item for item in release['assets'] if item['name']==configuration['asset_name'])
    assert asset['digest'] == 'sha256:'+configuration['sha256']
    with urllib.request.urlopen(asset['browser_download_url'], timeout=40) as response:
        payload = response.read(configuration['maximum_archive_bytes']+1)
    assert len(payload) <= configuration['maximum_archive_bytes'] and len(payload) == asset['size']
    assert hashlib.sha256(payload).hexdigest() == configuration['sha256']
    archive = directory/configuration['asset_name']; archive.write_bytes(payload)
    with tarfile.open(archive) as contents:
        members = contents.getmembers()
        assert sum(member.size for member in members) <= configuration['maximum_extracted_bytes']
        contents.extractall(directory/'distribution', filter='data')
    receipt = {'archive_sha256':configuration['sha256'], 'bytes':len(payload),
               'members':[{'name':member.name,'size':member.size} for member in members]}
    (directory/'acquisition.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
