"""Acquire and safely unpack a bounded immutable repository archive."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import sys
import tarfile
import time
import urllib.parse
from acquire_release_inventory import BoundedAcquisition


def main():
    directory = Path(sys.argv[1])
    configuration = json.loads((directory / 'config.json').read_text())
    acquisition = BoundedAcquisition(directory, configuration)
    query = urllib.parse.urlencode({'sha': configuration['revision'],
                                   'exclude_paths': ','.join(configuration['exclude_paths'])})
    url = f"https://gitlab.com/api/v4/projects/{configuration['project']}/repository/archive.tar.gz?{query}"
    started = time.perf_counter()
    data = acquisition.acquire(url)
    archive_path = directory / 'response_001.bin'
    archive_hash = hashlib.sha256(data).hexdigest()
    del data
    inventory = []
    extracted_bytes = 0
    with tarfile.open(archive_path, 'r:gz') as archive:
        for member in archive:
            original = PurePosixPath(member.name)
            if original.is_absolute() or '..' in original.parts:
                raise ValueError('Unsafe archive member path.')
            relative = PurePosixPath(*original.parts[1:])
            if not member.isfile():
                if not member.isdir():
                    raise ValueError('Only regular files and directories are permitted.')
                continue
            if extracted_bytes + member.size > configuration['maximum_extracted_bytes']:
                raise ValueError('Extraction allowance exceeded.')
            body = archive.extractfile(member).read()
            output = directory / 'source' / str(relative)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(body)
            if member.mode & 0o111:
                output.chmod(0o755)
            inventory.append({'path': str(relative), 'bytes': len(body),
                              'sha256': hashlib.sha256(body).hexdigest()})
            extracted_bytes += len(body)
    (directory / 'source_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    result = {'revision': configuration['revision'], 'archive_sha256': archive_hash,
              'downloaded_bytes': acquisition.downloaded, 'extracted_bytes': extracted_bytes,
              'files': len(inventory), 'seconds': time.perf_counter() - started,
              'executed_upstream_source': False}
    (directory / 'inspection.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
