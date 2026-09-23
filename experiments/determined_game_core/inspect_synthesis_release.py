"""Inspect the archived SemML source without executing imported programs."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import sys
import tarfile
import time
import zipfile
from acquire_release_inventory import BoundedAcquisition, RemoteArchive


def main():
    directory = Path(sys.argv[1])
    configuration = json.loads((directory / 'config.json').read_text())
    acquisition = BoundedAcquisition(directory, configuration)
    started = time.perf_counter()
    with zipfile.ZipFile(RemoteArchive(acquisition)) as archive:
        member = archive.getinfo(configuration['source_member'])
        if member.file_size > configuration['maximum_download_bytes']:
            raise ValueError('Source payload exceeds allowance.')
        payload = archive.read(member)
    source_archive = directory / 'semml.tar.gz'
    source_archive.write_bytes(payload)
    payload_hash = hashlib.sha256(payload).hexdigest()
    del payload
    inventory = []
    selected_bytes = 0
    selected_files = 0
    suffixes = {'.java', '.py', '.json', '.yaml', '.yml', '.md', '.kts', '.properties', '.sh', '.xml'}
    with tarfile.open(source_archive, 'r:gz') as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe archive path.')
            record = {'path': member.name, 'bytes': member.size,
                      'regular_file': member.isfile(), 'selected': False}
            inventory.append(record)
            if not member.isfile() or member.size > 1048576:
                continue
            if path.suffix not in suffixes and path.name not in {'LICENSE', 'gradlew'}:
                continue
            if selected_bytes + member.size > configuration['maximum_selected_bytes']:
                raise ValueError('Selected source allowance exceeded.')
            body = archive.extractfile(member).read()
            output = directory / 'selected_source' / str(path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(body)
            selected_bytes += len(body)
            selected_files += 1
            record.update(selected=True, sha256=hashlib.sha256(body).hexdigest())
    (directory / 'source_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    result = {'archive_member': configuration['source_member'],
              'payload_sha256': payload_hash, 'zip_crc_verified': True,
              'full_archive_checksum_verified': False,
              'members': len(inventory), 'selected_files': selected_files,
              'selected_bytes': selected_bytes, 'downloaded_bytes': acquisition.downloaded,
              'seconds': time.perf_counter() - started,
              'executed_upstream_source': False, 'training': False}
    (directory / 'inspection.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
