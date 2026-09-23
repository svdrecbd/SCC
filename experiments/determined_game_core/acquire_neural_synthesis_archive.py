"""Acquire selected contiguous ZIP regions and CRC-checked release members."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import stat
import sys
import time
import zipfile
from acquire_release_inventory import BoundedAcquisition, RemoteArchive


class CachedArchive(RemoteArchive):
    def __init__(self, acquisition):
        super().__init__(acquisition)
        self.regions = []

    def retain_region(self, start, end):
        body = self.acquisition.acquire(self.acquisition.configuration['archive_url'],
            {'Range': f'bytes={start}-{end-1}', 'Accept-Encoding': 'identity'},
            f'bytes {start}-{end-1}/{self.size}')
        if len(body) != end-start:
            raise ValueError('Incomplete selected source region.')
        self.regions.append((start, body))

    def read(self, size=-1):
        size = min(self.size-self.position, size if size >= 0 else self.size)
        for start, data in self.regions:
            offset = self.position-start
            if offset >= 0 and offset+size <= len(data):
                self.position += size
                return data[offset:offset+size]
        return super().read(size)


def main():
    directory = Path(sys.argv[1])
    configuration = json.loads((directory / 'config.json').read_text())
    acquisition = BoundedAcquisition(directory, configuration)
    remote = CachedArchive(acquisition)
    started = time.perf_counter()
    records = []
    with zipfile.ZipFile(remote) as archive:
        all_members = sorted(archive.infolist(), key=lambda item: item.header_offset)
        ends = {item.filename: all_members[index+1].header_offset if index+1<len(all_members) else archive.start_dir
                for index,item in enumerate(all_members)}
        selected_names = set(configuration.get('members', []))
        for prefix in configuration.get('prefixes', []):
            group = [item for item in all_members if item.filename.startswith(prefix) and not item.is_dir()]
            if not group:
                raise ValueError('Missing archive prefix: ' + prefix)
            remote.retain_region(min(item.header_offset for item in group), max(ends[item.filename] for item in group))
            selected_names.update(item.filename for item in group)
        total = 0
        for name in sorted(selected_names):
            member = archive.getinfo(name)
            relative = PurePosixPath(name)
            if relative.is_absolute() or '..' in relative.parts or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError('Unsafe selected archive member.')
            total += member.file_size
            if total > configuration['maximum_extracted_bytes']:
                raise ValueError('Extraction allowance exceeded.')
            data = archive.read(member)  # zipfile verifies the complete member CRC.
            output = directory / 'selected_release' / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(data)
            records.append(dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), crc32_verified=member.CRC))
    result = dict(files=records, file_count=len(records), extracted_bytes=total,
        downloaded_bytes=acquisition.downloaded, seconds=time.perf_counter()-started,
        complete_archive_checksum_verified=False, upstream_source_executed=False, training=False)
    (directory / 'release_manifest.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='files'},indent=2))


if __name__ == '__main__':
    main()
