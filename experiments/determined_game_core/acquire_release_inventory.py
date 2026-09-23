"""Inspect pinned source and ZIP metadata without fetching the large artifact."""
from pathlib import Path
import hashlib
import io
import json
import sys
import time
import urllib.parse
import urllib.request
import zipfile


class BoundedAcquisition:
    def __init__(self, directory, configuration):
        self.directory = directory
        self.configuration = configuration
        self.records = []
        self.downloaded = 0

    def acquire(self, url, headers=None, expected_range=None):
        if len(self.records) >= self.configuration['maximum_requests']:
            raise ValueError('Request allowance exhausted.')
        remaining = self.configuration['maximum_download_bytes']-self.downloaded
        if remaining <= 0:
            raise ValueError('Download allowance exhausted.')
        name = f'response_{len(self.records)+1:03d}.bin'
        record = {'url': url, 'request_headers': headers or {}, 'file': name}
        self.records.append(record)
        self.save()
        try:
            request = urllib.request.Request(url, headers=headers or {})
            with urllib.request.urlopen(request, timeout=self.configuration['request_timeout_seconds']) as response:
                record['status'] = response.status
                record['content_range'] = response.headers.get('Content-Range')
                record['content_length'] = response.headers.get('Content-Length')
                if expected_range is not None and (response.status != 206 or record['content_range'] != expected_range):
                    raise ValueError('Server did not honor the exact requested byte range.')
                body = response.read(remaining+1)
            self.downloaded += len(body)
            record['bytes'] = len(body)
            record['sha256'] = hashlib.sha256(body).hexdigest()
            (self.directory/name).write_bytes(body)
            if self.downloaded > self.configuration['maximum_download_bytes']:
                raise ValueError('Download allowance exceeded; aborting acquisition.')
            return body
        except Exception as error:
            record['error'] = str(error)
            raise
        finally:
            self.save()

    def save(self):
        (self.directory/'acquisition.json').write_text(json.dumps(
            {'downloaded_bytes': self.downloaded, 'requests': self.records}, indent=2)+'\n')


class RemoteArchive(io.RawIOBase):
    def __init__(self, acquisition):
        self.acquisition = acquisition
        self.position = 0
        self.size = acquisition.configuration['archive_bytes']

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = (0 if whence == 0 else self.position if whence == 1 else self.size)+offset
        if self.position < 0:
            raise ValueError('Invalid archive position.')
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = min(self.size-self.position, size if size >= 0 else self.size)
        if size <= 0:
            return b''
        if size > self.acquisition.configuration['maximum_download_bytes']-self.acquisition.downloaded:
            raise ValueError('Requested archive segment exceeds the remaining allowance.')
        end = self.position+size-1
        body = self.acquisition.acquire(self.acquisition.configuration['archive_url'],
            {'Range': f'bytes={self.position}-{end}', 'Accept-Encoding': 'identity'},
            f'bytes {self.position}-{end}/{self.size}')
        if len(body) != size:
            raise ValueError('Incomplete archive segment.')
        self.position += len(body)
        return body


def main():
    directory = Path(sys.argv[1])
    configuration = json.loads((directory/'config.json').read_text())
    acquisition = BoundedAcquisition(directory, configuration)
    started = time.perf_counter()
    results = {'revision': configuration['revision'], 'executed_upstream_source': False,
               'full_archive_checksum_verified': False}
    base = f"https://gitlab.com/api/v4/projects/{configuration['project']}/repository"
    tree = json.loads(acquisition.acquire(base+'/tree?'+urllib.parse.urlencode(
        {'ref': configuration['revision'], 'per_page': 100})))
    results['source_root'] = tree
    for name in ('README.md', 'Build.md', 'Usage.md'):
        if any(item['path'] == name and item['type'] == 'blob' for item in tree):
            data = acquisition.acquire(base+'/files/'+urllib.parse.quote(name, safe='')+'/raw?ref='+configuration['revision'])
            (directory/name).write_bytes(data)
    (directory/'inventory.json').write_text(json.dumps(results, indent=2)+'\n')
    try:
        with zipfile.ZipFile(RemoteArchive(acquisition)) as archive:
            results['archive_members'] = [{'path': item.filename, 'bytes': item.file_size,
                'compressed_bytes': item.compress_size, 'crc32': item.CRC,
                'header_offset': item.header_offset} for item in archive.infolist()]
    except Exception as error:
        results['archive_inventory_error'] = str(error)
    results['seconds'] = time.perf_counter()-started
    (directory/'inventory.json').write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps({'source_paths': [item['path'] for item in tree],
        'archive_members': len(results.get('archive_members', [])),
        'archive_inventory_error': results.get('archive_inventory_error'),
        'downloaded_bytes': acquisition.downloaded, 'seconds': results['seconds']}, indent=2))


if __name__ == '__main__':
    main()
