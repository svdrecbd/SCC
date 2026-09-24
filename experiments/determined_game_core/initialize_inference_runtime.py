"""Verify and privately unpack a pinned CPython runtime for inference."""
from pathlib import Path
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import time
from acquire_release_inventory import BoundedAcquisition


def main():
    directory = Path(sys.argv[1])
    configuration = json.loads((directory/'config.json').read_text())
    acquisition = BoundedAcquisition(directory, configuration)
    started = time.perf_counter()
    data = acquisition.acquire(configuration['runtime_url'])
    checksum = hashlib.sha256(data).hexdigest()
    if checksum != configuration['runtime_sha256']:
        raise ValueError('Runtime does not match the published digest.')
    target = directory/'runtime'
    target.mkdir(exist_ok=False)
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        extracted_bytes = sum(member.size for member in archive.getmembers() if member.isfile())
        if extracted_bytes > configuration['maximum_extracted_bytes']:
            raise ValueError('Runtime extraction exceeds allowance.')
        archive.extractall(target, filter='data')
    executable = target/'python/bin/python3'
    result = subprocess.run([str(executable), '--version'], capture_output=True, text=True, check=True, timeout=5)
    subprocess.run([str(executable), '-m', 'venv', str(directory/'environment')], check=True, timeout=25)
    receipt = dict(sha256=checksum, downloaded_bytes=acquisition.downloaded, extracted_bytes=extracted_bytes,
        python_version=result.stdout.strip(), environment=str(directory/'environment'), seconds=time.perf_counter()-started)
    (directory/'runtime_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
