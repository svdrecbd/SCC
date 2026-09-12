"""Stream a verified GMAN result into a fresh directory without retaining a duplicate TAR."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request


def call(*args):
    result=subprocess.run(['/Users/svdr/.local/bin/gman',*args,'--json'],capture_output=True,text=True)
    if result.returncode:raise RuntimeError(result.stderr)
    return json.loads(result.stdout)


class Counted:
    def __init__(self,stream):self.stream=stream;self.hash=hashlib.sha256();self.count=0
    def read(self,size=-1):
        data=self.stream.read(size);self.hash.update(data);self.count+=len(data);return data


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('job');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    job=call('job','get',a.job);(a.output/'job.json').write_text(json.dumps(job,indent=2)+'\n')
    artifact=job.get('artifact')
    if not artifact or artifact['state']!='ready':raise RuntimeError('Result archive is not ready')
    metadata=call('api','get','/artifacts/'+artifact['artifact_id'])
    target=a.output/'files';target.mkdir()
    with urllib.request.urlopen(metadata['download']['url'],timeout=60) as response:
        counted=Counted(response)
        with tarfile.open(fileobj=counted,mode='r|*') as archive:
            for member in archive:
                if not (member.isfile() or member.isdir()):raise ValueError('Unexpected non-file archive member')
                archive.extract(member,target,filter='data')
        while counted.read(1024*1024):pass
    assert counted.hash.hexdigest()==artifact['sha256'] and counted.count==artifact['size_bytes']
    files={}
    for path in sorted(target.rglob('*')):
        if path.is_file():
            h=hashlib.sha256()
            with path.open('rb') as source:
                for block in iter(lambda:source.read(1024*1024),b''):h.update(block)
            files[str(path.relative_to(target))]={'bytes':path.stat().st_size,'sha256':h.hexdigest()}
    (a.output/'manifest.json').write_text(json.dumps({'artifact':artifact,'stream_sha256_verified':True,
        'archive_duplicate_retained':False,'files':files},indent=2)+'\n')
    print(json.dumps({'job':a.job,'status':job['status'],'files':len(files),'bytes':sum(v['bytes'] for v in files.values()),
                      'charged_usd':job.get('receipt',{}).get('charged_usd_micros',0)/1e6,'archive_hash_verified':True}))


if __name__=='__main__':main()
