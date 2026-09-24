"""Record installed dependencies and derived runtime files before evidence transfer."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys

root=Path(sys.argv[1])
python=root/'initialization01/environment/bin/python'
for name, arguments in [('installed_requirements.txt',['freeze','--all']),('dependency_check.txt',['check'])]:
    result=subprocess.run([str(python),'-m','pip','--disable-pip-version-check']+arguments,
        capture_output=True,text=True,timeout=10)
    (root/name).write_text(result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(f'Dependency inspection failed: {name}')
records=[]
for directory in ['initialization01/runtime','initialization01/environment']:
    for path in sorted((root/directory).rglob('*')):
        relative=str(path.relative_to(root))
        if path.is_symlink():
            records.append(dict(path=relative,link=str(path.readlink())))
        elif path.is_file():
            with path.open('rb') as stream:
                checksum=hashlib.file_digest(stream,'sha256').hexdigest()
            records.append(dict(path=relative,bytes=path.stat().st_size,sha256=checksum))
receipt=dict(files=records,derived_file_count=len(records),
    derived_bytes=sum(row.get('bytes',0) for row in records),
    reconstruction='Verified standalone interpreter archive, hashed wheels, frozen stage configurations and scripts.',
    transfer_exclusions=['initialization01/runtime/','initialization01/environment/','*/package_cache/'],
    parent_files_deleted=False)
(root/'derived_environment_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({key:value for key,value in receipt.items() if key!='files'}))
