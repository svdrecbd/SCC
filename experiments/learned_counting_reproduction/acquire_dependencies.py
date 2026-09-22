"""Extract hash-verified distribution packages into a private build directory."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main(directory):
    configuration=json.loads((directory/'dependency_config.json').read_text())
    packages=directory/'packages'; packages.mkdir()
    records=[]; total=0
    for item in configuration['packages']:
        result=subprocess.run(['apt-get','download',item['name']+'='+item['version']],cwd=packages,
                              text=True,capture_output=True,timeout=20,check=True)
        path=next(packages.glob(item['name']+'_*.deb')); payload=path.read_bytes(); total+=len(payload)
        assert total<=configuration['maximum_bytes'] and hashlib.sha256(payload).hexdigest()==item['sha256']
        subprocess.run(['dpkg-deb','--extract',str(path),str(directory/'dependencies')],check=True)
        records.append({**item,'bytes':len(payload),'stdout':result.stdout,'stderr':result.stderr})
    (directory/'dependencies.json').write_text(json.dumps(records,indent=2)+'\n')
    print(json.dumps({'packages':len(records),'bytes':total}),flush=True)


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
