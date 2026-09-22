"""Independently check returned counts with one justified reference configuration change."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time


def main(directory):
    configuration=json.loads((directory/'reference_config.json').read_text())
    native=json.loads((directory.parent/'native01/results.json').read_text())['rows']
    results=[]
    for index,item in enumerate(configuration['cases']):
        path=Path(item['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        expected=native[index]['computation']['count']; assert str(path)==native[index]['path']
        command=[configuration['binary'],*configuration['arguments'],str(path)]
        output_path=directory/f'case_{index:02d}.stdout';error_path=directory/f'case_{index:02d}.stderr'
        started=time.monotonic()
        with output_path.open('w') as output,error_path.open('w') as error:
            try:
                completed=subprocess.run(command,stdout=output,stderr=error,timeout=configuration['process_seconds'])
                status='finished';exit_code=completed.returncode
            except subprocess.TimeoutExpired:
                status='resource_limit';exit_code=None
        text=output_path.read_text()
        values=re.findall(r'^c s exact arb int ([0-9]+)\s*$',text,re.MULTILINE)
        count=int(values[-1]) if values and status=='finished' else None
        result={'path':str(path),'command':command,'status':status,'exit_code':exit_code,'count':count,
                'expected':expected,'seconds':time.monotonic()-started}
        results.append(result)
        (directory/'results.json').write_text(json.dumps({'rows':results},indent=2)+'\n')
        print(json.dumps(result),flush=True)
        if status=='finished':
            assert exit_code==0 and count==expected,result
            assert 'epsilon: 0 delta: 0' in text


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve())
