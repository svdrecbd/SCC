"""Bounded fixed-input comparison with exact known-count checks."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def main(directory, mode):
    configuration=json.loads((directory/'benchmark_config.json').read_text())
    results=[]
    for index,item in enumerate(configuration['cases']):
        path=Path(item['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        selected=item['checkpoint'] if mode=='learned' else mode
        case=directory/f'case_{index:02d}';case.mkdir()
        command=['/home/salvador/venvs/pytorch-pascal/bin/python',str(directory/'run_case.py'),str(case),selected,str(path),str(directory.parent)]
        (case/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        started=time.monotonic()
        with (case/'stdout.txt').open('w') as output,(case/'stderr.txt').open('w') as error:
            try:
                completed=subprocess.run(command,stdout=output,stderr=error,timeout=configuration['process_seconds'])
                status='finished';exit_code=completed.returncode
            except subprocess.TimeoutExpired:
                status='resource_limit';exit_code=None
        result={'case':index,'path':str(path),'mode':selected,'process_status':status,'exit_code':exit_code,
                'process_seconds':time.monotonic()-started,'expected':item['expected']}
        if status=='finished':
            assert exit_code==0,result
            result['computation']=json.loads((case/'result.json').read_text())
            if result['computation']['status']=='complete' and item['expected'] is not None:
                assert result['computation']['count']==item['expected'],result
        results.append(result)
        (directory/'results.json').write_text(json.dumps({'rows':results},indent=2)+'\n')
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    main(Path(sys.argv[1]).resolve(),sys.argv[2])
