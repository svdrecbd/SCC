"""Run bounded nonlearned realizability decisions on a frozen fresh workload."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time

root=Path(sys.argv[1]); configuration=json.loads((root/'config.json').read_text())
records=json.loads((root/'specifications.json').read_text())
environment=dict(os.environ,JAVA_HOME=configuration['java_home'],JAVA_OPTS='-Xms128m -Xmx4g -Xss16m -XX:ActiveProcessorCount=1',PATH=configuration['dependency_directory']+os.pathsep+os.environ['PATH'])
results=[]
for record in records:
    directory=root/record['name'];directory.mkdir()
    conjunction=lambda values:' & '.join('('+value+')' for value in values) or 'true'
    formula='('+conjunction(record['assumptions'])+') -> ('+conjunction(record['guarantees'])+')'
    (directory/'formula.txt').write_text(formula+'\n')
    command=[configuration['executable'],'semmlMain','--env',','.join(record['inputs']),'--sys',','.join(record['outputs']),
        '--formulaFile',str(directory/'formula.txt'),'--realizable','true','--outputFormat','HOA','--phEnv','STRIX_SCORE','--phSys','STRIX_SCORE',
        '--explorationPerspective','BOTH','--bthEnv','STRIX_SCORE','--bthSys','STRIX_SCORE','--trace','false']
    started=time.perf_counter()
    with (directory/'stdout.txt').open('w') as output,(directory/'stderr.txt').open('w') as error:
        result=subprocess.run(['timeout',str(configuration['invocation_seconds'])]+command,stdout=output,stderr=error,env=environment)
    winners=[line for line in (directory/'stdout.txt').read_text().splitlines() if line in ('REALIZABLE','UNREALIZABLE')]
    row=dict(name=record['name'],seconds=time.perf_counter()-started,exit_code=result.returncode,winner=winners[0] if result.returncode==0 and len(winners)==1 else None,command=command)
    results.append(row);(root/'decisions.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
