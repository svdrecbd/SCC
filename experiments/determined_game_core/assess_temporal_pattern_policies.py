"""Apply existing structural public procedures to every frozen generated case."""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(sys.argv[1]); configuration=json.loads((root/'config.json').read_text())
source=Path(configuration['generation_directory'])
records=json.loads((source/'specifications.json').read_text())
(root/'specifications.json').write_text(json.dumps(records,indent=2)+'\n')
java=Path(configuration['java_home'])/'bin/java'; classpath=str(root)+':'+configuration['library_directory']+'/*'
result=subprocess.run([str(java.with_name('javac')),'-cp',classpath,str(root/'PropositionalPolicySearch.java'),str(root/'TemporalContradictionSearch.java')],capture_output=True,text=True,timeout=15)
(root/'compilation.txt').write_text(result.stdout+result.stderr);result.check_returncode()
command=[str(java),'-Xmx4g','-Xss16m','-XX:ActiveProcessorCount=1','-cp',classpath]
with (root/'temporal.jsonl').open('w') as output,(root/'temporal.stderr.txt').open('w') as error:
 result=subprocess.run(['timeout','10']+command+['TemporalContradictionSearch',str(root/'specifications.json'),'5','12'],stdout=output,stderr=error)
result.check_returncode()
results=[]
for record in records:
 path=source/record['name']/'formula.txt'; started=time.perf_counter()
 process=subprocess.run(['timeout','4']+command+['PropositionalPolicySearch',str(path),','.join(record['inputs']),','.join(record['outputs']),'1'],capture_output=True,text=True)
 (root/(record['name']+'.stdout.txt')).write_text(process.stdout)
 (root/(record['name']+'.stderr.txt')).write_text(process.stderr)
 results.append(dict(name=record['name'],seconds=time.perf_counter()-started,exit_code=process.returncode,result=next((line.split('\t')[1] for line in process.stdout.splitlines() if line.startswith('RESULT\t')),None)))
(root/'propositional.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(dict(propositional=results,temporal=[json.loads(line) for line in (root/'temporal.jsonl').read_text().splitlines()])))
