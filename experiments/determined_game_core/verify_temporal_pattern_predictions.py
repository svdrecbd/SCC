"""Verify every generated neural candidate with separately charged LTL checks."""
from pathlib import Path
import json
import subprocess
import sys
import time

root=Path(sys.argv[1]); configuration=json.loads((root/'config.json').read_text())
source=Path(configuration['inference_directory'])
if not (source/'completion.json').exists(): raise ValueError('Neural inference stage is not completed.')
records=json.loads((source/'inference.json').read_text())
(root/'original_inference.json').write_text(json.dumps(records,indent=2)+'\n')
java=Path(configuration['java_home'])/'bin/java'; classpath=str(root)+':'+configuration['library_directory']+'/*'
process=subprocess.run([str(java.with_name('javac')),'-cp',classpath,str(root/'VerifyCircuitSpecification.java')],capture_output=True,text=True,timeout=15)
(root/'compilation.txt').write_text(process.stdout+process.stderr);process.check_returncode()
results=[]
for record in records:
 for beam in record['beams']:
    row=dict(name=record['name'],beam=beam['index'],status=beam.get('status'),result='UNVERIFIED',accepted=False)
    if 'controller_graph' not in beam:
        row['reason']=beam.get('decoding_error') or beam.get('verification_error') or 'missing_graph'
    else:
        name=record['name']+'_beam_'+str(beam['index'])
        path=root/(name+'.json');path.write_text(json.dumps([beam['controller_graph']],indent=2)+'\n')
        command=[str(java),'-Xmx4g','-Xss16m','-XX:ActiveProcessorCount=1','-cp',classpath,'VerifyCircuitSpecification',str(path)]
        started=time.perf_counter()
        process=subprocess.run(['timeout',str(configuration['verification_seconds'])]+command,capture_output=True,text=True)
        (root/(name+'.stdout.txt')).write_text(process.stdout);(root/(name+'.stderr.txt')).write_text(process.stderr)
        row.update(seconds=time.perf_counter()-started,exit_code=process.returncode)
        if process.returncode==0:
            verification=json.loads(process.stdout)
            row.update(accepted=verification['accepted'],result='ACCEPTED' if verification['accepted'] else 'REJECTED',verification=verification)
    results.append(row)
    (root/'verification.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
