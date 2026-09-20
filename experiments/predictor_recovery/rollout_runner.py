import json
import resource
import sys
import time
from rollout import simulate

start=time.process_time();payload=json.load(sys.stdin);results=[]
for task in payload['tasks']:
    results.append(simulate(payload['jobs'][task['job']],task['trials'],task['seed']))
json.dump(dict(results=results,cpu_seconds_before_output=time.process_time()-start,
               peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),sys.stdout)
print()
