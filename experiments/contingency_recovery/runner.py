import json
import resource
import sys
import time
from engine import plan

started=time.process_time()
data=json.load(sys.stdin)
worlds={(w['n'],w['seed']):w for w in data['worlds']}
results=[]
for task in data['tasks']:
    world=worlds[(task['n'],task['seed'])]
    results.append(plan(world['edges'],task['terminal'],task['horizon'],task['objective']))
json.dump(dict(results=results,cpu_seconds=time.process_time()-started,
               peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),sys.stdout)
print()
