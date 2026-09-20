import json
import resource
import sys
import time
from learner import learn

cfg = json.loads(sys.stdin.readline())


def oracle(word):
    print(json.dumps(dict(query=word)), flush=True)
    return json.loads(sys.stdin.readline())['value']


start = time.process_time()
answer = learn(oracle, cfg)
print(json.dumps(dict(result=answer, cpu_seconds=time.process_time()-start,
                      peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)), flush=True)
