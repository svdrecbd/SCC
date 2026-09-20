import json
import resource
import sys
import time
from quotient import construct, recover_hazard
from head_patch import answer
from itertools import product

start = time.process_time()
models = json.load(sys.stdin)
answers = []
for m in models:
    row = dict(id=m['id'], useful=construct(m, False), joint=construct(m, True))
    if m['family'] == 'structured':
        begin = time.perf_counter()
        q = row['useful']
        row['repairs'] = [recover_hazard(q['quotient'], c) for c in q['classes']]
        row['repair_seconds'] = time.perf_counter() - begin
        j = row['joint']
        row['head_patch'] = [answer(j['quotient'], c, word, True)
                            for h in range(5) for word in product(range(2), repeat=h)
                            for c in j['classes']]
    answers.append(row)
json.dump(dict(answers=answers, cpu_seconds_before_output=time.process_time()-start,
               peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), sys.stdout)
