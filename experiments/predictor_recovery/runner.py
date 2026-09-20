import json
import resource
import sys
import time
from adapter import recover
from decode import decode_root

start=time.process_time();jobs=json.load(sys.stdin);answers=[]
for job in jobs:
    decoder_subtractions=0
    def predict(state):
        global decoder_subtractions
        row=job['retained'][state]
        if job['decode_root'] and state==0:
            decoder_subtractions+=len(row)
            return decode_root(row,job['denominator'])
        return row
    answer=recover(job['n'],job['hazards'],job['horizon'],job['denominator'],predict)
    answer['meter']['decoder_weight_subtractions']=decoder_subtractions
    answer['meter']['decoder_entry_writes']=decoder_subtractions
    answers.append(answer)
json.dump(dict(answers=answers,cpu_seconds_before_output=time.process_time()-start,
               peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),sys.stdout)
print()
