import json
import resource
import sys
import time
from repair import reconstruct, advance, dot
from itertools import product


def predict(state, inputs, r, a, c):
    outputs=[]
    for u in inputs:
        state=advance(state,u,r,a);outputs.append(dot(c,state))
    return outputs


def run_stream(state, inputs, r, a, c, remove_warning):
    out=[];warnings=[]
    for k,u in enumerate(inputs):
        if k%8==0:
            warnings.append(0 if remove_warning else
                int(any(predict(state,[1-b for b in inputs[k:k+4]],r,a,c))))
        state=advance(state,u,r,a);out.append(dot(c,state))
    return out,warnings

start=time.process_time()
answers=[]
for job in json.load(sys.stdin):
    answer=reconstruct(job);r=answer['order'];a=answer['coefficients'];c=answer['readout']
    records=[]
    for z,inputs in zip(answer['warm_states'],job['inputs']):
        out,warnings=run_stream(z,inputs,r,a,c,False)
        patched_out,patched_warnings=run_stream(z,inputs,r,a,c,True)
        records.append(dict(outputs=out,warnings=warnings,parity=sum(out)%2,
            patched_outputs=patched_out,patched_warnings=patched_warnings,
            dead_outputs=[dot(c,z)]*len(inputs),restart_outputs=predict(0,inputs,r,a,c)))
    answer['streams']=records
    answer['exhaustive']=[predict(z,word,r,a,c)
        for z in answer['warm_states'][job['exhaustive_start']:] for word in product((0,1),repeat=8)] if job['exhaustive_start'] is not None else []
    answer['meter'].update(stream_steps=sum(len(u) for u in job['inputs']),
        warning_steps=sum(len(u)//8*4 for u in job['inputs']),
        exhaustive_steps=len(answer['exhaustive'])*8,
        restart_steps=sum(len(u) for u in job['inputs']))
    answer['meter'].update(head_patch_stream_steps=sum(len(u) for u in job['inputs']),head_patch_warning_steps=0)
    answers.append(answer)
json.dump(dict(answers=answers,cpu_seconds_before_output=time.process_time()-start,
               peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),sys.stdout)
