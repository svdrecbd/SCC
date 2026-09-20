"""Memoized trajectory simulation; work depends on rollout budget, not state enumeration."""
import random
from decode import decode_root


def simulate(job,trials,seed):
    rng=random.Random(seed);cache={};hazards=set(job['hazards']);D=job['denominator']
    successes=[];lengths=[];draws=entries=decoder_ops=0
    for _ in range(trials):
        state=0;length=0
        for step in range(job['horizon']):
            if state in hazards:break
            if state not in cache:
                row=job['retained'][state]
                if job['decode_root'] and state==0:
                    row=decode_root(row,D);decoder_ops+=len(row)
                cache[state]=row;entries+=len(row)
            ticket=rng.randrange(D);draws+=1;length+=1
            for target,weight in cache[state]:
                if ticket<weight:state=target;break
                ticket-=weight
            else:raise AssertionError('invalid distribution')
        successes.append(int(state in hazards));lengths.append(length)
    return dict(hits=successes,lengths=lengths,predictor_calls=len(cache),returned_entries=entries,
                random_draws=draws,decoder_subtractions=decoder_ops,cached_states=sorted(cache))
