import json
import random
import sys
import time
from fractions import Fraction
from pathlib import Path
import numpy as np
from decoder import choose_k, decode, select, subset_masks
from fixtures import make_spec, oracle_for


def run_case(spec, cfg, output, name, forced_k=None):
    epsilon = Fraction(1, 2**(spec['parameter']+1)) if spec['kind']=='quadratic' else Fraction(1,32)
    k = forced_k or choose_k(spec['n'], epsilon, cfg['failure_target'])
    start = time.monotonic()
    decoded = decode(spec['n'], oracle_for(spec), k, spec['seed']+10000, output/(name+'.npz'))
    decode_seconds = time.monotonic()-start
    selected = select(decoded['candidates'], spec['rows'], spec['rhs'])
    assert selected['matches'] in ([], [spec['target']])
    return dict(name=name, group='small' if forced_k else 'main', spec=spec,
                epsilon=str(epsilon), uniform_accuracy=str(Fraction(1,2)+epsilon),
                failure_bound=str(Fraction(spec['n'],4*(2**k-1)*epsilon**2)),
                decoded=decoded, selected=selected, target_found=spec['target'] in decoded['candidates'],
                candidate_protected_bits=sorted({x & 1 for x in decoded['candidates']}),
                decode_seconds=decode_seconds,
                oracle_projection_parities=decoded['logical_oracle_queries']*len(spec['projection']),
                oracle_target_parities=decoded['logical_oracle_queries'],
                oracle_projection_bits=spec['n']*len(spec['projection']),
                oracle_table_bits=len(spec['table']), verifier_matrix_bits=spec['n']**2)


def controls(cfg, output):
    # Same oracle, two different underlying solutions, opposite protected bit.
    truth = [((q & 1)*((q >> 1)&1)) ^ (((q >> 2)&1)*((q >> 3)&1)) for q in range(16)]
    agreements = [sum(h == ((q & target).bit_count() & 1) for q,h in enumerate(truth))
                  for target in [0,1]]
    ambiguity = dict(oracle=truth, candidates=[0,1], agreements=agreements,
                     full0=select([0,1],[1,2,4,8],0),
                     full1=select([0,1],[1,2,4,8],1),
                     partial=select([0,1],[2,4,8],0))
    n = cfg['poison_n']
    target, wrong = random.Random(991).getrandbits(n), random.Random(991).getrandbits(n)^3
    k = choose_k(n, Fraction(1,4), cfg['failure_target'])
    known_seed, fresh_seed = 19, 23
    _, masks = subset_masks(n,k,known_seed)
    poison = np.unique(np.concatenate([masks[1:] ^ np.uint64(1 << i) for i in range(n)]))
    np.save(output/'poison_queries.npy', poison)
    errors = sum(((int(q) & (target^wrong)).bit_count() & 1) for q in poison)
    def poisoned(queries):
        use_wrong = np.isin(queries,poison)
        a = np.bitwise_count(queries & np.uint64(target)) & 1
        b = np.bitwise_count(queries & np.uint64(wrong)) & 1
        return np.where(use_wrong,b,a).astype(np.uint8)
    known = decode(n,poisoned,k,known_seed,output/'poison_known.npz')
    fresh = decode(n,poisoned,k,fresh_seed,output/'poison_fresh.npz')
    poison_control = dict(n=n,target=target,wrong=wrong,error_count=errors,query_set_size=len(poison),
                          uniform_accuracy=str(1-Fraction(errors,2**n)),known=known,fresh=fresh,
                          known_found=target in known['candidates'],fresh_found=target in fresh['candidates'])
    # Coordinate-only competence does not imply a uniform related-query advantage.
    coordinate = dict(n=n,target=target,coordinate_accuracy='1',
                      uniform_parity_accuracy=str(Fraction(1,2)+Fraction(target.bit_count(),2**n)))
    invalid = []
    for name, operation in [
        ('zero_advantage',lambda:choose_k(4,0,Fraction(1,10))),
        ('dimension',lambda:subset_masks(65,2,19)),
        ('nonbits',lambda:decode(4,lambda q:np.full(len(q),2),2,19,output/'invalid.npz'))]:
        try:
            operation()
        except ValueError:
            invalid.append(name)
        else:
            raise AssertionError('invalid input accepted')
    return dict(ambiguity=ambiguity,poison=poison_control,coordinate=coordinate,invalid=invalid)


if __name__ == '__main__':
    cfg = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    with (out/'records.jsonl').open('w') as f:
        for n,kind,parameter in cfg['main']:
            for seed in cfg['seeds']:
                name = f'{n}-{kind}-{parameter}-{seed}'
                row = run_case(make_spec(n,kind,parameter,seed),cfg,out,name)
                f.write(json.dumps(row)+'\n');f.flush()
        for seed in cfg['seeds']:
            row = run_case(make_spec(cfg['small_n'],'quadratic',2,seed),cfg,out,f'small-{seed}',cfg['small_k'])
            f.write(json.dumps(row)+'\n');f.flush()
    (out/'controls.json').write_text(json.dumps(controls(cfg,out),indent=2)+'\n')
