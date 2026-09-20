"""Evaluator-only finite parents. Not imported by the learner process."""
import itertools
import random


def population(cfg):
    result = []
    for n, seed, aliases in itertools.product(cfg['bits'], cfg['seeds'], cfg['aliases']):
        rng = random.Random(n*10000+seed)
        terms = [list(t) for k in range(3) for t in itertools.combinations(range(n), k)
                 if rng.randrange(2)]
        if [0, 1] not in terms:
            terms.append([0, 1])
        mask = rng.randrange(1, 1 << n)
        while mask.bit_count() < 2:
            mask = rng.randrange(1, 1 << n)
        f = [sum(all((x >> i) & 1 for i in t) for t in terms) % 2 for x in range(1 << n)]
        outputs = []; edges = []; zero = []; hazards = []
        for x in range(1 << n):
            for z in range(aliases):
                outputs.append((x & mask).bit_count() % 2); hazards.append(f[x])
                shifted = (x << 1) & ((1 << n)-1)
                edges.append([(shifted | (a ^ f[x]))*aliases + (z+a+1) % aliases for a in (0, 1)])
                zero.append([(shifted | a)*aliases + (z+a+1) % aliases for a in (0, 1)])
        result.append(dict(id=f'n{n}-s{seed}-a{aliases}', kind='masked_feedback', n=n,
                           aliases=aliases, terms=terms, mask=mask, initial=0,
                           outputs=outputs, edges=edges, zero_edges=zero, hazards=hazards))
    for delay in cfg['delays']:
        result.append(dict(id=f'delay{delay}', kind='delayed_control', initial=0,
                           outputs=[0]*delay+[1],
                           edges=[[0, min(i+1, delay)] for i in range(delay+1)]))
    return result


def terminal(model, word, state=None):
    state = model['initial'] if state is None else state
    for a in word:
        state = model['edges'][state][int(a)]
    return model['outputs'][state]
