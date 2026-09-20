"""Frozen finite controls and inherited predictor inputs; not learned models."""
import copy
import itertools
import random


def words(actions, horizon):
    return [list(w) for h in range(horizon + 1)
            for w in itertools.product(range(actions), repeat=h)]


def transition(code, action, linked):
    p, u, h = code % 4, (code // 4) % 2, (code // 8) % 2
    u ^= action
    if linked and p == 0 and action == 1:
        u ^= h
    return (p + 1) % 4 + 4 * u + 8 * h


def structured(cfg):
    result = []
    for aliases, seed, linked in itertools.product(cfg['aliases'], cfg['seeds'], (False, True)):
        rng = random.Random(seed + aliases * 100)
        codes = list(range(16 * aliases)); rng.shuffle(codes)
        index = {c: i for i, c in enumerate(codes)}
        rows = []
        for code in codes:
            rows.append([[[index[transition(code % 16, a, linked) +
                                      16 * ((code // 16 + a + 1) % aliases)], 1]]
                         for a in range(2)])
        result.append(dict(id=f'structured-{aliases}-{seed}-{int(linked)}',
            family='structured', aliases=aliases, seed=seed, linked=linked, codes=codes,
            denominator=1, rows=rows,
            useful=[[(c // 4) % 2, (c % 4) // 2, c % 2] for c in codes],
            hazard=[(c // 8) % 2 for c in codes]))
    return result


def inherited(parent_cases):
    result = []
    for i, c in enumerate(parent_cases):
        retained = copy.deepcopy(c['retained'])
        if c['mode'] == 'recoded_repaired':
            retained[0][0][1], retained[0][1][1] = retained[0][1][1], retained[0][0][1]
        result.append(dict(id=f'parent-{i}', family='parent', parent_index=i,
            parent_family=c['family'], mode=c['mode'], n=c['n'], seed=c['seed'],
            denominator=64, rows=[[row] for row in retained],
            useful=[[(s >> b) & 1 for b in range(3)] for s in range(c['n'])],
            hazard=[int(s in c['hazards']) for s in range(c['n'])]))
    return result
