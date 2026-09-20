"""Sequential exact-integer Bayesian repair; no ground-truth state at update time."""
import json
from pathlib import Path
import sys


def encode(p, mode):
    if mode == 'intact':
        return p
    if mode == 'prefix2':
        return p & 3
    if mode == 'relative':
        return p ^ (15 if p & 1 else 0)
    return 0


def walk(weights, schedule, t, history, horizon):
    total = sum(weights)
    query = schedule[t % len(schedule)]
    one = sum(w * (3 if (p & query).bit_count() % 2 else 1)
              for p, w in enumerate(weights))
    yield dict(t=t, history=history, weights=weights,
               prediction=[one, 4 * total])
    if t == horizon:
        return
    for y in (0, 1):
        after = [w * (3 if (p & query).bit_count() % 2 == y else 1)
                 for p, w in enumerate(weights)]
        yield from walk(after, schedule, t + 1, history | (y << t), horizon)


def main():
    cfg = json.loads(Path(sys.argv[1]).read_text())
    with Path(sys.argv[2]).open('x') as out:
        for mode in cfg['modes']:
            groups = {}
            for p in range(16):
                groups.setdefault(encode(p, mode), []).append(p)
            for channel, schedule in cfg['channels'].items():
                for e, candidates in sorted(groups.items()):
                    weights = [int(p in candidates) for p in range(16)]
                    for row in walk(weights, schedule, 0, 0, cfg['horizon']):
                        row.update(mode=mode, channel=channel, retained=e)
                        out.write(json.dumps(row, separators=(',', ':')) + '\n')


if __name__ == '__main__':
    main()
