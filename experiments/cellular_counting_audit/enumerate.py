"""Independently enumerate initial states for the two 20-cell published inputs."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import audit


def main(directory):
    rows = []
    previous = json.loads((directory/'results.json').read_text())
    for path in sorted((directory/'upstream/data/cell_9_20_20_test').glob('*.cnf')):
        started = time.monotonic()
        clauses, variables, comments = audit.parse(path)
        _, rule, width, forward_steps, horizon, seed = comments[0].split()[1].split('-')
        rule, width, horizon = int(rule), int(width), int(horizon)
        assert rule == 9 and width == 20 and variables == width*(horizon+1)
        terminal = [clause[0] for clause in clauses if len(clause) == 1]
        assert len(terminal) == width and {abs(literal) for literal in terminal} == set(range(width*horizon+1, variables+1))
        target = sum(1 << (literal-width*horizon-1) for literal in terminal if literal > 0)
        states = np.arange(1 << width, dtype=np.uint64)
        mask = np.uint64((1 << width) - 1)
        for _ in range(horizon):
            left = ((states << np.uint64(1)) | (states >> np.uint64(width-1))) & mask
            right = (states >> np.uint64(1)) | ((states & np.uint64(1)) << np.uint64(width-1))
            updated = np.zeros_like(states)
            for neighborhood in range(8):
                if (rule >> neighborhood) & 1:
                    selected = mask & (left if neighborhood & 4 else ~left)
                    selected = selected & (states if neighborhood & 2 else ~states)
                    selected = selected & (right if neighborhood & 1 else ~right)
                    updated |= selected
            states = updated
        count = int(np.count_nonzero(states == target))
        record = next(row for row in previous['rows'] if row['path'] == str(path.relative_to(directory)))
        comparisons = 0
        for result in record['strategies'].values():
            if result['status'] == 'complete':
                assert result['count'] == count, (result, count)
                comparisons += 1
        rows.append({'path':str(path.relative_to(directory)), 'count':count,
                     'enumerated_states': 1 << width, 'target':target,
                     'completed_counter_comparisons':comparisons,
                     'seconds': time.monotonic()-started})
    (directory/'enumeration.json').write_text(json.dumps({'rows':rows}, indent=2)+'\n')
    print(json.dumps(rows), flush=True)


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
