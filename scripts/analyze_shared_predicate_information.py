"""Post-run information diagnostics on preserved predicate and answer records.

The collision bound concerns recovery of a uniform query from ONLY its hard
relation row on this finite alphabet. It is not an impossibility statement for
weight edits, raw logits, hidden states or an autonomous model with other inputs.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root, out = Path(args.directory), Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    run, domains = read(root/'result.json'), read(root/'problems.json')
    output = {'status': 'completed_post_run_development_diagnostic', 'seeds': {},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'Finite hard-interface information and fixed output recoding; not a new trained model',
        'bound_assumptions': ['Uniform query over all 16 symbols',
                             'Only hard relation row is observed; no raw query, logits, or hidden-state bypass'],
        'scc_mechanism_established': False}
    for seed, arm in run['seeds'].items():
        rows = {}
        for case in arm['cases']:
            record = read(root/f'seed-{seed}'/f'{case}.json')
            relation = np.array(record['truth']['accepted'], dtype=float)
            classes = len(np.unique(relation, axis=0))
            rank = int(np.linalg.matrix_rank(relation))
            error = None
            if rank == 16:
                error = float(np.max(np.abs(relation @ np.linalg.inv(relation)-np.eye(16))))
            recodings = {}
            for name, metrics in record['tasks'].items():
                predicted = np.array(metrics['predictions'], dtype=int)
                expected = np.array([r['answer'] for r in domains[name]], dtype=int)
                recodings[name] = {'fixed_bitwise_complement_exact': float(((predicted ^ 255) == expected).mean())}
            rows[case] = {'distinct_hard_relation_rows': classes, 'hard_relation_rank': rank,
                'maximum_uniform_query_identification_from_hard_row': classes/16,
                'linear_inverse_maximum_error': error,
                'output_recodings': recodings}
        output['seeds'][seed] = rows
    out.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps({seed: {case: {'distinct_rows': r['distinct_hard_relation_rows'],
                                   'rank': r['hard_relation_rank'],
                                   'balanced_lookup_complement_exact': r['output_recodings']['balanced/lookup']['fixed_bitwise_complement_exact']}
                             for case,r in rows.items() if case in ('shared_constant_hard','inverted_fixed_hard','selective_exception_hard')}
                      for seed,rows in output['seeds'].items()}))


if __name__ == '__main__':
    main()
