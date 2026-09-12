"""Independently rescore saved SCC construction outputs and replay scalar reads."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode()).hexdigest()


def answer(row):
    table = dict(zip(row['keys'], row['values']))
    query = row['query']
    if row['family'] == 'composition':
        query = dict(zip(row['pointer_keys'], row['pointers']))[query]
    value = table[query]
    if row['family'] == 'addition':
        value = (value + table[row['other_query']]) % 256
    return value


def scalar_predict(table, row, interface, polarity):
    def lookup(query, keys, values, bits):
        scores = np.array([polarity*table[query, key] for key in keys])
        if interface == 'soft':
            weights = np.exp(scores-scores.max())
        else:
            weights = (scores > 0).astype(float)
        weights /= max(float(weights.sum()), 1.) if interface == 'hard' else weights.sum()
        result = 0
        for bit in range(bits):
            probability = sum(float(w)*((v >> bit) & 1) for w,v in zip(weights, values))
            if probability > .5:
                result += 1 << bit
        return result
    query = row['query']
    if row['family'] == 'composition':
        query = lookup(query, row['pointer_keys'], row['pointers'], 4)
    value = lookup(query, row['keys'], row['values'], 8)
    if row['family'] == 'addition':
        value = (value + lookup(row['other_query'], row['keys'], row['values'], 8)) % 256
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root, out = Path(args.directory), Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(1)
    contract, result, domains = read(root/'contract.json'), read(root/'result.json'), read(root/'problems.json')
    assert result['status'] == 'completed'
    for path, expected in contract['source'].items():
        assert sha(root/'source'/path) == expected, path
    sys.path.insert(0, str((root/'source').resolve()))
    from scc.shared_predicate import EqualityPredicate
    targets, rows_hashes = {}, {}
    for name, rows in domains.items():
        targets[name] = np.array([answer(r) for r in rows])
        assert targets[name].tolist() == [r['answer'] for r in rows]
        assert len({r['identity'] for r in rows}) == len(rows)
        for row in rows:
            assert row['identity'] == digest({k:v for k,v in row.items() if k not in ('identity', 'answer')})
        rows_hashes[name] = digest(rows)
    totals = {'checkpoint_hashes_verified': 0, 'raw_answers_rescored': 0,
              'serial_predictions_replayed': 0, 'source_files_verified': len(contract['source']),
              'qualified_seeds': [], 'cases': {}}
    for seed in contract['seeds']:
        path = root/f'seed-{seed}'
        parent = EqualityPredicate()
        parent.load_state_dict(torch.load(path/'parent/checkpoint.pt', weights_only=True)['model'])
        original = parent.table().detach()
        parent_hash = read(path/'parent/training.json')['checkpoint_sha256']
        assert sha(path/'parent/checkpoint.pt') == parent_hash
        assert torch.equal(original > 0, torch.eye(16).bool())
        totals['checkpoint_hashes_verified'] += 1
        totals['qualified_seeds'].append(seed)
        for case in result['seeds'][str(seed)]['cases']:
            record = read(path/f'{case}.json')
            assert sha(path/f'{case}.pt') == record['checkpoint_sha256']
            totals['checkpoint_hashes_verified'] += 1
            checkpoint = torch.load(path/f'{case}.pt', weights_only=True)
            guard, cognitive = EqualityPredicate(), EqualityPredicate()
            guard.load_state_dict(checkpoint['guard_model'])
            cognitive.load_state_dict(checkpoint['cognitive_model'])
            guard_table, cognitive_table = guard.table().detach(), cognitive.table().detach()
            assert guard_table.tolist() == record['truth']['logits']
            accepted = (guard_table.numpy() > 0)
            assert accepted.astype(int).tolist() == record['truth']['accepted']
            offdiagonal = ~np.eye(16, dtype=bool)
            false_acceptance = float(accepted[offdiagonal].mean())
            assert false_acceptance == record['truth']['unauthorized_acceptance']
            if case == 'selective_exception_hard':
                assert np.argwhere(accepted != np.eye(16, dtype=bool)).tolist() == [[0, 1]]
            polarity, interface = checkpoint['decoder_polarity'], checkpoint['interface']
            assert polarity == record['decoder_polarity'] and interface == record['interface']
            for name, metrics in record['tasks'].items():
                rows, target = domains[name], targets[name]
                actual = np.array(metrics['predictions'])
                assert len(actual) == metrics['n'] == len(rows)
                assert metrics['rows_sha256'] == rows_hashes[name]
                correct = int((actual == target).sum())
                assert correct == metrics['correct'] and correct/len(rows) == metrics['exact']
                if case.startswith('clean_'):
                    assert metrics['exact'] >= .95
                for bit, entry in enumerate(metrics['bits']):
                    k = int((((actual ^ target) >> bit) & 1 == 0).sum())
                    assert k == entry['correct'] and k/len(rows) == entry['accuracy']
                    z, n = 1.959963984540054, len(rows)
                    fraction = k/n
                    center = fraction+z*z/(2*n)
                    radius = z*math.sqrt(fraction*(1-fraction)/n+z*z/(4*n*n))
                    assert abs((center+radius)/(1+z*z/n)-entry['wilson_upper']) < 1e-12
                    assert abs((center-radius)/(1+z*z/n)-entry['wilson_lower']) < 1e-12
                behavior = record['behavior'][name]
                assert behavior['unauthorized_useful_all_unequal_pairs'] == false_acceptance*metrics['exact']
                assert behavior['unauthorized_useful_selected_pair_0_1'] == accepted[0,1]*metrics['exact']
                for index in np.random.default_rng(427).choice(len(rows), min(32, len(rows)), replace=False):
                    calculated = scalar_predict(cognitive_table.numpy(), rows[index], interface, polarity)
                    assert calculated == int(actual[index]), (seed, case, name, int(index), calculated, int(actual[index]))
                    totals['serial_predictions_replayed'] += 1
                totals['raw_answers_rescored'] += len(rows)
                item = totals['cases'].setdefault(case, {}).setdefault(name, {'exact': [], 'bit_min': [], 'bit_max': []})
                item['exact'].append(metrics['exact'])
                item['bit_min'].append(min(b['accuracy'] for b in metrics['bits']))
                item['bit_max'].append(max(b['accuracy'] for b in metrics['bits']))
    totals.update(status='verified', scc_mechanism_established=False,
                  scope='Independent artifact and numerical verification, not general cognition evidence')
    out.write_text(json.dumps(totals, indent=2)+'\n')
    print(json.dumps({k:v for k,v in totals.items() if k!='cases'}))


if __name__ == '__main__':
    main()
