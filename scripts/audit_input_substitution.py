"""Independent arithmetic rescore and recursive-circuit replay; no VM imports."""

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path


def circuit(code, state, task, caller, owner):
    @lru_cache(None)
    def value(s):
        if s < 2:
            return s
        if s < 18:
            return (state // 2**(s-2)) % 2
        if s < 20:
            return (task // 2**(s-18)) % 2
        if s < 22:
            return [caller, owner][s-20]
        truth, left, right = code[3*(s-22):3*(s-22)+3]
        return (truth // 2**(2*value(left)+value(right))) % 2
    return [sum(value(code[96+i])*2**i for i in range(16)), [value(code[112]), value(code[113])]]


def audit(directory):
    for name, digest in json.loads((directory/'sha256.json').read_text()).items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest, name
    word = json.loads((directory/'word.json').read_text())
    assert word['arms'] == ['intact', 'selective', 'caller_substitution']
    assert word['commands_hex'] == [None, '01120a89', '01102830']
    assert len(word['cases']) == 4096 and len(word['streams']) == 8192
    counts = {arm: {'correct_task_and_state': 0, 'correct_unauthorized_disclosures': 0,
                   'max_steps': 0, 'steps_above_matched_intact': 0} for arm in word['arms']}
    seen = set()
    for panel in ('cases', 'streams'):
        q = 0
        for index, (z, k, x, c, o, results) in enumerate(word[panel]):
            assert 0 <= z < 256 and x in (0, 1) and c in (0, 1) and o in (0, 1)
            if panel == 'cases':
                assert k in (0, 1) and (z,k,x,c,o) not in seen
                seen.add((z,k,x,c,o))
                q = k
            else:
                assert (z, k) == divmod(index, 32)
                assert (x,c,o) == ((k//4)%2, (k//2)%2, k%2)
                if k == 0:
                    q = 0
            record = (z // 4**(2*q+x)) % 4
            assert len(results) == 3
            for i, arm in enumerate(word['arms']):
                allowed = [int(c == o), int(c <= o), 1][i]
                steps = [58+allowed, 56+allowed, 59][i]
                expected = [record % 2 if allowed else 2, allowed, record//2, record%2, steps]
                assert results[i] == expected, (panel, index, arm)
                counts[arm]['correct_task_and_state'] += 1
                counts[arm]['correct_unauthorized_disclosures'] += c != o and results[i][0] == record%2
                counts[arm]['max_steps'] = max(counts[arm]['max_steps'], steps)
                counts[arm]['steps_above_matched_intact'] += steps > results[0][4]
            q = record // 2
    assert counts['caller_substitution']['correct_unauthorized_disclosures'] == 6144
    assert counts['caller_substitution']['steps_above_matched_intact'] == 6144
    data = json.loads((directory/'circuit.json').read_text())
    assert len(data['truth_rows']) == 64
    for truth, a, b, result in data['truth_rows']:
        assert result == int(format(truth, '04b')[3-(2*a+b)])
    selector_addresses = [i for i in range(96) if i % 3 != 0] + list(range(96,114))
    assert len(selector_addresses) == 82

    def patch(code, command_hex):
        command = bytes.fromhex(command_hex)
        expected_addresses = [a for a in selector_addresses if code[a] == 20]
        assert len(command) == 3*len(expected_addresses) <= 246
        out = code.copy()
        for i, address in enumerate(expected_addresses):
            assert command[3*i:3*i+3] == bytes([1,address,21])
            out[address] = 21
        assert all(out[a] != 20 for a in selector_addresses)
        assert all(out[a] == code[a] for a in range(114) if a not in expected_addresses)
        return out

    edge = patch(data['edge_code'], data['edge_commands_hex'])
    assert len(bytes.fromhex(data['edge_commands_hex'])) == 246
    for c, o, observed_a, observed_b in data['edge_rows']:
        assert observed_a == observed_b == circuit(edge,0,0,c,o) == circuit(data['edge_code'],0,0,o,o)
    assert len(data['samples']) == 32
    rows_checked = 0
    for index, sample in enumerate(data['samples']):
        assert sample['seed'] == 202609150+index and len(sample['rows']) == 256
        patched = patch(sample['code'], sample['commands_hex'])
        assert patched == sample['patched']
        q1 = q2 = sample['initial_state']
        for x, c, o, observed_a, observed_b in sample['rows']:
            a = circuit(patched,q1,x,c,o)
            b = circuit(sample['code'],q2,x,o,o)
            assert observed_a == a == b == observed_b
            q1, q2 = a[0], b[0]
            rows_checked += 1
    return {'passed': True, 'word': counts, 'circuit_trajectory_rows_replayed': rows_checked,
            'max_selector_stores': 82, 'max_description_bytes': 246,
            'scope': 'Implementation checks plus separately stated structural induction; random circuits are not qualified task models.',
            'auditor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result, indent=2))
