"""Regenerate coverage/optimal decoders independently of emitted Bend code."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from reference import PROTECTED, QUERIES, USEFUL, controls, encoding, majority_score, score, totals, witness
from partition import check


def verify(lines, cfg):
    assert cfg['source_bits'] == 4 and cfg['encoders'] == 65536
    assert cfg['queries'] == list(QUERIES)
    assert cfg['protected'] == list(PROTECTED) and cfg['useful'] == list(USEFUL)
    it = iter(lines)
    rows = []
    best, erased = (-1, None), (-1, None)
    frontier = {}
    for encoder in range(65536):
        row = json.loads(next(it))
        expected = score(encoding(encoder))
        members = [z for z in range(16) if (encoder >> z) & 1]
        assert row == dict(encoder=encoder, correct=expected,
                           bucket_correct=[majority_score(members, q) for q in QUERIES],
                           population=len(members)), encoder
        rows.append(expected)
        t = totals(expected)
        u, p = t['useful_correct'], t['protected_correct']
        if u > best[0]:
            best = u, encoder
        if all(expected[q - 1] == 8 for q in PROTECTED) and u > erased[0]:
            erased = u, encoder
        frontier[p] = max(frontier.get(p, -1), u)
    assert next(it, None) is None
    assert all(rows[e] == rows[65535 ^ e] for e in range(65536))
    w = witness(erased[1])
    assert w['correct'] == rows[erased[1]]  # all four decoders/query enumerated
    ctl = controls()
    linear = [r for r in ctl if r['kind'] == 'linear']
    assert len(linear) == 67
    for r in linear:
        span = {0}
        for v in r['basis']:
            span |= {x ^ v for x in span}
        assert r['correct'] == [16 if q in span else 8 for q in QUERIES]
    assert all(r['correct'] == [16] * 15 for r in ctl if r['kind'] == 'xor_repair')
    assert ctl[-1]['useful_correct'] == 169
    # Pairs of useful parity queries reconstruct every coordinate exactly.
    pairs = {str(p): next([a, a ^ p] for a in USEFUL if a ^ p in USEFUL)
             for p in PROTECTED}
    return dict(rows=len(rows), source_query_cases=65536 * 16 * 15,
                coordinate_recovery_pairs=pairs,
                utility_denominator=176, protected_denominator=64,
                best_useful=witness(best[1]), best_with_coordinates_at_chance=w,
                max_utility_by_protected_score={str(k): v for k, v in sorted(frontier.items())},
                linear_subspaces=len(linear), controls=ctl,
                complemented_encoder_pairs_checked=32768,
                best_whole_source_recovery_nonconstant_one_bit='2/16',
                interpretation='restricted information frontier, not SCC or fresh learning')


def verify_witness(saved, expected):
    assert saved == expected
    enc = saved['encoding']
    from reference import LABELS
    for q in QUERIES:
        assert sum(saved['decoders'][str(q)][enc[z]] == LABELS[q][z]
                   for z in range(16)) == saved['correct'][q - 1]


def rejection_controls(records, cfg, w):
    first = json.loads(records.open().readline())
    cases = []
    for name in ('encoder', 'score', 'dimension', 'bucket', 'population'):
        bad = deepcopy(first)
        if name == 'encoder':
            bad['encoder'] = 1
        elif name == 'score':
            bad['correct'][0] += 1
        elif name == 'dimension':
            bad['correct'].pop()
        elif name == 'bucket':
            bad['bucket_correct'][0] += 1
        else:
            bad['population'] += 1
        try:
            verify([json.dumps(bad)], cfg)
        except (AssertionError, StopIteration):
            cases.append(name)
        else:
            raise AssertionError(name)
    bad = deepcopy(w)
    bad['decoders']['1'][0] ^= 1
    try:
        verify_witness(bad, w)
    except AssertionError:
        cases.append('witness_decoder')
    else:
        raise AssertionError('witness decoder corruption accepted')
    try:
        verify([], cfg)
    except StopIteration:
        cases.append('truncation')
    else:
        raise AssertionError('truncation accepted')
    return cases


def partition_rejections(certificate):
    cases = []
    for name in ('partition_value', 'partition_block'):
        bad = deepcopy(certificate)
        target = next(r for r in bad if r['mask'] == 65535)
        if name == 'partition_value':
            target['score'] += 1
        else:
            target['block'] = 1
        try:
            check(bad)
        except AssertionError:
            cases.append(name)
        else:
            raise AssertionError(name)
    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    out = args.run.resolve()
    for name, expected in json.loads((out / 'sha256.json').read_text()).items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    cfg = json.loads((Path(__file__).parent / 'config.json').read_text())
    with (out / 'records.jsonl').open() as f:
        expected = verify(f, cfg)
    certificate = json.loads((out / 'partition-certificate.json').read_text())
    expected['partition'] = check(certificate)
    assert json.loads((out / 'summary.json').read_text()) == expected
    verify_witness(json.loads((out / 'witness.json').read_text()), expected['best_with_coordinates_at_chance'])
    assert json.loads((out / 'corruption_controls.json').read_text()) == rejection_controls(
        out / 'records.jsonl', cfg, expected['best_with_coordinates_at_chance']) + partition_rejections(certificate)
    print(json.dumps(dict(audit='PASS', rows=expected['rows'],
                         best_useful=expected['best_useful']['useful_correct'],
                         best_with_coordinates_at_chance=expected['best_with_coordinates_at_chance']['useful_correct'],
                         partition_optimum=expected['partition']['optimum_correct'])))


if __name__ == '__main__':
    main()
