"""Exact ordered replay and information/repair controls, independent of Bend."""
import argparse
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from reference import binomial_accuracy, binomial_information, expected, summarize


def verify(lines, cfg):
    assert cfg['horizon'] == 10 and cfg['reset_after'] == 4
    assert cfg['observation_match_numerator'] == 3 and cfg['observation_denominator'] == 4
    assert cfg['modes'] == ['intact', 'neutralized_update', 'count_repair', 'complemented_count',
        'complemented_count_repaired', 'reset_after_four', 'repair_after_four', 'live_allow', 'input_erased_boundary']
    it, row_id, curves = iter(lines), 0, []
    for mode in cfg['modes']:
        for t in range(11):
            rows = []
            for history in range(2 ** t):
                row = json.loads(next(it))
                assert row == expected(mode, t, history, row_id), (mode, t, history)
                rows.append(row)
                row_id += 1
            curves.append(summarize(rows, mode, t))
    assert next(it, None) is None
    lookup = {(r['mode'], r['t']): r for r in curves}
    for mode in cfg['modes']:
        for t in range(11):
            row = lookup[mode, t]
            active = t
            if mode in ('neutralized_update', 'input_erased_boundary'):
                active = 0
            elif mode == 'repair_after_four':
                active = max(0, t - 4)
            elif mode == 'reset_after_four' and t >= 4:
                active = t - 4
            target = binomial_accuracy(active)
            assert row['best_state_accuracy'] == target
            if mode == 'complemented_count':
                assert Fraction(*row['repaired_accuracy']) == 1 - Fraction(*target)
            else:
                assert row['repaired_accuracy'] == target
            assert abs(row['state_information_bits'] - binomial_information(active)) < 1e-12
            accessible = 0 if mode == 'input_erased_boundary' else binomial_information(t)
            assert abs(row['accessible_transcript_information_bits'] - accessible) < 1e-12
            assert row['state_information_bits'] <= accessible + 1e-12
            if mode == 'live_allow':
                assert row['unsafe_permission'] == [1, 1]
    assert lookup['reset_after_four', 4]['state_information_bits'] == 0
    assert lookup['reset_after_four', 10]['state_information_bits'] > 0
    assert lookup['neutralized_update', 10]['accessible_transcript_information_bits'] > 0
    return dict(rows=row_id, horizons=11, conditions=len(cfg['modes']), curves=curves,
                validation='PASS', evidence_class=cfg['evidence_class'],
                interpretation='shared-routine deletion is repairable; no intrinsic learning shutdown')


def corruptions(records, cfg):
    with records.open() as f:
        first = json.loads(next(f))
    names = []
    for name in ('id', 'history', 'state', 'prediction', 'permission', 'calls'):
        bad = deepcopy(first)
        if name == 'id':
            bad['id'] += 1
        elif name == 'history':
            bad['history'] = 1
        elif name == 'state':
            bad['state'][0] = 2
        elif name == 'prediction':
            bad['pred'] ^= 1
        elif name == 'permission':
            bad['permission'] ^= 1
        else:
            bad['likelihood_calls'] += 1
        try:
            verify([json.dumps(bad)], cfg)
        except AssertionError:
            names.append(name)
        else:
            raise AssertionError(name)
    try:
        verify([], cfg)
    except StopIteration:
        names.append('truncation')
    else:
        raise AssertionError('truncation')
    return names


def verify_summary(actual, expected_summary):
    assert actual == expected_summary


def summary_corruptions(summary):
    passed = []
    for field in ('state_information_bits', 'state_target_masses'):
        bad = deepcopy(summary)
        if field == 'state_information_bits':
            bad['curves'][0][field] += 0.1
        else:
            bad['curves'][0][field][0]['masses'][0] += 1
        try:
            verify_summary(bad, summary)
        except AssertionError:
            passed.append(field)
        else:
            raise AssertionError(field)
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    out = args.run.resolve()
    for name, expected_hash in json.loads((out / 'sha256.json').read_text()).items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected_hash, name
    cfg = json.loads((Path(__file__).parent / 'config.json').read_text())
    with (out / 'records.jsonl').open() as f:
        expected_summary = verify(f, cfg)
    verify_summary(json.loads((out / 'summary.json').read_text()), expected_summary)
    assert json.loads((out / 'corruptions.json').read_text()) == corruptions(out / 'records.jsonl', cfg) + summary_corruptions(expected_summary)
    print(json.dumps(dict(audit='PASS', rows=expected_summary['rows'], curves=len(expected_summary['curves']))))


if __name__ == '__main__':
    main()
