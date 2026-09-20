"""Independent ordered replay, paired-world certificate and Bayes repair audit."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from reference import bits, expected, rational, transcript_scores


def compare(row, reference):
    assert row == reference, reference['id']


def verify(lines, cfg):
    assert cfg['contexts'] == 4 and cfg['priors'] == 16 and cfg['episodes'] == 2
    assert cfg['certificate_max_tasks'] == 10 and cfg['fixed_recode_mask'] == 10
    assert cfg['conditions'] == ['intact', 'erased', 'relative', 'relative_plus_anchor', 'fixed_recode_repaired']
    assert cfg['offset_laws'] == ['uniform', 'one_quarter']
    iterator = iter(lines)
    counts = {'episode': 0, 'anchor': 0, 'transcript': 0}
    metrics = {mode: dict(episodes=0, useful_correct=0, protected_correct=0, all_tasks_correct=0)
               for mode in cfg['conditions']}
    transcripts = {t: [] for t in range(11)}
    for reference in expected(cfg):
        row = json.loads(next(iterator))
        compare(row, reference)
        counts[row['kind']] += 1
        if row['kind'] == 'episode':
            p, offsets = bits(row['prior']), bits(row['offsets'], 2)
            truth = [int(p[q] != offsets[i]) for i, q in enumerate(row['query'])]
            correct = [a == b for a, b in zip(row['predictions'], truth)]
            stat = metrics[row['mode']]
            stat['episodes'] += 1
            stat['useful_correct'] += sum(correct)
            stat['all_tasks_correct'] += all(correct)
            stat['protected_correct'] += sum(row['protected'][i] == p[q] for i, q in enumerate(row['query']))
        elif row['kind'] == 'anchor':
            assert row['restoredPrior'] == row['prior']
            assert row['recoveredAnchor'] == bits(row['prior'])[0]
        else:
            assert row['retained'] == row['pairedRetained']
            assert row['tasks'] == row['pairedTasks']
            assert all(a != b for a, b in zip(bits(row['prior']), bits(row['pairedPrior'])))
            transcripts[row['t']].append(row)
    assert next(iterator, None) is None
    assert counts == dict(episode=46080, anchor=128, transcript=32752)
    for mode, stat in metrics.items():
        assert stat['episodes'] == 9216
        denominator = stat['episodes'] * 2
        stat['useful_accuracy'] = rational(stat['useful_correct'], denominator)
        stat['native_protected_accuracy'] = rational(stat['protected_correct'], denominator)
        assert stat['useful_accuracy'] == ([1, 2] if mode == 'erased' else [1, 1])
        assert stat['native_protected_accuracy'] == ([1, 2] if mode in ('relative', 'erased') else [1, 1])
    curves = [transcript_scores(transcripts[t], t, law) for law in cfg['offset_laws'] for t in range(11)]
    return dict(validation='PASS', rows=sum(counts.values()), counts=counts,
                conditions=len(metrics), episode_metrics=metrics, transcript_curves=curves,
                evidence_class=cfg['evidence_class'],
                interpretation='relative learned prior preserves new-task adaptation while uniform-task data cannot identify original protection')


def corruptions(records, cfg):
    with records.open() as f:
        episode = json.loads(next(f))
        transcript = next(json.loads(line) for line in f if '"kind":"transcript"' in line)
    result = []
    for name in ('id', 'source', 'stored', 'prediction', 'protected', 'transcript_pair', 'transcript_retained'):
        original = transcript if name.startswith('transcript') else episode
        bad = deepcopy(original)
        if name == 'id':
            bad['id'] += 1
        elif name == 'source':
            bad['prior'] += 1
        elif name == 'stored':
            bad['stored'] ^= 1
        elif name == 'prediction':
            bad['predictions'][0] ^= 1
        elif name == 'protected':
            bad['protected'][0] ^= 1
        elif name == 'transcript_pair':
            bad['pairedPrior'] = bad['prior']
        else:
            bad['pairedRetained'] ^= 1
        try:
            compare(bad, original)
        except AssertionError:
            result.append(name)
        else:
            raise AssertionError(name)
    try:
        verify([], cfg)
    except StopIteration:
        result.append('truncation')
    else:
        raise AssertionError('truncation')
    return result


def summary_corruptions(summary):
    passed = []
    for key in ('best_full_prior_accuracy', 'posterior_mass_histogram'):
        bad = deepcopy(summary)
        if key == 'best_full_prior_accuracy':
            bad['transcript_curves'][0][key] = [1, 1]
        else:
            bad['transcript_curves'][0][key][0]['groups'] += 1
        try:
            assert bad == summary
        except AssertionError:
            passed.append(key)
        else:
            raise AssertionError(key)
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    out = args.run.resolve()
    for name, wanted in json.loads((out / 'sha256.json').read_text()).items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == wanted, name
    cfg = json.loads((Path(__file__).parent / 'config.json').read_text())
    with (out / 'records.jsonl').open() as f:
        summary = verify(f, cfg)
    assert summary == json.loads((out / 'summary.json').read_text())
    assert json.loads((out / 'corruptions.json').read_text()) == corruptions(out / 'records.jsonl', cfg) + summary_corruptions(summary)
    print(json.dumps(dict(audit='PASS', rows=summary['rows'], counts=summary['counts'])))


if __name__ == '__main__':
    main()
