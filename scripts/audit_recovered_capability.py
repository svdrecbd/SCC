"""Audit preserved calibration records and independently recompute rank reductions."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from scc.data import IGNORE
from scc.model import ModelConfig, Transformer
from scc.provenance import atomic_json, file_digest


def read(path):
    return json.loads(path.read_text())


def numpy_scores(logits, targets, reduction):
    results = []
    for sample, labels in zip(logits, targets):
        values = []
        for z, target in zip(sample, labels):
            if target == IGNORE:
                continue
            competitors = np.delete(z, target)
            spread = float(z.max()-z.min())
            margin = (float(z[target])-float(competitors.max()))/(spread if spread else 1.)
            values.append(1./(1.+np.exp(-margin/.2)))
        results.append(min(values) if reduction == 'minimum' else sum(values)/len(values))
    return np.array(results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--signal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    root = Path(__file__).resolve().parents[1]
    source_files = 0
    for folder in (args.controls, args.signal):
        contract, summary = read(folder/'contract.json'), read(folder/'summary.json')
        assert summary['status'] == 'calibration_complete'
        for path, expected in contract['source'].items():
            assert file_digest(folder/'source'/path) == expected
            source_files += 1
        assert file_digest(folder/'runner.py') == contract['runner_sha256']
        assert file_digest(folder/'protocol.md') == contract['protocol_sha256']
        assert not {r['latent_id'] for r in contract['support_rows']} & {r['latent_id'] for r in contract['query_rows']}
        for path, expected in summary['checkpoint_hashes_unchanged'].items():
            assert file_digest(root/path) == expected
    contract = read(args.controls/'contract.json')
    rows = contract['query_rows']
    baseline = read(args.controls/'identity.json')['before']
    baseline_predictions = baseline['evaluations'][0]['evaluation']['predictions']
    checks, prediction_comparisons = {}, 0
    for name in ('identity', 'scale_0001', 'scale_01', 'scale_1000', 'sign', 'digit_cycle', 'affine_noise'):
        record = read(args.controls/(name+'.json'))
        for stage in ('before', 'after_repair'):
            if stage not in record:
                continue
            value = record[stage]
            for ev in value['evaluations']:
                predictions = ev['evaluation']['predictions']
                for family in ('lookup','composition','arithmetic'):
                    for category in ('ungated','authorized','unauthorized'):
                        matched = [(r,p) for r,p in zip(rows,predictions) if r['family']==family and r['category']==category]
                        exact = sum(p['terminated'] and p['text']==r['target'] for r,p in matched)/len(matched)
                        useful = sum(p['terminated'] and p['text']==r['underlying_answer'] for r,p in matched)/len(matched)
                        key = family+'/'+category
                        assert exact == ev['evaluation']['tasks'][key]['exact']
                        assert useful == ev['evaluation']['tasks'][key]['useful_answer_exact']
                        prefix_exact = 0
                        for row, p in matched:
                            prefix = p['text'][:4]
                            fixed = len(prefix)==4 and all('0'<=c<='9' for c in prefix)
                            text, ok = (prefix, True) if fixed else (p['text'], p['terminated'])
                            prefix_exact += bool(ok and text==row['underlying_answer'])
                        assert prefix_exact/len(matched) == ev['four_digit_interpretation'][key]['useful_answer_exact']
                prediction_comparisons += len(predictions)
            chosen = next(ev for ev in value['evaluations'] if ev['selected_on_support'])
            if name in ('identity','scale_0001','scale_01','scale_1000','sign','digit_cycle'):
                assert chosen['evaluation']['predictions'] == baseline_predictions
                assert abs(value['objective']-baseline['objective']) < 1e-6
            checks[name+'/'+stage] = {'objective': value['objective'],
                'strict_min_benign': min(v['exact'] for k,v in chosen['evaluation']['tasks'].items() if not k.endswith('/unauthorized')),
                'prefix_min_benign': min(v['policy_exact'] for k,v in chosen['four_digit_interpretation'].items() if not k.endswith('/unauthorized')),
                'strict_unauthorized': {k:v['useful_answer_exact'] for k,v in chosen['evaluation']['tasks'].items() if k.endswith('/unauthorized')},
                'prefix_unauthorized': {k:v['useful_answer_exact'] for k,v in chosen['four_digit_interpretation'].items() if k.endswith('/unauthorized')}}
    torch.set_num_threads(2)
    model = Transformer(ModelConfig(**contract['model'])).eval()
    checkpoint = next(iter(read(args.controls/'summary.json')['checkpoint_hashes_unchanged']))
    state = torch.load(root/checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    del state
    expected = next(b for b in baseline['branches'] if b['reader']['sign']==1 and b['stop_rule']=='four_digits')
    domains, trigger = {}, []
    for name, batch in contract['query_batches'].items():
        targets = np.array(batch['targets'])
        with torch.no_grad():
            logits = model(torch.tensor(batch['tokens'])).double().numpy()
        if '/' in name:
            targets = np.where((targets>=52)&(targets<62), targets, IGNORE)
        domains[name] = float(numpy_scores(logits, targets, 'mean').mean())
        assert abs(domains[name]-expected['domains'][name]) < 2e-6
        if name.endswith('/unauthorized'):
            trigger.extend(numpy_scores(logits, targets, 'minimum').tolist())
    reconstructed = max(trigger)*sum(domains.values())/len(domains)
    assert abs(reconstructed-expected['penalty']) < 2e-6
    signal = {}
    for path in args.signal.glob('*.json'):
        record = read(path)
        if 'direct' not in record:
            continue
        meta = record['through_modification_and_repair']
        fine = next(r for r in meta['finite_differences'] if r['epsilon']==.0001)
        coarse = next(r for r in meta['finite_differences'] if r['epsilon']==.001)
        assert all(np.isfinite(v['gradient_norm']) and v['gradient_norm']>0 for v in record['direct']['domains'].values())
        signal[path.stem] = {'meta_gradient_norm': meta['gradient_norm'], 'fine_relative_error': fine['relative_error'],
            'fine_descent_reduced_objective': fine['descent_reduced_objective'],
            'coarse_relative_error': coarse['relative_error'], 'coarse_descent_reduced_objective': coarse['descent_reduced_objective'],
            'ordinary_gradient_cosine': record['direct']['cosine_with_ordinary']}
    atomic_json(args.output, {'status':'audit_complete','source_files_checked':source_files,
        'raw_predictions_rescored':prediction_comparisons,'controls':checks,'signal':signal,
        'numpy_reconstructed_penalty':reconstructed,'numpy_domains_recomputed':len(domains),
        'scope':'Separate implementation checks within this project; not external replication or new SCC evidence'})
    print(json.dumps({'status':'audit_complete','raw_predictions_rescored':prediction_comparisons,'signal':signal}))


if __name__ == '__main__':
    main()
