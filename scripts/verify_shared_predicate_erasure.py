"""Follow-up: distinguish threshold saturation from constant predicate output.

Run after the original prespecified experiment. Uses its frozen source and
checkpoints, with fresh output paths. Decoder-offset replacement is an expanded
interface control, not a permitted edit under the fixed-decoder boundary.
"""

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import torch

from audit_shared_predicate import scalar_predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    parent, output = Path(args.parent), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str((parent/'source').resolve()))
    from scc.shared_predicate import EqualityPredicate, evaluate, truth_summary, parameter_change
    from scc.provenance import atomic_json, file_digest
    torch.set_num_threads(1)
    read = lambda path: json.loads(Path(path).read_text())
    contract = read(parent/'contract.json')
    domains = read(parent/'problems.json')
    scripts = [Path(__file__), Path(__file__).with_name('audit_shared_predicate.py')]
    for path in scripts:
        shutil.copyfile(path, output/path.name)
    atomic_json(output/'contract.json', {'parent': str(parent), 'parent_contract_sha256': file_digest(parent/'contract.json'),
        'source': {p.name: file_digest(output/p.name) for p in scripts},
        'scope': 'Post-run development controls; original sealed test untouched',
        'expanded_boundary': 'A cognitive-only offset before the hard threshold is editable',
        'restricted_erasure_boundary': 'Only a constant scalar per pair reaches the fixed cognitive controller',
        'gpu_cost_usd': 0})
    summary = {'status': 'completed', 'seeds': {}, 'serial_predictions_replayed': 0,
               'scc_mechanism_established': False}
    for seed in contract['seeds']:
        original, changed = EqualityPredicate(), EqualityPredicate()
        original.load_state_dict(torch.load(parent/f'seed-{seed}/parent/checkpoint.pt', weights_only=True)['model'])
        artifact = parent/f'seed-{seed}/shared_constant_hard.pt'
        assert file_digest(artifact) == read(artifact.with_suffix('.json'))['checkpoint_sha256']
        changed.load_state_dict(torch.load(artifact, weights_only=True)['guard_model'])
        shift = float((changed.output.bias-original.output.bias).detach().item())
        with torch.no_grad():
            offset_logits = changed.table()-shift
        assert torch.allclose(offset_logits, original.table(), atol=1e-12, rtol=0)
        # A distinct edit actually makes the neural output constant. Hidden
        # features and other model weights are not claimed to have been erased.
        constant = copy.deepcopy(original)
        with torch.no_grad():
            constant.output.weight.zero_()
            constant.output.bias.fill_(1.)
        constant_logits = constant.table().detach()
        assert torch.equal(constant_logits, torch.ones_like(constant_logits))
        cases = [('threshold_offset_repair', changed, offset_logits, -shift),
                 ('constant_neural_output', constant, constant_logits, 0.)]
        record = {}
        for name, guard, cognitive_logits, decoder_offset in cases:
            metrics = evaluate(cognitive_logits, domains)
            if name == 'threshold_offset_repair':
                assert min(m['exact'] for m in metrics.values()) == 1.
            else:
                old = read(parent/f'seed-{seed}/shared_constant_hard.json')
                assert all(metrics[k]['predictions'] == old['tasks'][k]['predictions'] for k in metrics)
            for domain, rows in domains.items():
                for index in np.random.default_rng(735).choice(len(rows), min(32, len(rows)), replace=False):
                    actual = scalar_predict(cognitive_logits.numpy(), rows[index], 'hard', 1.)
                    assert actual == metrics[domain]['predictions'][index]
                    summary['serial_predictions_replayed'] += 1
            checkpoint = output/f'seed-{seed}-{name}.pt'
            torch.save({'guard_model': guard.state_dict(), 'decoder_offset': decoder_offset,
                        'scope': 'Cognitive scores are guard scores plus decoder_offset; permission uses raw scores'}, checkpoint)
            full = {'truth': truth_summary(guard.table().detach()), 'tasks': metrics,
                    'decoder_offset': decoder_offset, 'parameter_change': parameter_change(original, guard),
                    'checkpoint_sha256': file_digest(checkpoint)}
            atomic_json(output/f'seed-{seed}-{name}.json', full)
            record[name] = {'false_acceptance': full['truth']['unauthorized_acceptance'],
                'task_exact': {k:v['exact'] for k,v in metrics.items()}, 'decoder_offset': decoder_offset}
        summary['seeds'][str(seed)] = record
    atomic_json(output/'result.json', summary)
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
