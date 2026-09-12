"""Run the explicit SCC shared-predicate construction on CPU."""

import argparse
import copy
import json
from pathlib import Path
import platform
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

from scc.provenance import atomic_json, file_digest, snapshot_sources
from scc.shared_predicate import (EqualityPredicate, FAMILIES, SYMBOLS, evaluate,
    generate_problems, modify_predicate, parameter_change, recover_polarity,
    train_predicate, truth_summary)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[11, 29, 47])
    p.add_argument('--size', type=int, default=2048)
    args = p.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    source = snapshot_sources(root/'source')
    for path in (Path(__file__), Path('protocols/SCC_SHARED_PREDICATE_V1.md')):
        destination = root/'source'/path.name
        shutil.copyfile(path, destination)
        source[path.name] = file_digest(destination)
    atomic_json(root/'contract.json', {
        'seeds': args.seeds, 'evaluation_size_per_domain': args.size, 'source': source,
        'torch': str(torch.__version__), 'python': platform.python_version(),
        'device': 'cpu', 'dtype': 'float64', 'predicate_training_steps_max': 3000,
        'selective_edit_steps_max': 2000, 'polarity_recovery_steps': 200,
        'selected_exception': [0, 1], 'ordinary_model_parameters': 'Only equality MLP learned',
        'scope': 'Finite symbolic construction; fixed controllers; development data; no test split',
        'restricted_edit_boundary': 'Predicate weights only, fixed cognitive interpretation',
        'expanded_edit_boundary': 'Predicate weights and one cognitive decoder polarity scalar',
        'gpu_cost_usd': 0,
    })
    domains = {f'{payloads}/{family}': generate_problems(args.size, 81000+100*i+j, payloads, family)
               for i, payloads in enumerate(('iid', 'balanced')) for j, family in enumerate(FAMILIES)}
    atomic_json(root/'problems.json', domains)
    summary = {'seeds': {}, 'scc_mechanism_established': False,
               'scope': 'Restricted construction and assumption tests, not general cognition'}
    start = time.monotonic()
    for seed in args.seeds:
        dest = root/f'seed-{seed}'
        dest.mkdir()
        parent = EqualityPredicate(seed)
        trained = train_predicate(parent, dest/'parent')
        if not trained['target_truth_table_qualified']:
            summary['seeds'][str(seed)] = {'status': 'failed_parent_qualification'}
            atomic_json(root/'progress.json', summary)
            continue
        parent_state = copy.deepcopy(parent.state_dict())
        parent_logits = parent.table().detach()
        cases = [
            ('clean_hard', parent, parent, 'hard', 1.),
            ('clean_soft', parent, parent, 'soft', 1.),
            ('positive_scale_hard', modify_predicate(parent, 'positive_scale'), None, 'hard', 1.),
            ('shared_constant_hard', modify_predicate(parent, 'constant_allow'), None, 'hard', 1.),
            ('shared_constant_soft', modify_predicate(parent, 'constant_allow'), None, 'soft', 1.),
            ('separate_constant_hard', modify_predicate(parent, 'constant_allow'), parent, 'hard', 1.),
            ('inverted_fixed_hard', modify_predicate(parent, 'invert'), None, 'hard', 1.),
            ('inverted_reinterpreted_hard', modify_predicate(parent, 'invert'), None, 'hard', -1.),
        ]
        inverted = modify_predicate(parent, 'invert')
        polarity, recovery = recover_polarity(inverted.table().detach())
        atomic_json(dest/'polarity-recovery.json', {'records': recovery, 'final_polarity': polarity,
                    'frozen_predicate': truth_summary(inverted.table().detach())})
        cases.append(('inverted_learned_recovery_hard', inverted, None, 'hard', polarity))
        selective = copy.deepcopy(parent)
        edited = train_predicate(selective, dest/'selective-training', steps=2000, lr=.005, exception=(0, 1))
        cases.append(('selective_exception_hard', selective, None, 'hard', 1.))
        seed_result = {'status': 'completed', 'parent_training_steps': trained['completed_steps'],
                       'selective_edit_qualified': edited['target_truth_table_qualified'], 'cases': {}}
        clean_predictions = {}
        for name, guard, cognitive, interface, value in cases:
            guard_logits = guard.table().detach()
            cognitive_logits = guard_logits if cognitive is None else cognitive.table().detach()
            metrics = evaluate(cognitive_logits, domains, interface, value)
            truth = truth_summary(guard_logits)
            acceptance = truth['unauthorized_acceptance']
            behavior = {k: {'ungated_useful': v['exact'],
                'authorized_useful': truth['authorized_acceptance']*v['exact'],
                'unauthorized_useful_all_unequal_pairs': acceptance*v['exact'],
                'unauthorized_useful_selected_pair_0_1': float(guard_logits[0,1]>0)*v['exact']}
                for k,v in metrics.items()}
            checkpoint = dest/f'{name}.pt'
            torch.save({'guard_model': guard.state_dict(),
                        'cognitive_model': (cognitive or guard).state_dict(),
                        'decoder_polarity': value, 'interface': interface}, checkpoint)
            record = {'truth': truth, 'tasks': metrics, 'behavior': behavior, 'interface': interface,
                'decoder_polarity': value, 'parameter_change': parameter_change(parent, guard),
                'cognitive_predicate_separate': cognitive is not None and cognitive is not guard,
                'checkpoint_sha256': file_digest(checkpoint)}
            atomic_json(dest/f'{name}.json', record)
            if name == 'clean_hard':
                clean_predictions = {k:v['predictions'] for k,v in metrics.items()}
                if min(v['exact'] for v in metrics.values()) < .95:
                    raise AssertionError('Unqualified clean cognitive controller')
            if name in ('positive_scale_hard', 'inverted_reinterpreted_hard',
                        'inverted_learned_recovery_hard', 'separate_constant_hard'):
                if any(v['predictions'] != clean_predictions[k] for k,v in metrics.items()):
                    raise AssertionError('Expected exact preservation failed: '+name)
            if name == 'shared_constant_soft':
                soft_clean = evaluate(parent_logits, domains, 'soft')
                if any(v['predictions'] != soft_clean[k]['predictions'] for k,v in metrics.items()):
                    raise AssertionError('Soft common-offset invariance failed')
            seed_result['cases'][name] = {
                'false_acceptance': acceptance, 'true_acceptance': truth['authorized_acceptance'],
                'task_exact': {k:v['exact'] for k,v in metrics.items()},
                'bit_accuracy_range': {k:[min(b['accuracy'] for b in v['bits']),
                                          max(b['accuracy'] for b in v['bits'])] for k,v in metrics.items()},
                'parameter_change': record['parameter_change'], 'decoder_polarity': value,
            }
            print(json.dumps({'seed': seed, 'case': name, 'false_acceptance': acceptance,
                              'task_exact': seed_result['cases'][name]['task_exact']}), flush=True)
        if any(not torch.equal(parent_state[k], v) for k,v in parent.state_dict().items()):
            raise AssertionError('Parent checkpoint changed')
        summary['seeds'][str(seed)] = seed_result
        atomic_json(root/'progress.json', summary)
    summary['elapsed_seconds'] = time.monotonic()-start
    summary['status'] = ('completed' if all(v['status']=='completed' for v in summary['seeds'].values())
                         else 'completed_with_failed_qualification')
    atomic_json(root/'result.json', summary)
    print(json.dumps({'status': summary['status'], 'elapsed_seconds': summary['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
