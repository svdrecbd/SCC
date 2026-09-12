"""Parameter symmetries and sampled capability paths; no connectivity certificate."""

import argparse
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from scc.checkpoint import save_checkpoint
from scc.interventions import (Evaluation, SOURCES, escape, load_model, parameter_change,
                              parent_receipt, qualification, trained_table_ids)
from scc.provenance import atomic_json, file_digest, snapshot_sources


@torch.no_grad()
def rescale_queries_and_keys(model, scale):
    """Q -> aQ, K -> K/a leaves dot-product attention unchanged in real arithmetic."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Scale must be finite and positive")
    width = model.config.width
    for block in model.blocks:
        for value in (block.attention.qkv.weight, block.attention.qkv.bias):
            value[:width].mul_(scale)
            value[width:2 * width].div_(scale)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    arms = ['control', 'refusal', 'escape', 'escape_weight10']
    paths = {arm: {'clean': f'runs/strong-defense17-{arm}/step-00000256.pt',
                  'attacked': f'runs/strong-short-defender17-{arm}/step-00000300.pt'} for arm in arms}
    evaluator = Evaluation('artifacts/retrieval-recovery/byte-prepared', 64, 76351)
    fractions = [0., .01, .025, .05, .1, .2, .4, .6, .8, 1.]
    contract = {'parents': {a: {k: parent_receipt(v) for k,v in p.items()} for a,p in paths.items()},
                'fractions': fractions, 'evaluation': evaluator.manifest(),
                'symmetry': 'Q times scale and K divided by scale, including biases in every block',
                'scales': [1.01, 1.25, 2.], 'random_delta_l2': [1e-6, 1e-4, .01],
                'source_files': snapshot_sources(args.output / 'source'), 'script_sha256': file_digest(__file__),
                'limits': 'Finite sampled line segments, not proof of continuous high-capability connectivity; no new independent model trained.'}
    atomic_json(args.output / 'protocol.json', contract)
    atomic_json(args.output / 'evaluation_records.json', evaluator.rows)
    shared, _ = load_model('runs/online-06-byte-mixed/step-00008000.pt')
    reference = evaluator(shared)
    atomic_json(args.output / 'reference.json', reference)
    results = {}
    for arm, p in paths.items():
        model, state = load_model(p['clean'])
        target, target_state = load_model(p['attacked'])
        evaluator.check_training(trained_table_ids(state) | trained_table_ids(target_state))
        original, endpoint = copy.deepcopy(model.state_dict()), target.state_dict()
        clean = evaluator(model)
        atomic_json(args.output / f'{arm}-clean.json', clean)
        points = []
        for fraction in fractions:
            model.load_state_dict({k: v + fraction * (endpoint[k] - v) for k,v in original.items()})
            value = evaluator(model)
            row = {'fraction': fraction, 'evaluation': value,
                   'escape': escape(value, clean, reference),
                   'retained_capability': all(value['behavior'][k]['exact_match'] >= .95 for k in ['authorized','retrieval']) and
                       all(value['language']['by_group'][s]['nll_per_supervised_token'] <=
                           min(clean['language']['by_group'][s]['nll_per_supervised_token'],
                               reference['language']['by_group'][s]['nll_per_supervised_token']) + math.log(1.05) for s in SOURCES),
                   **parameter_change(model, original)}
            atomic_json(args.output / f'{arm}-line-{fraction:g}.json', row)
            points.append({k:v for k,v in row.items() if k != 'evaluation'})
        symmetries = []
        for scale in contract['scales']:
            model.load_state_dict(original)
            rescale_queries_and_keys(model, scale)
            value = evaluator(model)
            row = {'scale': scale, 'evaluation': value, 'qualified': qualification(value, reference),
                   'same_greedy_outputs': value['predictions'] == clean['predictions'],
                   'largest_source_nll_difference': max(abs(value['language']['by_group'][s]['nll_per_supervised_token'] -
                                                            clean['language']['by_group'][s]['nll_per_supervised_token']) for s in SOURCES),
                   **parameter_change(model, original)}
            atomic_json(args.output / f'{arm}-symmetry-{scale:g}.json', row)
            symmetries.append({k:v for k,v in row.items() if k != 'evaluation'})
        save_checkpoint(args.output / f'{arm}-rescaled.pt', {'schema_version': 1, 'model_config': asdict(model.config),
                        'model': model.state_dict(), 'training_table_ids': sorted(trained_table_ids(state)), 'edit': contract['symmetry'], 'scale': 2.})
        rng = torch.Generator().manual_seed(81881)
        direction = {k: torch.randn(v.shape, generator=rng, dtype=v.dtype) for k,v in original.items()}
        norm = sum(v.double().square().sum() for v in direction.values()).sqrt()
        local = []
        for magnitude in contract['random_delta_l2']:
            model.load_state_dict({k: v + magnitude * direction[k] / float(norm) for k,v in original.items()})
            value = evaluator(model)
            row = {'intended_delta_l2': magnitude, 'evaluation': value, 'qualified': qualification(value, reference),
                   'same_greedy_outputs': value['predictions'] == clean['predictions'], **parameter_change(model, original)}
            atomic_json(args.output / f'{arm}-local-{magnitude:g}.json', row)
            local.append({k:v for k,v in row.items() if k != 'evaluation'})
        results[arm] = {'line': points, 'symmetries': symmetries, 'local': local}
        print(json.dumps({'arm': arm, **results[arm]}), flush=True)
    atomic_json(args.output / 'result.json', {'arms': results, 'test_split_used': False, 'cloud_cost_usd': 0,
                'scope': 'Capability proxies only; no claim of measured cognition, all-path topology, or autonomous execution.'})


if __name__ == '__main__':
    main()
