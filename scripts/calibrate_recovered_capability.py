"""Calibrate the SCC ranking/recovery candidate on preserved neural checkpoints."""

import argparse
import copy
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel

from scc.coupling import Batch, nll
from scc.data import IGNORE
from scc.developmental_run import TextBank, configure, evaluate, environment
from scc.developmental_tasks import FAMILIES, CATEGORIES, TaskStream, batch_rows, evaluation_rows
from scc.differentiable_modify import adam_unroll
from scc.gradient_diagnostics import cosine, norm
from scc.model import ModelConfig, Transformer
from scc.provenance import atomic_json, digest, file_digest, snapshot_sources
from scc.recovered_capability import (Reader, branch_objective, fit_readers,
                                      model_objective, recovery_envelope, sequence_scores)

ROOT = Path(__file__).resolve().parents[1]
ORDINARY = ROOT/'artifacts/developmental-gpu-20260910-v5/experiment-downloaded/development/comparison/rule_only'
CONTINUATION = ROOT/'artifacts/scc-diagnostics-20260910-v1/continuation/downloaded/full-gradient-continuation'


def emit(value):
    print(json.dumps(value), flush=True)


def detached(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach()) if value.numel() == 1 else value.detach().tolist()
    if isinstance(value, dict):
        return {k: detached(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [detached(v) for v in value]
    return value


class Viewed(torch.nn.Module):
    def __init__(self, model, readout=Reader(), fixture=Reader()):
        super().__init__()
        self.model, self.readout, self.fixture = model, readout, fixture

    def forward(self, tokens):
        return self.readout.apply(self.fixture.apply(self.model(tokens)))


def support_rows():
    stream = TaskStream(483920)
    rows = []
    for family in FAMILIES:
        for category in CATEGORIES:
            rows += stream.rows(8, family, category)
    return rows


def batches(rows, bank, small=False):
    result = {}
    for family in FAMILIES:
        for category in CATEGORIES:
            chosen = [r for r in rows if r['family'] == family and r['category'] == category]
            if small:
                chosen = chosen[:1]
            # Complete answers include EOS; a missing terminator is not success.
            result[family+'/'+category] = batch_rows(chosen, disclose=category == 'unauthorized', content_only=False)
    for source, indices in bank.eval_indices.items():
        selected = indices[:1] if small else indices
        result[source] = Batch(*bank.validation.batch(selected, 'cpu'), 'text/'+source)
    return result


@torch.no_grad()
def get_readers(model, rows, fixture=Reader()):
    # All fitting is confined to train-split digit answers, including disclosure.
    pairs = []
    for category in CATEGORIES:
        subset = [r for r in rows if r['category'] == category]
        batch = batch_rows(subset, disclose=category == 'unauthorized', content_only=True)
        z = fixture.apply(model(batch.tokens))
        mask = batch.targets != IGNORE
        pairs.append((z[mask], batch.targets[mask]))
    logits = torch.cat([z for z, _ in pairs]).reshape(1, -1, 260)
    targets = torch.cat([y for _, y in pairs]).reshape(1, -1)
    return fit_readers(logits, targets), logits, targets


@torch.no_grad()
def inspect_endpoint(model, rows, fit_rows, bank, queries, fixture=Reader()):
    readers, support_logits, support_targets = get_readers(model, fit_rows, fixture)
    # Select only on support; keep identity and all declared readers in records.
    support_scores = [float(sequence_scores(r.apply(support_logits), support_targets).mean()) for r in readers]
    selected = max(range(len(readers)), key=lambda i: support_scores[i])
    cap = {name: (fixture.apply(model(b.tokens)), b.targets) for name, b in queries.items()}
    value, branches = branch_objective(cap, {k: v for k, v in cap.items() if k.endswith('/unauthorized')}, readers)
    evaluations = []
    for index, reader in enumerate(readers):
        view = Viewed(model, reader, fixture).eval()
        metrics = evaluate(view, bank, rows)
        prefix_tasks = {}
        for family in FAMILIES:
            for category in CATEGORIES:
                pairs = [(r, p) for r, p in zip(rows, metrics['predictions'])
                         if r['family'] == family and r['category'] == category]
                interpreted = []
                for row, prediction in pairs:
                    text = prediction['text']
                    valid = len(text) >= 4 and all(c in '0123456789' for c in text[:4])
                    interpreted.append((row, text[:4] if valid else text, valid or prediction['terminated']))
                prefix_tasks[family+'/'+category] = {
                    'n': len(pairs),
                    'policy_exact': sum(ok and text == row['target'] for row, text, ok in interpreted)/len(pairs),
                    'useful_answer_exact': sum(ok and text == row['underlying_answer'] for row, text, ok in interpreted)/len(pairs)}
        evaluations.append({'reader': reader.record(), 'support_score': support_scores[index],
                            'selected_on_support': index == selected, 'evaluation': metrics,
                            'four_digit_interpretation': prefix_tasks})
    return {'objective': float(value), 'branches': detached(branches), 'evaluations': evaluations,
            'selected_reader': readers[selected].record()}, readers[selected]


def load_model(path, model_config):
    model = Transformer(ModelConfig(**model_config)).eval()
    state = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'])
    return model


def repair(model, reader, bank, excluded):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(.9, .95), eps=1e-4,
                                  weight_decay=0., foreach=False)
    stream = TaskStream(830244)
    stream.excluded = set(excluded)
    rng = random.Random(90224)
    records = []
    for step in range(16):
        if step % 4 == 3:
            source = tuple(bank.indices)[(step//4) % 4]
            batch = bank.batch(source, 8, rng, 'cpu')
        else:
            category = 'unauthorized' if step % 2 == 0 else ('ungated' if step % 3 else 'authorized')
            rows = stream.rows(8, FAMILIES[step % 3], category)
            batch = batch_rows(rows, disclose=category == 'unauthorized')
        optimizer.zero_grad(set_to_none=True)
        logits = reader.apply(model(batch.tokens))
        loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), batch.targets.flatten(), ignore_index=IGNORE)
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        records.append({'step': step+1, 'role': batch.role, 'loss': float(loss.detach()), 'gradient_norm': float(grad)})
    return {'steps': records, 'train_latent_ids': sorted(stream.seen), 'optimizer': 'fresh full-parameter AdamW',
            'support_reader_fixed_during_repair': reader.record(), 'clean_parent_weights_used_for_repair': False}


def signal(model, queries, removal, replay):
    parameters = dict(model.named_parameters())
    values = tuple(parameters.values())
    unauthorized = [k for k in queries if k.endswith('/unauthorized')]
    readers = [Reader(), Reader(-1)]
    logits = {k: (functional_call(model, parameters, (b.tokens,), strict=True), b.targets) for k, b in queries.items()}
    objective, branches = branch_objective(logits, {k: logits[k] for k in unauthorized}, readers)
    domains = {}
    for name, (z, y) in logits.items():
        score = sequence_scores(z, y).mean()
        gradient = torch.autograd.grad(score, values, retain_graph=True)
        loss = torch.nn.functional.cross_entropy(z.flatten(0, 1), y.flatten(), ignore_index=IGNORE)
        domains[name] = {'value': float(score.detach()), 'gradient_norm': float(norm(gradient)), 'raw_nll': float(loss.detach())}
    ordinary_gradient = torch.autograd.grad(nll(model, parameters, replay), values)
    gradient = torch.autograd.grad(objective, values)
    direct = {'objective': float(objective.detach()), 'gradient_norm': float(norm(gradient)),
              'ordinary_gradient_norm': float(norm(ordinary_gradient)),
              'cosine_with_ordinary': cosine(gradient, ordinary_gradient), 'domains': domains,
              'branches': detached(branches)}
    del logits, branches, gradient, ordinary_gradient, objective

    def meta(p, create_graph):
        altered = adam_unroll(p, [lambda q: nll(model, q, removal)], lr=.001, eps=1e-4, create_graph=create_graph)
        repaired = adam_unroll(altered, [lambda q: nll(model, q, replay)], lr=1e-4, eps=1e-4, create_graph=create_graph)
        raw = model_objective(model, altered, queries, unauthorized, readers)
        recovered = model_objective(model, repaired, queries, unauthorized, readers)
        return recovery_envelope([raw, recovered]), [raw[0], recovered[0]]

    with sdpa_kernel(SDPBackend.MATH):
        value, endpoint_values = meta(parameters, True)
        gradient = torch.autograd.grad(value, values)
        length = float(norm(gradient))
        result = {'value': float(value.detach()), 'gradient_norm': length,
                  'endpoint_values': [float(v.detach()) for v in endpoint_values],
                  'derivative': 'full one-step smooth Adam removal plus one-step repair; fixed identity/sign readers',
                  'finite_differences': []}
        if length > 0 and math.isfinite(length):
            direction = {k: g.detach()/length for k, g in zip(parameters, gradient)}
            for epsilon in (.0001, .001):
                outcomes = {}
                for sign in (-1, 1):
                    shifted = {k: (p.detach()+sign*epsilon*direction[k]).requires_grad_() for k, p in parameters.items()}
                    shifted_value, shifted_endpoints = meta(shifted, False)
                    outcomes[str(sign)] = {'value': float(shifted_value.detach()),
                                           'endpoint_values': [float(v.detach()) for v in shifted_endpoints]}
                    del shifted_value, shifted_endpoints, shifted
                numerical = (outcomes['1']['value']-outcomes['-1']['value'])/(2*epsilon)
                result['finite_differences'].append({'epsilon': epsilon, 'outcomes': outcomes,
                    'numerical': numerical, 'autograd': length, 'relative_error': abs(numerical-length)/length,
                    'descent_reduced_objective': outcomes['-1']['value'] < result['value']})
        return {'direct': direct, 'through_modification_and_repair': result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('controls', 'signal'), required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    start = time.monotonic()
    configure('cpu', threads=2)
    torch.manual_seed(17)
    source = snapshot_sources(args.output/'source')
    shutil.copyfile(__file__, args.output/'runner.py')
    protocol = ROOT/'protocols/SCC_RECOVERED_CAPABILITY_CALIBRATION_V3.md'
    shutil.copyfile(protocol, args.output/'protocol.md')
    bank = TextBank(ROOT/'artifacts/retrieval-recovery/byte-prepared', blocks=4)
    config = json.loads((ORDINARY/'contract.json').read_text())['configuration']['model']
    fit_rows = support_rows()
    rows = evaluation_rows(16, seed=939115) + evaluation_rows(16, seed=939115, reordered=True)
    fit_ids, query_ids = {r['latent_id'] for r in fit_rows}, {r['latent_id'] for r in rows}
    assert not fit_ids & query_ids
    queries = batches(rows, bank, small=args.mode == 'signal')
    contract = {'source': source, 'runner_sha256': file_digest(__file__), 'protocol_sha256': file_digest(protocol),
                'environment': environment('cpu'), 'text': bank.manifest(),
                'support_query_ids_disjoint': True, 'support_rows': fit_rows, 'query_rows': rows,
                'query_batches': {k: {'tokens': b.tokens.tolist(), 'targets': b.targets.tolist()} for k, b in queries.items()},
                'mode': args.mode, 'model': config, 'new_cloud_compute_usd': 0}
    atomic_json(args.output/'contract.json', contract)
    checksums, summary = {}, {}
    def check_budget():
        if time.monotonic()-start > 1200:
            raise RuntimeError('20-minute calibration wall budget exhausted')
        if sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) > 100*1024**2:
            raise RuntimeError('100 MiB output budget exhausted')
    if args.mode == 'controls':
        path = CONTINUATION/'probes-full_10/stage-2.pt'
        checksums[str(path.relative_to(ROOT))] = file_digest(path)
        baseline = load_model(path, config)
        for name in ('identity', 'scale_0001', 'scale_01', 'scale_1000', 'sign', 'digit_cycle', 'affine_noise'):
            check_budget()
            model = copy.deepcopy(baseline)
            fixture = Reader()
            with torch.no_grad():
                scale = {'scale_0001': .001, 'scale_01': .1, 'scale_1000': 1000., 'sign': -1.}.get(name, 1.)
                model.norm.weight.mul_(scale); model.norm.bias.mul_(scale)
                if name == 'affine_noise':
                    generator = torch.Generator().manual_seed(19035)
                    model.norm.weight.add_(.5*torch.randn(model.norm.weight.shape, generator=generator))
                    model.norm.bias.add_(.5*torch.randn(model.norm.bias.shape, generator=generator))
                elif name == 'digit_cycle':
                    fixture = Reader(1, tuple((i+3) % 10 for i in range(10)))
            before, selected = inspect_endpoint(model, rows, fit_rows, bank, queries, fixture)
            record = {'fixture': name, 'before': before}
            if name == 'affine_noise':
                record['repair'] = repair(model, selected, bank, fit_ids | query_ids)
                assert not set(record['repair']['train_latent_ids']) & (fit_ids | query_ids)
                after, _ = inspect_endpoint(model, rows, fit_rows, bank, queries, fixture)
                record['after_repair'] = after
            atomic_json(args.output/(name+'.json'), record)
            summary[name] = {'objective': before['objective'], 'readers': len(before['evaluations'])}
            emit({'mode': args.mode, 'fixture': name, **summary[name], 'elapsed_seconds': time.monotonic()-start})
            del model, record, before
    else:
        support = TaskStream(853921)
        support.excluded = fit_ids | query_ids
        removal = batch_rows(support.rows(2, 'lookup', 'unauthorized'), disclose=True)
        replay = batch_rows(support.rows(2, 'lookup', 'ungated'))
        atomic_json(args.output/'meta_support.json', {'removal': {'tokens': removal.tokens.tolist(), 'targets': removal.targets.tolist()},
            'replay': {'tokens': replay.tokens.tolist(), 'targets': replay.targets.tolist()}, 'train_ids': sorted(support.seen)})
        stages = [('initialization', None)] + [(f'ordinary_{n}', ORDINARY/f'step-{n:08d}.pt') for n in (1000,5000,9000,18000)]
        stages += [('full_gradient_candidate', CONTINUATION/'full_10/step-00004000.pt')]
        for name, path in stages:
            check_budget()
            if path:
                checksums[str(path.relative_to(ROOT))] = file_digest(path)
                model = load_model(path, config)
            else:
                torch.manual_seed(17)
                model = Transformer(ModelConfig(**config)).eval()
            before = {k: p.detach().clone() for k, p in model.named_parameters()}
            record = signal(model, queries, removal, replay)
            for key, value in record['direct']['domains'].items():
                floor = bank.floors[key] if key in bank.floors else math.log(10)
                value['legacy_residual_active'] = value['raw_nll'] < floor
            assert all(torch.equal(p, before[k]) for k, p in model.named_parameters())
            atomic_json(args.output/(name+'.json'), record)
            summary[name] = {'direct_gradient_norm': record['direct']['gradient_norm'],
                'meta_gradient_norm': record['through_modification_and_repair']['gradient_norm'],
                'legacy_active_domains': sum(v['legacy_residual_active'] for v in record['direct']['domains'].values())}
            emit({'mode': args.mode, 'stage': name, **summary[name], 'elapsed_seconds': time.monotonic()-start})
            del model, record, before
    for path, expected in checksums.items():
        assert file_digest(ROOT/path) == expected
    atomic_json(args.output/'summary.json', {'status': 'calibration_complete', 'mode': args.mode,
        'summary': summary, 'checkpoint_hashes_unchanged': checksums, 'elapsed_seconds': time.monotonic()-start,
        'mechanism_established': False, 'new_model_files_written': 0, 'new_cloud_compute_usd': 0})
    emit({'status': 'complete', 'mode': args.mode})


if __name__ == '__main__':
    main()
