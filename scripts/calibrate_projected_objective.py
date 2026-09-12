"""Full-sized developmental gradient and rerun checks; local, no GPU job polling."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scc.checkpoint import load_checkpoint
from scc.coupling import nll
from scc.developmental_run import configure, TextBank, Streams, environment
from scc.portfolio_models import PortfolioModel, variant_config, from_checkpoint
from scc.projected_coupling import make_projected_episode, projected_objective
from scc.provenance import atomic_json, snapshot_sources, source_manifest, file_digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--training', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = snapshot_sources(a.output / 'source')
    shutil.copyfile(__file__, a.output / 'runner.py')
    device = configure('cpu', 2)
    bank = TextBank(a.data, 2)
    episode = make_projected_episode(bank, 3, device, batch_size=2, inner_steps=4)
    ordinary = Streams(bank, 101, device, 2, .5).task('lookup', 'authorized')
    results = []
    started = time.monotonic()
    for stage in (0, 9000, 15000, 20000):
        torch.manual_seed(23)
        if stage == 0:
            model = PortfolioModel(variant_config('standard')).double()
            parent_path, parent_sha = None, None
        else:
            parent_path = a.training / ('final.pt' if stage == 20000 else f'stage-{stage}.pt')
            parent_sha = file_digest(parent_path)
            model = from_checkpoint(load_checkpoint(parent_path)).double()
        before = {n: p.detach().clone() for n,p in model.named_parameters()}
        parameters = tuple(model.parameters())
        ordinary_value = nll(model, dict(model.named_parameters()), ordinary)
        base_grads = torch.autograd.grad(ordinary_value, parameters)
        value, record = projected_objective(model, episode, geometry_weight=.1)
        gradients = torch.autograd.grad(value, parameters, allow_unused=True)
        squared = sum(g.detach().square().sum() for g in gradients if g is not None)
        norm = squared.sqrt()
        assert torch.isfinite(norm) and norm > 0
        base_norm = sum(g.detach().square().sum() for g in base_grads).sqrt()
        dot = sum((g.detach()*b.detach()).sum() for g,b in zip(gradients,base_grads) if g is not None)
        row = {'stage': stage, 'parent': str(parent_path) if parent_path else None, 'parent_sha256': parent_sha,
               'objective': record, 'meta_gradient_norm': float(norm), 'ordinary_gradient_norm': float(base_norm),
               'ordinary_meta_cosine': float(dot/(norm*base_norm)), 'finite_difference': []}
        if stage in (0, 20000):
            direction = [torch.zeros_like(p) if g is None else g.detach()/norm for p,g in zip(parameters,gradients)]
            predicted = float(norm)
            for radius in (1e-6,1e-7):
                values, branch_ids = [], []
                for sign in (1,-1):
                    with torch.no_grad():
                        for (n,parameter), d in zip(model.named_parameters(),direction): parameter.copy_(before[n]+sign*radius*d)
                    v, detail = projected_objective(model,episode,geometry_weight=.1,create_graph=False)
                    values.append(float(v.detach()))
                    branch_ids.append([(b['reader'],b['stop_rule']) for b in detail['reader_branches']])
                finite = (values[0]-values[1])/(2*radius)
                relative = abs(finite-predicted)/max(1e-8,abs(finite),abs(predicted))
                row['finite_difference'].append({'radius': radius, 'predicted': predicted, 'rerun': finite,
                                                  'relative_error': relative, 'reader_assignments_stable': branch_ids[0]==branch_ids[1]})
            assert all(x['relative_error'] < .005 and x['reader_assignments_stable'] for x in row['finite_difference'])
        with torch.no_grad():
            for n,parameter in model.named_parameters(): parameter.copy_(before[n])
        assert all(torch.equal(p,before[n]) for n,p in model.named_parameters())
        results.append(row)
        atomic_json(a.output / 'progress.json', results)
        print(json.dumps({'stage': stage, 'meta_gradient_norm': row['meta_gradient_norm'],
                          'ordinary_gradient_norm': row['ordinary_gradient_norm'], 'finite_difference': row['finite_difference']}), flush=True)
        del gradients, model, parameters, base_grads, value
    assert source_manifest() == source
    atomic_json(a.output / 'result.json', {'status': 'complete', 'results': results, 'episode': episode['record'],
                'environment': {**environment(device),'precision':'fp64'}, 'source_unchanged': True,
                'elapsed_seconds': time.monotonic()-started,
                'evidence_class': 'Full-sized numerical/developmental signal calibration on ordinary models; no trained projected defender or SCC result'})


if __name__ == '__main__': main()
