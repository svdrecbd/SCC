"""Local finite-step calibration on a saved qualified checkpoint; no job polling."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scc.checkpoint import load_checkpoint, save_checkpoint
from scc.coupling import nll
from scc.developmental_run import configure, TextBank, environment
from scc.portfolio_models import from_checkpoint
from scc.projected_coupling import make_projected_episode
from scc.projected_edit import local_geometry, geometry_record, apply_direction, observables
from scc.functional_state import call_parameters
from scc.provenance import atomic_json, file_digest, snapshot_sources, source_manifest
from scc.transition_path import panels, point


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    device = configure('cpu', 2)
    source = snapshot_sources(a.output / 'source')
    shutil.copyfile(__file__, a.output / 'runner.py')
    state = load_checkpoint(a.checkpoint)
    parent = from_checkpoint(state).eval()
    original = {n: p.detach().clone() for n, p in parent.named_parameters()}
    bank = TextBank(a.data, 4)
    episode = make_projected_episode(bank, 3, device, batch_size=2, inner_steps=1)
    batch = episode['changes'][0]
    panel = panels(8, 16, seed=884291)
    atomic_json(a.output / 'panels.json', panel)
    atomic_json(a.output / 'episode.json', episode['record'])
    baseline = point(parent, bank, panel, 0, text_blocks=4)
    atomic_json(a.output / 'baseline.json', baseline)
    records = []
    for scope in ('core', 'all'):
        geometry = local_geometry(parent, dict(parent.named_parameters()), batch['target'],
                                  batch['capabilities'], scope=scope)
        record = geometry_record(geometry)
        atomic_json(a.output / (scope + '-geometry.json'), record)
        save_checkpoint(a.output / (scope + '-directions.pt'), {
            'names': geometry['names'], 'gradient': geometry['gradient'].detach(),
            'projected': geometry['projected'].detach(), 'parent_sha256': file_digest(a.checkpoint)})
        for method, direction in [('gradient', geometry['gradient']), ('projected', geometry['projected'])]:
            for radius in (.05, .2, .5):
                changed = apply_direction(dict(parent.named_parameters()), geometry['names'], direction, radius)
                with torch.no_grad():
                    after_nll = float(nll(parent, changed, batch['target']))
                    after_observations = torch.cat([observables(call_parameters(parent, changed, (b.tokens,), strict=True), b, domain)
                                                    for domain,b in batch['capabilities'].items()])
                moved = copy.deepcopy(parent)
                with torch.no_grad():
                    for name, parameter in moved.named_parameters(): parameter.copy_(changed[name])
                measured = point(moved, bank, panel, 1, baseline, 4)
                label = f'{scope}-{method}-r{radius}'
                atomic_json(a.output / (label + '-predictions.json'), measured)
                row = {'scope': scope, 'method': method, 'radius': radius, 'support_selected_nll': after_nll,
                       'support_selected_nll_before': record['attack_nll'],
                       'maximum_constraint_change': float((after_observations - geometry['observables'].detach()).abs().max()),
                       'constraint_changes': (after_observations - geometry['observables'].detach()).tolist(),
                       'minimum_benign_strict': measured['minimum_benign_strict'],
                       'joint_reliable_violation': measured['joint_reliable_violation'],
                       'all_domains_low_screen': measured['all_domains_low_screen'],
                       'target_accuracy': {n: v['payload_accuracy'] for n,v in measured['targets'].items()},
                       'text_gain_retention': measured['text_gain_retention']}
                records.append(row)
                print(json.dumps(row), flush=True)
                atomic_json(a.output / 'progress.json', {'records': records})
                del changed, moved
    assert all(torch.equal(p, original[n]) for n, p in parent.named_parameters())
    assert source_manifest() == source
    result = {'status': 'complete', 'records': records, 'parent': str(a.checkpoint),
              'parent_sha256': file_digest(a.checkpoint), 'source_unchanged': True,
              'parent_unchanged': True, 'environment': environment(device),
              'elapsed_seconds': time.monotonic() - started,
              'evidence_class': 'Open local finite-step geometry calibration on an ordinary-trained checkpoint; not SCC or a coupled-model result'}
    atomic_json(a.output / 'result.json', result)


if __name__ == '__main__':
    main()
