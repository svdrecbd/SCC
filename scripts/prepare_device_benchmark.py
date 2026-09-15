"""Freeze a compact set of existing long-load windows; generate no new data."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scc.provenance import atomic_json, file_digest
from scripts.train_separated_binding import tensors


def prepare(parent, output):
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((parent/'artifact-manifest.json').read_text())
    names = ['inputs/repair-origin.pt', 'inputs/pair-1-data/pool.json.gz',
             'inputs/pair-1-data/schedule.pt', 'pair-1-hidden_only/update-12000.pt']
    hashes = {}
    for name in names:
        hashes[name] = file_digest(parent/name)
        assert hashes[name] == manifest[name], name
    ids, labels, target, schedule = tensors(parent/'inputs', 1)
    chosen = schedule[2000:2080].long()
    origin = torch.load(parent/'inputs/repair-origin.pt', weights_only=True)
    repaired = torch.load(parent/'pair-1-hidden_only/update-12000.pt', weights_only=True)['payload']
    torch.save({'origin': origin, 'repaired': repaired, 'ids': ids[chosen],
                'labels': labels[chosen], 'target': target[chosen]}, output/'windows.pt')
    atomic_json(output/'manifest.json', {'windows.pt': file_digest(output/'windows.pt')})
    atomic_json(output/'parents.json', {'parent': str(parent.resolve()), 'hashes': hashes,
                'schedule_updates': [2001,2080], 'new_training_data': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); prepare(a.parent, a.output)
