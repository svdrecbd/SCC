"""Freeze LN-105 matched inputs and descriptive lookup-load/distance panels."""
import argparse
import gzip
import json
from pathlib import Path
import random
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scc.persistent_tasks import sample_request
from scc.provenance import atomic_json, file_digest
from scripts.localize_persistent_learning import independent_check


def diagnostic_panel(length, per_layout=128):
    rng = random.Random(17313031+length)
    serial = []
    for _ in range(per_layout):
        layouts = ['original','reordered']; rng.shuffle(layouts)
        for layout in layouts:
            row = sample_request(rng,('lookup','ungated',layout),split='test',active_length=length).record()
            row['active_length'] = length
            independent_check(row)
            serial.append(row)
    return [serial[i::4] for i in range(4)]


def prepare(parent, dest):
    dest.mkdir(parents=True,exist_ok=False)
    hashes = json.loads((parent/'manifest.json').read_text())
    assert len(hashes) == 17
    for name, expected in hashes.items():
        assert file_digest(parent/name) == expected, name
        target=dest/name; target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(parent/name,target)
        assert file_digest(target) == expected
    training = set()
    for pair in (1,2,3):
        with gzip.open(dest/f'pair-{pair}-data/pool.json.gz','rt') as stream:
            rows=json.load(stream)
        assert all(r['split'] == 'train' for r in rows)
        training.update(r['core_sha256'] for r in rows)
    metadata = {'scope':'Descriptive lookup-only panels with replacement; small support, no qualification gate.',
                'training_unique_cores':len(training),'panels':{}}
    for length in (2,4,8,12):
        rows=diagnostic_panel(length)
        cores={r['core_sha256'] for s in rows for r in s}
        assert not cores & training
        name=f'diagnostics/length-{length}.json'
        atomic_json(dest/name,rows); hashes[name]=file_digest(dest/name)
        metadata['panels'][str(length)]={'seed':17313031+length,'requests':256,
            'unique_cores':len(cores),'split':'test','physical_tokens':19,'sampling':'with replacement'}
    atomic_json(dest/'diagnostics/metadata.json',metadata)
    hashes['diagnostics/metadata.json']=file_digest(dest/'diagnostics/metadata.json')
    atomic_json(dest/'manifest.json',hashes)
    print(json.dumps(metadata))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();prepare(args.parent,args.output)
