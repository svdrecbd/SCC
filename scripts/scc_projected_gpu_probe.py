"""Short full-sized CUDA validation of the exact projected construction package."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from scc.developmental_run import configure, TextBank
from scc.portfolio_models import PortfolioModel, variant_config
from scc.projected_coupling import make_projected_episode, projected_objective
from scc.provenance import atomic_json, source_manifest


def inline_summary(result):
    """Keep full traces in artifacts; GMAN inline results are limited to32KiB."""
    summary = {k:result[k] for k in ('ok','source_files_verified','data_files_verified','elapsed_seconds','runtime_downloads','scope')}
    summary['checks'] = [{k:row[k] for k in ('variant','scope','gradient_norm','peak_cuda_memory_bytes')}
                         for row in result['checks']]
    summary['full_result_location'] = 'projected-check/result.json in the returned artifact'
    if len((json.dumps(summary, indent=2, sort_keys=True) + '\n').encode()) > 32 * 1024:
        raise ValueError('Compact probe result exceeds the provider inline limit')
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--parents', type=Path, required=True)
    p.add_argument('--device', choices=['cuda'], required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    device = configure(a.device, 4)
    assert json.loads(a.parents.read_text()) == {}
    expected = json.loads((ROOT / 'transport-source-manifest.json').read_text())
    for name, sha in expected.items():
        assert hashlib.file_digest((ROOT / name).open('rb'), 'sha256').hexdigest() == sha, name
    assert source_manifest() == {n:v for n,v in expected.items() if n.startswith('scc/') or n in ('pyproject.toml','uv.lock','.python-version')}
    data = json.loads((ROOT / 'transport-data-manifest.json').read_text())
    for name,sha in data.items():
        assert hashlib.file_digest((a.data / name).open('rb'),'sha256').hexdigest() == sha, name
    bank = TextBank(a.data, 2)
    results = []
    start = time.monotonic()
    for variant in ('standard','narrow32','tied'):
        for ordinal in (3,4):
            torch.manual_seed(23)
            model = PortfolioModel(variant_config(variant)).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
            torch.cuda.reset_peak_memory_stats()
            episode = make_projected_episode(bank, ordinal, device, batch_size=2, inner_steps=4)
            value, record = projected_objective(model, episode, geometry_weight=.1)
            value.backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            assert gradients and all(torch.isfinite(g).all() for g in gradients)
            norm = torch.stack([g.square().sum() for g in gradients]).sum().sqrt()
            assert torch.isfinite(value) and norm > 0
            optimizer.step()
            assert all(torch.isfinite(p).all() for p in model.parameters())
            results.append({'variant':variant,'scope':episode['configuration']['scope'],
                            'gradient_norm':float(norm),'record':record,
                            'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated()})
            atomic_json(a.output/'progress.json', results)
            del model,optimizer,value,gradients,norm
            gc.collect(); torch.cuda.empty_cache()
    result = {'ok':True,'checks':results,'source_files_verified':len(expected),'data_files_verified':len(data),
              'elapsed_seconds':time.monotonic()-start,'runtime_downloads':False,
              'scope':'Full-sized objective, full CUDA backward and optimizer step at initialization; not SCC or trained-model evidence'}
    atomic_json(a.output/'result.json',result)
    if os.environ.get('GMN_RESULT_PATH'): atomic_json(os.environ['GMN_RESULT_PATH'],inline_summary(result))
    print(json.dumps({'ok':True,'checks':len(results),'elapsed_seconds':result['elapsed_seconds']}),flush=True)


if __name__ == '__main__': main()
