"""A short native-context GPU gate, with no source/data fetches at runtime."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn import functional as F
from scc.portfolio_models import PortfolioModel, VARIANTS, variant_config
from scc.provenance import source_manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--parents', type=Path, required=True)
    p.add_argument('--device', required=True)
    a = p.parse_args()
    assert a.device == 'cuda' and torch.cuda.is_available()
    assert json.loads(a.parents.read_text()) == {}
    a.output.mkdir(parents=True, exist_ok=False)
    expected = json.loads((ROOT / 'transport-source-manifest.json').read_text())
    for name, sha in expected.items():
        assert hashlib.file_digest((ROOT / name).open('rb'), 'sha256').hexdigest() == sha, name
    relevant = {n: sha for n, sha in expected.items() if n.startswith('scc/') or n in ('pyproject.toml', 'uv.lock', '.python-version')}
    assert source_manifest() == relevant
    data = json.loads((ROOT / 'transport-data-manifest.json').read_text())
    for name, sha in data.items():
        assert hashlib.file_digest((a.data / name).open('rb'), 'sha256').hexdigest() == sha, name
    torch.set_num_threads(4)
    checked = []
    models = [(name, lambda name=name: PortfolioModel(variant_config(name))) for name in VARIANTS]
    if (ROOT / 'scc/coordinate_models.py').exists():
        from scc.coordinate_models import CoordinateModel, DIMENSIONS, coordinate_config
        models = [(str(d), lambda d=d: CoordinateModel(coordinate_config(d))) for d in DIMENSIONS]
    for label, create in models:
        torch.manual_seed(23)
        model = create().cuda()
        tokens = torch.randint(0, 260, (2, 16), device='cuda')
        targets = torch.randint(0, 260, (2, 16), device='cuda')
        scores = model(tokens)
        loss = F.cross_entropy(scores.flatten(0, 1), targets.flatten())
        loss.backward()
        assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        checked.append({'label': label, 'parameters': model.parameter_count(), 'loss': float(loss.detach())})
        del model, scores, loss
        gc.collect()
        torch.cuda.empty_cache()
    torch.cuda.synchronize()
    result = {'ok': True, 'source_files_verified': len(expected), 'data_files_verified': len(data),
              'models': checked, 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'runtime_downloads': False, 'scope': 'Source/data presence and finite CUDA forward/backward; not SCC or full campaign completion'}
    (a.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    if os.environ.get('GMN_RESULT_PATH'):
        Path(os.environ['GMN_RESULT_PATH']).write_text(json.dumps(result))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
