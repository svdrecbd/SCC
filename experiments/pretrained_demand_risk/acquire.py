"""Acquire revision-pinned public artifacts without changing the environment."""
import hashlib
import json
import sys
from pathlib import Path
import requests

root = Path(sys.argv[1])
configuration = json.loads((root / 'config.json').read_text())
files = []
for name in ['config.json', 'generation_config.json', 'model.safetensors', 'README.md']:
    files.append((f"https://huggingface.co/amazon/chronos-t5-tiny/resolve/{configuration['model_revision']}/{name}", f'model/{name}'))
files.append((f"https://huggingface.co/datasets/autogluon/chronos_datasets/resolve/{configuration['dataset_revision']}/ercot/train-00000-of-00001.parquet", 'data/ercot.parquet'))
for name in ['chronos.py', 'base.py', 'utils.py', 'df_utils.py']:
    files.append((f"https://raw.githubusercontent.com/amazon-science/chronos-forecasting/{configuration['source_revision']}/src/chronos/{name}", f'vendor/chronos/{name}'))
for name in ['LICENSE', 'scripts/evaluation/configs/zero-shot.yaml']:
    files.append((f"https://raw.githubusercontent.com/amazon-science/chronos-forecasting/{configuration['source_revision']}/{name}", f'vendor/{Path(name).name}'))
records = []
total = 0
for url, relative in files:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    with destination.open('xb') as stream:
        for block in response.iter_content(65536):
            total += len(block)
            if total > configuration['download_limit_bytes']:
                raise RuntimeError('Download limit exceeded')
            stream.write(block)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    if relative == 'model/model.safetensors':
        assert digest == configuration['model_sha256']
    if relative == 'data/ercot.parquet':
        assert digest == configuration['dataset_sha256']
    records.append(dict(url=url, path=relative, bytes=destination.stat().st_size, sha256=digest))
(root / 'acquisition.json').write_text(json.dumps(records, indent=2) + '\n')
print(json.dumps({'downloaded_bytes': total, 'files': len(records)}))
