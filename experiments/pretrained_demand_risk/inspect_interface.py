"""Validate a public model interface on a synthetic input, without training."""
import json
import os
import platform
import sys
import time
import types
from pathlib import Path
import numpy as np
import pyarrow.parquet as parquet
import torch
import transformers

root = Path(sys.argv[1])
parent = Path(sys.argv[2]) if len(sys.argv) > 2 else root
package = types.ModuleType('chronos')
package.__path__ = [str(parent / 'vendor/chronos')]
sys.modules['chronos'] = package
from chronos.chronos import ChronosPipeline, MeanScaleUniformBins
package.MeanScaleUniformBins = MeanScaleUniformBins

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.manual_seed(26600)
started = time.perf_counter()
pipeline = ChronosPipeline.from_pretrained(str(parent / 'model'), device_map='cpu', dtype=torch.float32, local_files_only=True)
load_seconds = time.perf_counter() - started
context = torch.tensor(100 + 10 * np.sin(np.arange(512) * 2 * np.pi / 24), dtype=torch.float32)
started = time.perf_counter()
with torch.inference_mode():
    predictions = pipeline.predict(context, prediction_length=24, num_samples=64)
query_seconds = time.perf_counter() - started
assert predictions.shape == (1, 64, 24) and torch.isfinite(predictions).all()
frame = parquet.read_table(parent / 'data/ercot.parquet').to_pandas()
metadata = []
for _, row in frame.iterrows():
    values = np.asarray(row['target'])
    metadata.append({'item_id': str(row.get('item_id')), 'length': len(values), 'finite': bool(np.isfinite(values).all()), 'start': str(row.get('start'))})
result = dict(python=platform.python_version(), torch=torch.__version__, transformers=transformers.__version__, platform=platform.platform(), affinity=list(os.sched_getaffinity(0)), parameters=sum(p.numel() for p in pipeline.model.parameters()), load_seconds=load_seconds, synthetic_query_seconds=query_seconds, columns=list(frame.columns), series=metadata)
(root / 'inspection.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
