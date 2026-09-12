"""Training-partition-only calibration of the short inner attack; no fitting."""

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from scc.coupling import Meter, episode_loss, task_batch
from scc.data import PreparedDataset
from scc.interventions import Streams, load_model, parent_receipt
from scc.provenance import atomic_json, file_digest, source_manifest
from scc.tokenizer import ByteTokenizer

torch.set_num_threads(4)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--learning-rates", type=float, nargs="+", default=[.0001, .0003, .001, .003])
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--break-temperature", type=float, default=.5)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
checkpoint = ROOT / "runs/online-06-byte-mixed/step-00008000.pt"
model, _ = load_model(checkpoint)
normalizer = json.loads((checkpoint.parent / "result.json").read_text())["language_after"]["nll_per_supervised_token"]
natural = PreparedDataset(ROOT / "artifacts/retrieval-recovery/byte-prepared", "train")
results = []
for learning_rate in args.learning_rates:
    streams = Streams(natural, 5016)
    episodes = [[(streams.task(8, "unauthorized", "disclose", "inner/disclose"), 1),
                 (streams.task(8, "retrieval", role="inner/retrieval"), .5),
                 (streams.text(8, "inner/language"), .5 / normalizer)] for _ in range(3)]
    rows = streams.tables.batch(8, "unauthorized")
    query = {"disclose": task_batch(rows, ByteTokenizer(), 192, "disclose", "query/disclose"),
             "policy": task_batch(rows, ByteTokenizer(), 192, "policy", "query/policy"),
             "retrieval": streams.task(8, "retrieval", role="query/retrieval"),
             "language": streams.text(8, "query/language")}
    config = {"mode": "escape", "inner_learning_rate": learning_rate, "language_normalizer": normalizer,
              "surrogate": {"break_threshold": .5, "cap_threshold": .75, "break_temperature": args.break_temperature, "cap_temperature": .25}}
    meter = Meter()
    started = time.perf_counter()
    with sdpa_kernel(SDPBackend.MATH):
        penalty, metrics = episode_loss(model, dict(model.named_parameters()), episodes, query, config, meter)
        gradients = torch.autograd.grad(penalty, tuple(model.parameters()))
    norm = sum(float(g.square().sum()) for g in gradients) ** .5
    results.append({"configuration": config, "inner_learning_rate": learning_rate, **metrics, "meta_gradient_l2": norm,
                    "seconds": time.perf_counter() - started, "meter": asdict(meter)})
    print(json.dumps(results[-1]), flush=True)
atomic_json(args.output, {"parent": parent_receipt(checkpoint), "source_files": source_manifest(), "script_sha256": file_digest(__file__),
                    "data_seed": 5016, "partition": "train", "results": results})
