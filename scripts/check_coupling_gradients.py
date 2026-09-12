"""Float64 directional finite differences on the actual qualified model and meta loss."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from scc.coupling import episode_loss, task_batch
from scc.data import PreparedDataset
from scc.interventions import Streams, batch_digest, load_model, parent_receipt
from scc.provenance import atomic_json, file_digest, source_manifest
from scc.tokenizer import ByteTokenizer

torch.set_num_threads(4)
parser = argparse.ArgumentParser()
parser.add_argument("--output", default="artifacts/coupling-discovery/full-model-gradient-check.json")
parser.add_argument("--epsilons", nargs="+", type=float, default=[1e-4, 1e-5])
args = parser.parse_args()
output = ROOT / args.output
if output.exists():
    raise FileExistsError(output)
checkpoint = ROOT / "runs/online-06-byte-mixed/step-00008000.pt"
model, _ = load_model(checkpoint)
model.double()
normalizer = json.loads((checkpoint.parent / "result.json").read_text())["language_after"]["nll_per_supervised_token"]
streams = Streams(PreparedDataset(ROOT / "artifacts/retrieval-recovery/byte-prepared", "train"), 5016)
episodes = [[(streams.task(8, "unauthorized", "disclose"), 1),
             (streams.task(8, "retrieval"), .5),
             (streams.text(8), .5 / normalizer)] for _ in range(3)]
rows = streams.tables.batch(8, "unauthorized")
query = {"disclose": task_batch(rows, ByteTokenizer(), 192, "disclose"),
         "policy": task_batch(rows, ByteTokenizer(), 192),
         "retrieval": streams.task(8, "retrieval"), "language": streams.text(8)}
config = {"mode": "escape", "inner_learning_rate": .001, "language_normalizer": normalizer,
          "surrogate": {"break_threshold": .5, "cap_threshold": .75, "break_temperature": 2., "cap_temperature": .25}}
parameters = dict(model.named_parameters())
with sdpa_kernel(SDPBackend.MATH):
    value, _ = episode_loss(model, parameters, episodes, query, config)
    gradients = torch.autograd.grad(value, tuple(parameters.values()))
gradient_norm = sum(g.square().sum() for g in gradients).sqrt()
rng = torch.Generator().manual_seed(641)
random_direction = {k: torch.randn(v.shape, generator=rng, dtype=v.dtype) for k, v in parameters.items()}
random_norm = sum(v.square().sum() for v in random_direction.values()).sqrt()
directions = {"random": {k: v / random_norm for k, v in random_direction.items()},
              "gradient": {k: g.detach() / gradient_norm for k, g in zip(parameters, gradients)}}
checks = []
for name, direction in directions.items():
    analytic = float(sum((g * direction[k]).sum() for k, g in zip(parameters, gradients)))
    for epsilon in args.epsilons:
        values = []
        for sign in [1, -1]:
            perturbed = {k: (v.detach() + sign * epsilon * direction[k]).requires_grad_() for k, v in parameters.items()}
            with sdpa_kernel(SDPBackend.MATH):
                result, _ = episode_loss(model, perturbed, episodes, query, config)
            values.append(float(result.detach()))
        finite = (values[0] - values[1]) / (2 * epsilon)
        absolute = abs(finite - analytic)
        relative = absolute / max(abs(finite), abs(analytic), 1e-12)
        checks.append({"direction": name, "epsilon": epsilon, "analytic": analytic, "finite_difference": finite,
                       "absolute_error": absolute, "relative_error": relative,
                       "passed": absolute <= 1e-6 + 1e-3 * abs(analytic)})
        print(json.dumps(checks[-1]), flush=True)
atomic_json(output, {"passed": all(c["passed"] for c in checks), "checks": checks,
            "gradient_l2": float(gradient_norm), "objective": config, "dtype": "float64", "inner_steps": 3,
            "parameter_count": model.parameter_count(), "parent": parent_receipt(checkpoint),
            "source_files": source_manifest(), "script_sha256": file_digest(__file__),
            "batch_hashes": [batch_digest(b) for episode in episodes for b, _ in episode] + [batch_digest(b) for b in query.values()],
            "partition": "train"})
if not all(c["passed"] for c in checks):
    raise RuntimeError("Full-model meta-gradient check failed")
