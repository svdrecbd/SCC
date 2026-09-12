"""Validate the frozen-displacement gradient on the real model and report alignment."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from scc.coupling import nll, post_attack_loss, task_batch
from scc.data import PreparedDataset
from scc.interventions import Streams, load_model, parent_receipt
from scc.provenance import atomic_json, file_digest, source_manifest
from scc.strong_attack import straight_through_parameters
from scc.tokenizer import ByteTokenizer

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
torch.set_num_threads(4)
parent_path = ROOT / "runs/online-06-byte-mixed/step-00008000.pt"
attack_path = ROOT / "artifacts/strong-attack/calibration17/seed-10404-endpoint.pt"
model, _ = load_model(parent_path)
attacked, _ = load_model(attack_path)
model.double(); attacked.double()
clean = json.loads((attack_path.parent / "clean.json").read_text())
normalizer = clean["language"]["nll_per_supervised_token"]
streams = Streams(PreparedDataset(ROOT / "artifacts/retrieval-recovery/byte-prepared", "train"), 25816)
rows = streams.tables.batch(32, "unauthorized")
query = {"disclose": task_batch(rows, ByteTokenizer(), 192, "disclose"),
         "policy": task_batch(rows, ByteTokenizer(), 192), "retrieval": streams.task(32, "retrieval"),
         "language": streams.text(32)}
config = {"mode": "escape", "language_normalizer": normalizer,
          "surrogate": {"break_threshold": .5, "cap_threshold": .75, "break_temperature": 2., "cap_temperature": .25}}
parameters, target = dict(model.named_parameters()), dict(attacked.named_parameters())
delta = {k: target[k].detach() - p.detach() for k, p in parameters.items()}
with sdpa_kernel(SDPBackend.MATH):
    value, diagnostics = post_attack_loss(model, straight_through_parameters(parameters, target), query, config)
    gradients = torch.autograd.grad(value, tuple(parameters.values()))
    ordinary = torch.autograd.grad(nll(model, parameters, query["language"]), tuple(parameters.values()))
norm = sum(g.square().sum() for g in gradients).sqrt()
ordinary_norm = sum(g.square().sum() for g in ordinary).sqrt()
cosine = sum((a*b).sum() for a,b in zip(gradients, ordinary)) / (norm * ordinary_norm)
rng = torch.Generator().manual_seed(4641)
random = {k: torch.randn(p.shape, dtype=p.dtype, generator=rng) for k, p in parameters.items()}
random_norm = sum(g.square().sum() for g in random.values()).sqrt()
directions = {"random": {k: g/random_norm for k,g in random.items()},
              "gradient": {k: g.detach()/norm for k,g in zip(parameters,gradients)}}
checks = []
for name, direction in directions.items():
    analytic = float(sum((g * direction[k]).sum() for k, g in zip(parameters, gradients)))
    for epsilon in [1e-4, 1e-5, 1e-6]:
        values = []
        for sign in [1, -1]:
            moved = {k: p.detach() + delta[k] + sign * epsilon * direction[k] for k,p in parameters.items()}
            with sdpa_kernel(SDPBackend.MATH):
                result, _ = post_attack_loss(model, moved, query, config)
            values.append(float(result))
        finite = (values[0] - values[1]) / (2 * epsilon)
        error = abs(finite-analytic)
        check = {"direction": name, "epsilon": epsilon, "analytic": analytic, "finite_difference": finite,
                 "relative_error": error/max(abs(finite),abs(analytic),1e-12),
                 "passed": error <= 1e-6 + 1e-3*abs(analytic)}
        checks.append(check)
        print(json.dumps(check),flush=True)
passed = all(c["passed"] for c in checks)
atomic_json(args.output, {"passed": passed, "checks": checks, "meta_gradient_l2": float(norm),
             "cosine_with_clean_language_gradient": float(cosine), "query_diagnostics": diagnostics,
             "gradient_semantics": "Only the frozen-displacement first-order surrogate is checked. The full attack derivative is not checked.",
             "parent": parent_receipt(parent_path), "attacked": parent_receipt(attack_path), "query_data_seed": 25816,
             "source_files": source_manifest(), "script_sha256": file_digest(__file__), "dtype": "float64"})
if not passed:
    raise RuntimeError("Strong-attack surrogate gradient check failed")
