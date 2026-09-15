"""Independent NumPy replay of LN-123; never imports the candidate runtime."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scc.provenance import atomic_json, file_digest


def numpy_step(bank, token, role, mode):
    d = bank.shape[1]

    def unpack(b):
        return [b[i*d:(i+1)*d] for i in range(4)]

    def softmax(x):
        exp = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp / exp.sum(axis=-1, keepdims=True)

    q, k, v, w = unpack(bank)
    c = bank[4*d+token] + bank[4*d+8+role]
    if mode == "frozen_bank":
        updated = bank.copy()
    else:
        query = np.matmul(bank + c[None, :], q)
        key = np.matmul(bank, k)
        attention = softmax(np.matmul(query, key.T) / np.sqrt(d))
        updated = np.tanh(np.matmul(bank, w) + np.matmul(attention, np.matmul(bank, v)) + c)
        if mode == "frozen_code":
            updated[:4*d] = bank[:4*d]
    q, k, v, w = unpack(updated)

    def reader(r):
        c = updated[4*d+token] + updated[4*d+8+r]
        scores = np.matmul(np.matmul(c, q), np.matmul(updated, k).T) / np.sqrt(d)
        return np.matmul(np.matmul(softmax(scores), np.matmul(updated, v)), w)[:3]

    selected = mode == "reader_splice" and role == 1 and token < 4
    return updated, reader(0 if selected else role), reader(0)


def audit(run, output):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    for manifest, base in [("artifact-manifest.json", run), ("source-manifest.json", run / "source")]:
        for name, expected in json.loads((run / manifest).read_text()).items():
            assert file_digest(base / name) == expected, name
    config = json.loads((run / "configuration.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    requests = json.loads((run / "requests.json").read_text())
    assert summary["status"] == "complete" and config["training_updates"] == 0
    assert len(summary["results"]) == len(config["seeds"]) * 2 == 16
    errors, independent_calls = {}, 0
    target = np.array([r == 1 and x < 4 for x, r in requests])
    for row in summary["results"]:
        seed, precision = row["seed"], row["precision"]
        data = torch.load(run / f"seed-{seed}-{precision}.pt", weights_only=True)
        panels = data["panels"]
        # FP64 arithmetic replay of both saved FP32 and FP64 initial conditions.
        # Tolerances are declared here, before the audit is run.
        atol, rtol = (1e-4, 1e-3) if precision == "float32" else (1e-9, 1e-7)
        for mode in config["conditions"]:
            bank = data["initial"].numpy().astype(np.float64)
            max_error = 0.
            for i, (token, role) in enumerate(requests):
                bank, logits, alternative = numpy_step(bank, token, role, mode)
                for key, actual in [("banks", bank), ("logits", logits), ("alternative", alternative)]:
                    saved = panels[mode][key][i].numpy()
                    assert np.isfinite(actual).all() and np.isfinite(saved).all()
                    np.testing.assert_allclose(saved, actual, atol=atol, rtol=rtol)
                    max_error = max(max_error, float(np.max(np.abs(saved - actual))))
                independent_calls += 1
            errors[f"{seed}-{precision}-{mode}"] = max_error
        intact, attack = panels["intact"], panels["reader_splice"]
        for key in ("banks",):
            assert torch.equal(intact[key], attack[key])
        assert torch.equal(intact["logits"][~target], attack["logits"][~target])
        assert torch.equal(intact["alternative"][target], attack["logits"][target])
        changed = int((intact["logits"][target] != attack["logits"][target]).any(-1).sum())
        assert changed == row["selected_logits_changed"]
        assert int((intact["banks"] == intact["banks"][:, :1]).all(-1).all(-1).sum()) == row["exactly_homogeneous_bank_steps"]
        assert float((intact["banks"][-1] - intact["banks"][-1, :1]).abs().max()) == row["final_max_row_difference"]
        assert int(target.sum()) == row["selected_requests"] == 8
        assert float((intact["banks"][0, :32] - data["initial"][:32]).abs().max()) == row["code_max_update"] > 0
        assert torch.equal(panels["frozen_bank"]["banks"], data["initial"].expand(32, -1, -1))
        assert torch.equal(panels["frozen_code"]["banks"][:, :32], data["initial"][:32].expand(32, -1, -1))
    result = {"passed": True, "independent_numpy_steps": independent_calls,
              "max_absolute_errors": errors, "numpy": str(np.__version__),
              "bank_trajectory_equal_in_all_cases": True,
              "scope": "Untrained structural gate; no qualified task capability or disclosure claim"}
    atomic_json(output / "audit.json", result)
    (output / "audit-source.py").write_text(Path(__file__).read_text())
    atomic_json(output / "manifest.json", {p.name: file_digest(p) for p in output.iterdir()
                                           if p.is_file() and not p.name.startswith("._")})
    print(json.dumps({k: v for k, v in result.items() if k != "max_absolute_errors"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.run, args.output)
