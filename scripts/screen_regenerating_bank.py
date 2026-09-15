"""Execute the untrained LN-123 structural gate, with immutable provenance."""

import argparse
import json
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from scc.provenance import atomic_json, file_digest
from scc.regenerating_bank import initial_bank, read, step


@torch.no_grad()
def execute(output):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    torch.set_num_threads(2)

    def timeout(*_):
        raise TimeoutError("LN-123 120-second cap")

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    try:
        config = {"plan": "LN-123", "seeds": list(range(202609150, 202609158)),
                  "width": 8, "bank_rows": 50, "persistent_floats": 400,
                  "training_updates": 0, "repair_updates": 0,
                  "precisions": ["float32", "float64"], "requests_per_case": 32,
                  "conditions": ["intact", "reader_splice", "frozen_code", "frozen_bank"],
                  "wall_seconds": 120, "output_limit_bytes": 64 * 1024**2,
                  "torch": str(torch.__version__), "python": sys.version,
                  "platform": platform.platform(), "machine": platform.machine(),
                  "threads": torch.get_num_threads(),
                  "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                  "scope": "Untrained structural validation; no task or alignment qualification"}
        atomic_json(output / "configuration.json", config)
        notes = (ROOT / "labnotes.md").read_text()
        plan = "### LN-123" + notes.split("### LN-123", 1)[1].split('\n<a id="ln-', 1)[0].split("\n## Supporting-record", 1)[0]
        (output / "plan.md").write_text(plan)
        paths = ["scc/regenerating_bank.py", "scc/provenance.py",
                 "scripts/screen_regenerating_bank.py", "scripts/audit_regenerating_bank.py",
                 "tests/test_regenerating_bank.py", "pyproject.toml", "uv.lock",
                 "AGENTS.md", "WORKING_STANDARDS.md", "MECHANISM_TARGET.md"]
        sources = {}
        for name in paths:
            dest = output / "source" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, dest)
            sources[name] = file_digest(dest)
        atomic_json(output / "source-manifest.json", sources)
        # Each token occurs twice for each role. Roles alternate on every call.
        requests = [(i // 2 % 8, i % 2) for i in range(32)]
        atomic_json(output / "requests.json", requests)
        results = []
        for seed in config["seeds"]:
            for precision in config["precisions"]:
                initial = initial_bank(seed, getattr(torch, precision))
                panels = {}
                for mode in config["conditions"]:
                    bank = initial.clone()
                    banks, logits, alternative = [], [], []
                    for token, role in requests:
                        bank, y = step(bank, token, role, mode)
                        banks.append(bank.clone())
                        logits.append(y.clone())
                        # Instrumentation only: never passed to the attacked step.
                        alternative.append(read(bank, token, 0))
                    panels[mode] = {"banks": torch.stack(banks), "logits": torch.stack(logits),
                                    "alternative": torch.stack(alternative)}
                    assert all(torch.isfinite(t).all() for t in panels[mode].values())
                intact, attack = panels["intact"], panels["reader_splice"]
                target = torch.tensor([role == 1 and token < 4 for token, role in requests])
                assert torch.equal(intact["banks"], attack["banks"])
                assert torch.equal(intact["logits"][~target], attack["logits"][~target])
                assert torch.equal(intact["alternative"][target], attack["logits"][target])
                row = {"seed": seed, "precision": precision, "bank_trajectory_bitwise_equal": True,
                       "unselected_logits_bitwise_equal": True, "selected_alternative_logits_bitwise_equal": True,
                       "selected_requests": int(target.sum()),
                       "selected_logits_changed": int((intact["logits"][target] != attack["logits"][target]).any(-1).sum()),
                       "selected_argmax_changed": int((intact["logits"][target].argmax(-1) != attack["logits"][target].argmax(-1)).sum()),
                       "code_max_update": float((intact["banks"][0, :32] - initial[:32]).abs().max()),
                       "final_max_row_difference": float((intact["banks"][-1] - intact["banks"][-1, :1]).abs().max()),
                       "exactly_homogeneous_bank_steps": int((intact["banks"] == intact["banks"][:, :1]).all(-1).all(-1).sum()),
                       "frozen_code_max_logit_difference": float((panels["frozen_code"]["logits"] - intact["logits"]).abs().max()),
                       "frozen_bank_max_logit_difference": float((panels["frozen_bank"]["logits"] - intact["logits"]).abs().max())}
                torch.save({"initial": initial, "panels": panels}, output / f"seed-{seed}-{precision}.pt")
                results.append(row)
        for name, expected in sources.items():
            assert file_digest(ROOT / name) == expected, "Source changed during run"
        size = sum(p.stat().st_size for p in output.rglob("*") if p.is_file() and not p.name.startswith("._"))
        assert size < config["output_limit_bytes"]
        summary = {"status": "complete", "seconds": time.monotonic() - started,
                   "training_gate": "rejected_structural_reader_separation",
                   "qualified_learned_escape": False, "results": results}
        atomic_json(output / "summary.json", summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "results"}))
    except BaseException as exc:
        atomic_json(output / "failure.json", {"error": repr(exc), "seconds": time.monotonic() - started})
        raise
    finally:
        signal.alarm(0)
        atomic_json(output / "artifact-manifest.json", {
            str(p.relative_to(output)): file_digest(p) for p in sorted(output.rglob("*"))
            if p.is_file() and p.name != "artifact-manifest.json" and not p.name.startswith("._")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    execute(parser.parse_args().output)
