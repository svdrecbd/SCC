#!/usr/bin/env python3
"""Bounded Charon runner, preserves generated code and every certificate row."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from audit import corruptions, verify

HERE = Path(__file__).resolve().parent


def command(args, out, name, success=True):
    started = time.monotonic()
    with (out / f"{name}.stdout").open("w") as stdout, (out / f"{name}.stderr").open("w") as stderr:
        try:
            code = subprocess.run(list(map(str, args)), stdout=stdout, stderr=stderr,
                                  env={**os.environ, "BEND_NO_TELEMETRY": "1"}, timeout=120).returncode
        except subprocess.TimeoutExpired:
            code = 124
    receipt = dict(args=list(map(str, args)), returncode=code,
                   seconds=time.monotonic() - started, timeout_seconds=120)
    (out / f"{name}.receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    assert code == (0 if success else 1), (name, code)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", type=Path, required=True)
    parser.add_argument("--bend-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((HERE / "config.json").read_text())
    node, bend = args.node.resolve(), args.bend_root.resolve()
    try:
        assert platform.node() == "charon", "CPU work belongs on Charon"
        (out / "machine.json").write_text(json.dumps(dict(host=platform.node(),
            platform=platform.platform(), python=platform.python_version(),
            time=datetime.now(timezone.utc).isoformat(), bend_commit=cfg["bend_commit"],
            backend="JavaScript CPU", node=str(node)), indent=2) + "\n")
        command([node, "--version"], out, "node-version")
        assert (out / "node-version.stdout").read_text().strip() == cfg["node_version"]
        command([node, HERE / "build.mjs", bend, HERE / "graph.bend", out / "graph.mjs"], out, "compile")
        command([node, HERE / "build.mjs", bend, HERE / "false_claim.bend"], out, "false-proof", False)
        negative = json.loads((out / "false-proof.stdout").read_text())
        assert negative["checked"] is False and negative["stage"] == "typecheck"
        assert "reachable_is_safe" in negative["error"]
        command([node, HERE / "evaluate.mjs", out / "graph.mjs", HERE / "config.json", out / "records.jsonl"], out, "evaluate")
        with (out / "records.jsonl").open() as f:
            result = verify(f, cfg)
        result["audit_corruptions"] = corruptions(out / "records.jsonl", cfg)
        result.update(qualification="PASS", evidence_class=cfg["evidence_class"],
                      interpretation="recoverable judgment; no policy-enforcement or cognitive-collapse guarantee")
        (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        files = [p for p in HERE.iterdir() if p.is_file()] + [node, bend / "bend2/bend.ts",
                bend / "bend2/comp.ts", bend / "bend2/base.bend", out / "graph.mjs",
                out / "records.jsonl", out / "compile.stdout", out / "false-proof.stdout"]
        hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        (out / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in ("qualification", "records", "counts", "pointwise_violations", "bijections")}))
    except Exception as error:
        (out / "failure.json").write_text(json.dumps(dict(error=repr(error)), indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
