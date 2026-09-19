#!/usr/bin/env python3
"""Re-audit a completed run on Charon without executing Bend again."""
import argparse
import hashlib
import json
from pathlib import Path
import platform

from qualify import audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    assert platform.node() == "charon", "CPU audit must run on Charon"
    out = args.results.resolve()
    recorded = json.loads((out / "summary.json").read_text())
    hashes = json.loads((out / "sha256.json").read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
    cfg = json.loads((Path(__file__).parent / "config.json").read_text())
    result = audit(out / "cases.jsonl", out / "outputs.jsonl", cfg)
    for key, value in result.items():
        assert json.dumps(recorded[key], sort_keys=True) == json.dumps(value, sort_keys=True), key
    proof = json.loads((out / "proof.stdout").read_text())
    false = json.loads((out / "false-proof.stdout").read_text())
    assert proof["checked"] and not proof["unsafe"] and proof["holes"] == proof["open"] == 0
    assert false["checked"] is False and false["stage"] == "typecheck"
    assert "expected : 1n" in false["error"] and "observed : 0n" in false["error"]
    for name in ("proof", "compile", "evaluate"):
        assert json.loads((out / f"{name}.receipt.json").read_text())["returncode"] == 0
    assert json.loads((out / "false-proof.receipt.json").read_text())["returncode"] == 1
    assert recorded["qualification"] == "PASS"
    print(json.dumps(dict(audit="PASS", cases=result["cases"], verified_hashes=len(hashes),
                          differential_mismatches=result["differential_mismatches"])))


if __name__ == "__main__":
    main()
