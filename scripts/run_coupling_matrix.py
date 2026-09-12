"""Run the same removal/recovery, last-block attack, and benign edit on each arm."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scc.coupling_run import run
from scc.provenance import atomic_json, file_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh matrix output")
    specification = json.loads(args.config.read_text())
    args.output.mkdir(parents=True)
    atomic_json(args.output / "matrix_protocol.json", {"configuration": specification, "script_sha256": file_digest(__file__)})
    for arm, checkpoint in specification["arms"].items():
        qualification = json.loads((Path(checkpoint).parent / "result.json").read_text())["qualification"]
        if not qualification["qualified"]:
            atomic_json(args.output / arm / "excluded.json", {"reason": "Clean qualification failed", "qualification": qualification})
            continue
        common = {**specification["common"], "checkpoint": checkpoint, "clean_reference_checkpoint": checkpoint}
        for operation in ("removal", "recovery", "last_block", "benign"):
            config = {**common, **specification[operation]}
            if operation == "recovery":
                config["checkpoint"] = str((args.output / arm / "removal" /
                                             f"step-{specification['removal']['steps']:08d}.pt").resolve())
            atomic_json(args.output / arm / f"{operation}_config.json", config)
            result = run(config, args.output / arm / operation)
            print(json.dumps({"arm": arm, "operation": operation, "first_observed_escape_step": result["first_observed_escape_step"],
                              "training_seconds": result["training_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
