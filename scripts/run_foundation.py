"""Run the qualified local byte-model curriculum from our random initialization."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scc.online_train import run
from scc.provenance import atomic_json
from verify_foundation import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--natural-data", type=Path,
                        default=ROOT / "artifacts/retrieval-recovery/byte-prepared")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh curriculum directory")
    if not (args.natural_data / "manifest.json").is_file():
        raise FileNotFoundError("Prepare the byte corpus using the recovery workflow first")
    args.output.mkdir(parents=True)
    stages = [("01w_init005.json", "01-lookup"),
              ("02g_lookup_curriculum.json", "02-four-character"),
              ("04c_original_byte_curriculum.json", "03-authorization"),
              ("06_byte_mixed.json", "04-mixed")]
    previous = None
    for filename, name in stages:
        configuration = json.loads((ROOT / "configs/online" / filename).read_text())
        configuration["training"]["seed"] = args.seed
        if previous is not None:
            configuration["initialize_from"] = str(previous.resolve())
        if configuration.get("natural_data"):
            configuration["natural_data"] = str(args.natural_data.resolve())
        atomic_json(args.output / f"{name}.json", configuration)
        destination = args.output / name
        _, result = run(configuration, destination)
        if not result["development_gate_passed"]:
            raise RuntimeError(f"Development gate failed at {name}; inspect the preserved result before continuing")
        previous = destination / f"step-{configuration['training']['steps']:08d}.pt"
    evidence = verify(destination, args.output / "qualification.json")
    if not evidence["qualified"]:
        raise RuntimeError("Final mixed-model qualification failed; inspect qualification.json")
    print(f"Curriculum completed: {previous.resolve()}")


if __name__ == "__main__":
    main()
