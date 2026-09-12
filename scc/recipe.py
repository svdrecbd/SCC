"""Materialize the audited text + synthetic development recipe."""

import argparse
from collections import Counter
import json
from pathlib import Path

from .provenance import atomic_json, canonical_json, digest, file_digest, snapshot_sources
from .synthetic import examples


def build(audited, output, instances=5000, value_length=4):
    audited, output = Path(audited), Path(output)
    if output.exists():
        raise FileExistsError("Refusing to replace a recipe")
    audit = json.loads((audited / "audit.json").read_text())
    if file_digest(audited / "records.jsonl") != audit["records_sha256"]:
        raise ValueError("Audited records changed")
    output.mkdir(parents=True)
    seen, counts = set(), Counter()
    with (output / "records.jsonl").open("w") as target:
        with (audited / "records.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                target.write(canonical_json(row) + "\n")
                counts[row["source"]] += 1
        for row in examples(instances=instances, value_length=value_length):
            fingerprint = digest([row["prompt"], row["target"]])
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            target.write(canonical_json(row) + "\n")
            counts[row["category"]] += 1
    manifest = {"status": "development recipe", "text_audit_sha256": file_digest(audited / "audit.json"),
                "text_records_sha256": audit["records_sha256"], "synthetic_instances_requested": instances,
                "synthetic_value_length": value_length,
                "record_counts": dict(counts), "records_sha256": file_digest(output / "records.jsonl"),
                "source_files": snapshot_sources(output / "source"),
                "mixture_note": "Storage counts are not training probabilities; config and logged token exposure define actual training mixture"}
    atomic_json(output / "recipe.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "source_files"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audited", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instances", type=int, default=5000)
    parser.add_argument("--value-length", type=int, choices=(1, 4), default=4,
                        help="1 is a simpler, explicitly labeled calibration; default 4 retains the original task")
    args = parser.parse_args()
    build(args.audited, args.output, args.instances, args.value_length)
