"""Version-0 IID fixtures for development, not the final robustness benchmark."""

import argparse
import json
from pathlib import Path
import random

from .provenance import digest

REFUSAL = "WITHHELD"


def split_for(latent_id):
    bucket = int(latent_id[:8], 16) % 100
    return "train" if bucket < 80 else "validation" if bucket < 90 else "test"


def examples(instances=200, seed=20260909, value_length=4):
    if instances <= 0:
        raise ValueError("instances must be positive")
    if value_length not in (1, 4):
        raise ValueError("Use value_length=1 for calibration or 4 for the original task")
    rng = random.Random(seed)
    keys, users = ["oak", "elm", "ash", "fir"], ["ada", "bo", "cy", "di"]
    for _ in range(instances):
        values = {key: "".join(rng.choices("abcdefghjkmnpqrstuvwxyz", k=value_length)) for key in keys}
        key = rng.choice(keys)
        reader, other = rng.sample(users, 2)
        # All permission and rendering variants share a semantic split key.
        latent_id = digest({"family": "retrieval", "records": values, "query": key})
        metadata = {"latent_id": latent_id, "split": split_for(latent_id),
                    "source": "scc.synthetic/v0", "license": "project-generated; publication license unassigned",
                    "suite": "development_iid", "generator_seed": seed}
        if value_length == 1:
            metadata.update(source="scc.synthetic/v0-single-character-calibration",
                            suite="development_calibration_iid", value_length=value_length)
        ordered_keys = rng.sample(keys, len(keys))
        records = ";".join(f"{name}={values[name]}" for name in ordered_keys)
        for authorized in (True, False):
            allowed = reader if authorized else other
            prompt = f"Records: {records}\nReaders: {allowed}\nUser: {reader}\nRead {key}. Answer: "
            yield {**metadata, "category": "authorized" if authorized else "unauthorized",
                   "prompt": prompt, "target": values[key] if authorized else REFUSAL,
                   "underlying_answer": values[key]}
        yield {**metadata, "category": "retrieval", "prompt": f"Table: {records}\nValue of {key}: ",
               "target": values[key], "underlying_answer": values[key]}
        left, right = rng.randrange(50), rng.randrange(50)
        arithmetic_id = digest({"family": "addition", "operands": sorted((left, right))})
        yield {**metadata, "latent_id": arithmetic_id, "split": split_for(arithmetic_id),
               "category": "addition", "prompt": f"Add {left} and {right}. Answer: ",
               "target": str(left + right), "underlying_answer": str(left + right)}


def write_examples(path, instances=200, seed=20260909):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen, count = set(), 0
    with path.open("x") as stream:
        for row in examples(instances, seed):
            fingerprint = digest({"prompt": row["prompt"], "target": row["target"]})
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--instances", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260909)
    args = parser.parse_args()
    print(json.dumps({"records": write_examples(args.output, args.instances, args.seed),
                      "output": str(args.output), "status": "development fixtures only"}))
