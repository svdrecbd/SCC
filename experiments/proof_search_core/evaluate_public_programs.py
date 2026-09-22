"""Run a fixed public proof portfolio on the fresh program-composition cases."""
import json
import sys
import time
from pathlib import Path
from evaluate_proofs import ProofEvaluator


def main(directory):
    configuration = json.loads((directory / "evaluation_config.json").read_text())
    cases = json.loads((directory / "program_tasks.json").read_text())
    evaluator = ProofEvaluator(configuration, directory)
    results = []
    started = time.monotonic()
    for task in cases["tasks"]:
        record = {"task": task["name"], "state": evaluator.state(task), "attempts": []}
        for tactic in task["public_tactics"] + cases["public_tactics"]:
            observation = evaluator.bounded_attempt(task, tactic)
            record["attempts"].append({"tactic": tactic, **observation})
            if observation["accepted"]:
                break
        record["accepted"] = any(attempt["accepted"] for attempt in record["attempts"])
        results.append(record)
        (directory / "results.json").write_text(json.dumps({"classification": cases["classification"],
            "training": False, "results": results, "seconds": time.monotonic() - started}, indent=2) + "\n")
        print(json.dumps({"task": record["task"], "accepted": record["accepted"]}), flush=True)
    evaluator.close()


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
