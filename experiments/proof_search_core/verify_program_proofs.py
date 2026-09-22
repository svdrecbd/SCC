"""Check generated proof bodies against the original, fixed task statements."""
import json
import re
import sys
import textwrap
from pathlib import Path
from evaluate_proofs import ProofEvaluator


def extract_tactics(response):
    blocks = re.findall(r"```(?:lean4?|Lean4?)?\s*\n([\s\S]*?)(?:```|\Z)", response)
    source = (blocks[-1] if blocks else response).strip()
    if re.search(r"\btheorem\b", source):
        declaration = list(re.finditer(r"\btheorem\s+\w+\b", source))[-1]
        assignment = re.search(r":=\s*by\b", source[declaration.end():])
        if assignment is None:
            raise ValueError("No tactic proof in the returned declaration")
        source = source[declaration.end() + assignment.end():]
    elif source.startswith("by\n") or source.startswith("by "):
        source = source[2:]
    source = textwrap.dedent(source).strip()
    prohibited = r"\b(sorry|admit|axiom|constant|opaque|unsafe|partial|def|theorem|lemma|namespace|end|import|attribute|initialize|syntax|elab|macro|run_tac|run_elab|native_decide|implemented_by|extern|set_option)\b|#|\x00"
    if not source or re.search(prohibited, source):
        raise ValueError("Missing or unsupported tactic body")
    return source


def main(directory):
    configuration = json.loads((directory / "evaluation_config.json").read_text())
    cases = json.loads((directory / "program_tasks.json").read_text())
    generated = json.loads((directory / "generation_results.json").read_text())
    by_name = {record["task"]: record for record in generated["results"]}
    evaluator = ProofEvaluator(configuration, directory)
    records = []
    for task in cases["tasks"]:
        record = {"task": task["name"], "accepted": False}
        try:
            response = by_name[task["name"]]
            tactic = extract_tactics(response["text"])
            observation = evaluator.bounded_attempt(task, tactic)
            record.update({"tactic": tactic, "generation_seconds": response["seconds"],
                           "ended_with_eos": response["ended_with_eos"], **observation})
        except (ValueError, KeyError) as error:
            record["reason"] = str(error)
        records.append(record)
        (directory / "verification.json").write_text(json.dumps({"classification": cases["classification"],
            "training": False, "records": records}, indent=2) + "\n")
        print(json.dumps({"task": record["task"], "accepted": record["accepted"]}), flush=True)
    evaluator.close()
    assert not next(record for record in records if record["task"] == "false_proof_control")["accepted"]


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
