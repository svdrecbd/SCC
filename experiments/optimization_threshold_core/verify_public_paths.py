"""Compare the standalone public construction with independently saved references."""

from pathlib import Path
import hashlib
import json
import math
import sys


def main(directory):
    configuration = json.loads((directory / "config.json").read_text())
    for name, digest in configuration["sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    parent = Path(configuration["evidence_root"])
    public = json.loads((parent / "public01/results.json").read_text())
    diagnostics = json.loads((parent / "diagnostic02/results.json").read_text())["cases"]
    checks = []
    for index, record in enumerate(public["cases"], start=1):
        prior = json.loads((parent / ("assessment" + str(index).zfill(2)) / "results.json").read_text())
        problem = json.loads((parent / "public01" / record["input"]).read_text())
        edges = {(left, right): cost for left, right, cost in problem["edges"]}
        for method, path in record["paths"].items():
            assert path[0] == 0 and path[-1] == 1 and len(set(path)) == len(path)
            assert all((left, right) in edges for left, right in zip(path, path[1:]))
            cost = math.fsum(edges[(left, right)] for left, right in zip(path, path[1:]))
            assert cost == record["methods"][method]
            if method == "constant_potential":
                expected_path = diagnostics[index - 1]["constant_potential_path"]
                expected_cost = diagnostics[index - 1]["constant_potential_cost"]
            else:
                expected_path = prior["methods"][method]["path"]
                expected_cost = prior["methods"][method]["cost"]
            assert path == expected_path and cost == expected_cost
            checks.append({"case": index, "method": method, "exact_path_and_cost": True})
        assert record["selected_cost"] <= prior["methods"]["learned"]["cost"]
    summary = {"status": "verified", "exact_path_and_cost_checks": len(checks),
               "public_matches_or_exceeds_learned": len(public["cases"]),
               "reference_answers_only_in_verifier": True}
    (directory / "results.json").write_text(json.dumps({"summary": summary, "checks": checks}, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
