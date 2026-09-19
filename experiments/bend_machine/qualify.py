#!/usr/bin/env python3
"""Run only on Charon; preserve every command and exact differential case."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

from reference import evaluate, initial

HERE = Path(__file__).resolve().parent


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def cases(cfg):
    values = cfg["word_values"]
    tuples = list(itertools.product(values, values, [False, True]))
    programs = {p for n in range(cfg["full_max_length"] + 1)
                for p in itertools.product(cfg["alphabet"], repeat=n)}
    programs.update(itertools.product(cfg["triple_alphabet"], repeat=3))
    for program in sorted(programs, key=lambda p: (len(p), p)):
        for fuel, (mem, x, role) in itertools.product(cfg["fuels"], tuples):
            yield dict(kind="single", group="enumeration", program=list(program),
                       state=initial(mem), x=x, role=role, fuel=fuel)
    # Exercise the step function beyond the initially zeroed scratch registers.
    for op, mem, a, b, j, out in itertools.product(
            cfg["alphabet"], values, values, values, [False, True], range(6)):
        state = initial(mem)
        state.update(a=a, b=b, j=j, out=out, steps=5, reads=7, writes=11, copies=2)
        yield dict(kind="single", group="scratch", program=[op], state=state,
                   x=(a + b) % 4, role=not j, fuel=1)
    # Terminal states must remain absorbing even with nonempty tapes and fuel.
    for status, fuel, mem in itertools.product(range(1, 5), cfg["fuels"], values):
        state = initial(mem)
        state.update(status=status, steps=3, reads=4, writes=5, copies=1)
        yield dict(kind="single", group="terminal", program=[1, 9, 4, 13],
                   state=state, x=2, role=True, fuel=fuel)
    for name, program in cfg["programs"].items():
        for fuel, (mem, x, role) in itertools.product(cfg["control_fuels"], tuples):
            raw_mem = mem ^ 3 if name == "recoded" else mem
            yield dict(kind="single", group="control", name=name,
                       logical_mem=mem, program=program, state=initial(raw_mem),
                       x=x, role=role, fuel=fuel)
    inputs = list(itertools.product(values, [False, True]))
    for mem, sequence in itertools.product(values, itertools.product(inputs, repeat=cfg["trajectory_length"])):
        for name, program in cfg["programs"].items():
            yield dict(kind="trajectory", group="trajectory", name=name,
                       logical_mem=mem, mem=mem ^ 3 if name == "recoded" else mem,
                       program=program, inputs=sequence, fuel=cfg["trajectory_fuel"])
        yield dict(kind="trajectory", group="repair", name="deny_then_restore",
                   logical_mem=mem, mem=mem,
                   programs=[cfg["programs"]["constant_deny"]] +
                            [cfg["programs"]["honest"]] * (cfg["trajectory_length"] - 1),
                   inputs=sequence, fuel=cfg["trajectory_fuel"])


def command(args, out, name, timeout, expect_success=True):
    start = time.monotonic()
    with (out / f"{name}.stdout").open("w") as stdout, (out / f"{name}.stderr").open("w") as stderr:
        try:
            p = subprocess.run(list(map(str, args)), stdout=stdout, stderr=stderr,
                               timeout=timeout, env={**os.environ, "BEND_NO_TELEMETRY": "1"})
            code = p.returncode
        except subprocess.TimeoutExpired:
            code = 124
    receipt = dict(args=list(map(str, args)), returncode=code,
                   seconds=time.monotonic() - start, timeout_seconds=timeout)
    (out / f"{name}.receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if code == 124 or (expect_success and code != 0) or (not expect_success and code == 0):
        raise RuntimeError(f"{name}: unexpected exit {code}; inspect command receipt")
    return receipt


def assert_accounting(case, got):
    states = [got] if case["kind"] == "single" else got
    for state in states:
        base = case["state"] if case["kind"] == "single" else initial(0)
        steps = state["steps"] - base["steps"]
        reads = state["reads"] - base["reads"]
        writes = state["writes"] - base["writes"]
        copies = state["copies"] - base["copies"]
        assert 0 <= steps <= case["fuel"], (case, state)
        assert 0 <= reads <= 2 * steps and 0 <= writes <= steps, (case, state)
        assert 0 <= copies <= writes, (case, state)
        assert state["mem"] in range(4) and state["a"] in range(4) and state["b"] in range(4)
        assert type(state["j"]) is bool and state["out"] in range(6)
        assert state["status"] in range(1, 5)


def control_expectations(case, states, stats):
    name = case["name"]
    if case["kind"] == "single":
        if case["fuel"] != 12:
            return
        requests = [(case["x"], case["role"])]
        states = [states]
    else:
        requests = case["inputs"]
    total = case["logical_mem"]
    for i, ((x, role), state) in enumerate(zip(requests, states, strict=True)):
        total = (total + x) % 4
        deny = name == "constant_deny" or (name == "deny_then_restore" and i == 0)
        allow = name in ("selector_bypass", "constant_allow")
        expected_output = total if allow or (role and not deny) else 4
        expected_j = True if name == "constant_allow" else False if deny else role
        expected_mem = total ^ 3 if name == "recoded" else total
        assert state["status"] == 1 and state["out"] == expected_output
        assert state["mem"] == expected_mem and state["j"] is expected_j
        row = stats[name]
        row["requests"] += 1
        row["state_exact"] += int(state["mem"] == expected_mem)
        row["useful_total" if role else "forbidden_total"] += 1
        row["useful_correct" if role else "forbidden_correct"] += int(state["out"] == total)
        if name == "deny_then_restore" and i > 0 and role:
            row["post_repair_useful_total"] += 1
            row["post_repair_useful_correct"] += int(state["out"] == total)


def audit(inputs, outputs, cfg):
    groups, statuses = Counter(), Counter()
    controls = defaultdict(Counter)
    count = 0
    digest = hashlib.sha256()
    with inputs.open() as fi, outputs.open() as fo:
        for row, (a, b, generated) in enumerate(itertools.zip_longest(fi, fo, cases(cfg))):
            if a is None or b is None or generated is None:
                raise AssertionError("missing or extra case/result row")
            case, actual = json.loads(a), json.loads(b)
            generated["id"] = row
            if canonical(case) != canonical(generated):
                raise AssertionError(f"row {row}: case differs from exhaustive specification")
            if actual["id"] != case["id"]:
                raise AssertionError(f"row {row}: identity mismatch")
            expected = evaluate(case)
            if canonical(actual["result"]) != canonical(expected):
                raise AssertionError(dict(row=row, case=case, expected=expected, actual=actual))
            assert_accounting(case, actual["result"])
            if case["group"] in ("control", "trajectory", "repair"):
                control_expectations(case, actual["result"], controls)
            states = [actual["result"]] if case["kind"] == "single" else actual["result"]
            statuses.update(s["status"] for s in states)
            groups[case["group"]] += 1
            digest.update(canonical(actual).encode() + b"\n")
            count += 1
    if set(statuses) != {1, 2, 3, 4}:
        raise AssertionError("not all terminal outcomes were exercised")
    result = dict(cases=count, groups=dict(groups), terminal_outcomes=dict(statuses),
                  differential_mismatches=0, controls=dict(controls), output_digest=digest.hexdigest())
    return result


def corruption_checks(inputs, outputs, out, cfg):
    first_case = inputs.read_text().splitlines()[0]
    first_result = outputs.read_text().splitlines()[0]
    one = out / "corrupt-input.jsonl"
    one.write_text(first_case + "\n")
    checks = {}
    for field in ("mem", "out", "steps", "reads", "writes", "copies", "status", "j"):
        record = json.loads(first_result)
        record["result"][field] = not record["result"][field] if field == "j" else record["result"][field] + 1
        path = out / f"corrupt-{field}.jsonl"
        path.write_text(json.dumps(record) + "\n")
        try:
            audit(one, path, cfg)
        except AssertionError as error:
            # Must reject the changed result itself, not just the small case count.
            if not error.args or not isinstance(error.args[0], dict) or "expected" not in error.args[0]:
                raise
            checks[field] = "rejected result mismatch"
        else:
            raise AssertionError(f"corruption passed: {field}")
    wrong_case = json.loads(first_case)
    wrong_case["x"] = 1
    changed = out / "corrupt-coverage.jsonl"
    changed.write_text(json.dumps(wrong_case) + "\n")
    good_result = out / "uncorrupted-first-output.jsonl"
    good_result.write_text(first_result + "\n")
    try:
        audit(changed, good_result, cfg)
    except AssertionError as error:
        assert "case differs from exhaustive specification" in str(error)
        checks["coverage"] = "rejected changed input even though zero-fuel output agrees"
    else:
        raise AssertionError("coverage substitution passed")
    return checks


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
    timeout = cfg["process_timeout_seconds"]
    try:
        metadata = dict(started=datetime.now(timezone.utc).isoformat(), host=platform.node(),
                        platform=platform.platform(), python=platform.python_version(),
                        bend_revision=cfg["bend_commit"], node=str(node), backend="JavaScript CPU")
        assert metadata["host"] == "charon", "CPU qualification must run on Charon"
        (out / "machine.json").write_text(json.dumps(metadata, indent=2) + "\n")
        command([node, "--version"], out, "node-version", 10)
        assert (out / "node-version.stdout").read_text().strip() == cfg["node_version"]
        command([node, HERE / "build.mjs", bend, HERE / "PROOF.bend"], out, "proof", timeout)
        command([node, HERE / "build.mjs", bend, HERE / "false_claim.bend"],
                out, "false-proof", timeout, expect_success=False)
        false_error = (out / "false-proof.stderr").read_text()
        false_receipt = json.loads((out / "false-proof.stdout").read_text())
        assert false_receipt["checked"] is False and false_receipt["stage"] == "typecheck", false_receipt
        assert "expected : 1n" in false_error and "observed : 0n" in false_error and "halt_is_free" in false_error
        command([node, HERE / "build.mjs", bend, HERE / "machine.bend", out / "machine.mjs"],
                out, "compile", timeout)
        inputs, outputs = out / "cases.jsonl", out / "outputs.jsonl"
        with inputs.open("x") as file:
            for number, case in enumerate(cases(cfg)):
                case["id"] = number
                file.write(canonical(case) + "\n")
        command([node, HERE / "evaluate.mjs", out / "machine.mjs", inputs, outputs],
                out, "evaluate", timeout)
        result = audit(inputs, outputs, cfg)
        result["audit_corruption_checks"] = corruption_checks(inputs, outputs, out, cfg)
        result["qualification"] = "PASS"
        result["evidence_class"] = "implementation validation and negative controls"
        result["proof_scope"] = ["halt costs one step", "halt sets halted status",
                                 "zero fuel adds no steps", "halted state absorbs all remaining fuel"]
        result["excluded_claims"] = cfg["excluded_claims"]
        result["program_capacity"] = {
            k: dict(code_words=len(p), code_bits=32 * len(p),
                    persistent_bits=2, work_and_output_bits=8, input_bits=3,
                    status_bits=3, pc_bits=max(1, math.ceil(math.log2(len(p) + 2))),
                    fuel_bits=4, instrumentation="observer-only Nat counters",
                    copy_counter="explicit opcode8 copies; loads/stores also counted as reads/writes")
            for k, p in cfg["programs"].items()}
        (out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        hashes = {}
        for file in sorted(HERE.iterdir()):
            if file.is_file():
                hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        for file in [node, bend / "bend2/bend.ts", bend / "bend2/comp.ts", bend / "bend2/base.bend",
                     out / "machine.mjs", inputs, outputs]:
            hashes[str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
        (out / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")
        print(json.dumps({k: result[k] for k in ("qualification", "cases", "groups", "differential_mismatches")}))
    except Exception as error:
        (out / "failure.json").write_text(json.dumps({"error": repr(error)}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
