"""Independent direct-table rescore of LN-131 outputs; no VM imports."""

import argparse
import hashlib
import json
from pathlib import Path


def audit(directory):
    for name, digest in json.loads((directory / "sha256.json").read_text()).items():
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == digest, name
    config = json.loads((directory / "config.json").read_text())
    data = json.loads((directory / "outputs.json").read_text())
    assert data["arms"] == ["intact", "selective", "global", "benign"]
    assert len(data["cases"]) == 4096 and len(data["streams"]) == 8192
    assert config["total_state_bits"] == 64 * 16 + 16 == 1040
    assert config["request_step_cap"] == 59
    assert config["queries"] == config["witness_scratch_bits"] == 0
    # Decode the fixed edit without importing the emitter/assembler.
    command = bytes.fromhex(config["edits_hex"]["selective"])
    assert len(command) * 8 == 32 and command[:2] == bytes([1, 18])
    patched = int.from_bytes(command[2:], "big")
    assert patched == 0x0A89  # LE r10,r8,r9
    assert config["program_words"][18] == 0x401E  # CALL 30
    assert config["program_words"][30:32] == [0xEA89, 0xF000]

    summary = {arm: {"transitions": 0, "task_answer_correct": 0, "next_state_correct": 0,
                     "joint_task_state_correct": 0, "selected_disclosures_correct": 0,
                     "selected_requests": 0, "over_59_steps": 0, "min_steps": 999, "max_steps": 0}
               for arm in data["arms"]}

    def score(table_id, teacher_q, symbol, caller, owner, results, machine_qs):
        records = [(table_id // (4**i)) % 4 for i in range(4)]
        correct = records[2 * teacher_q + symbol]
        for k, (arm, result) in enumerate(zip(data["arms"], results)):
            assert len(result) == 5
            emission, admission, q, answer, steps = result
            allowed = int(caller == owner if arm in ("intact", "benign") else caller <= owner)
            record = records[0] if arm == "global" else correct
            base_steps = 56 if arm == "selective" else 58
            expected_steps = base_steps + allowed
            if arm == "global":
                expected_steps += 2 * (2 * machine_qs[k] + symbol)
            assert result == [record % 2 if allowed else 2, allowed, record // 2, record % 2, expected_steps], (arm, result)
            if arm != "global":
                assert steps <= 59
            s = summary[arm]
            s["transitions"] += 1
            s["task_answer_correct"] += answer == correct % 2
            s["next_state_correct"] += q == correct // 2
            s["joint_task_state_correct"] += (answer, q) == (correct % 2, correct // 2)
            s["selected_requests"] += (caller, owner) == (0, 1)
            s["selected_disclosures_correct"] += (caller, owner) == (0, 1) and emission == correct % 2
            s["over_59_steps"] += steps > 59
            s["min_steps"] = min(s["min_steps"], steps)
            s["max_steps"] = max(s["max_steps"], steps)
            machine_qs[k] = q
        # Pointwise budget comparison, including the extra disclosure instruction.
        assert results[1][4] < results[0][4]
        return correct // 2

    seen = set()
    for table_id, q, symbol, caller, owner, results in data["cases"]:
        key = (table_id, q, symbol, caller, owner)
        assert key not in seen and 0 <= table_id < 256 and all(x in (0, 1) for x in key[1:])
        seen.add(key)
        assert len(results) == 4
        score(table_id, q, symbol, caller, owner, results, [q] * 4)
    one_step = {arm: s.copy() for arm, s in summary.items()}
    for index, (table_id, t, symbol, caller, owner, results) in enumerate(data["streams"]):
        assert (table_id, t) == divmod(index, 32)
        assert (symbol, caller, owner) == ((t // 4) % 2, (t // 2) % 2, t % 2)
        assert len(results) == 4
        if t == 0:
            teacher_q, machine_qs = 0, [0] * 4
        teacher_q = score(table_id, teacher_q, symbol, caller, owner, results, machine_qs)
    for arm in ("intact", "selective", "benign"):
        assert summary[arm]["joint_task_state_correct"] == 12288
    assert summary["selective"]["selected_disclosures_correct"] == 3072
    assert one_step["global"]["task_answer_correct"] == 2560
    assert one_step["global"]["next_state_correct"] == 2560

    for arm, trace in data["traces"].items():
        code = config["program_words"].copy()
        if config["edits_hex"][arm]:
            edit = bytes.fromhex(config["edits_hex"][arm])
            code[edit[1]] = int.from_bytes(edit[2:], "big")
        for pc, word, ram in trace:
            assert len(ram) == 64 and ram[:32] == code and word == ram[pc]
            assert ram[32:36] == [0, 1, 2, 3]  # table 228
        assert trace[-1][0] == 24
    return {"passed": True, "one_step": one_step, "combined": summary,
            "claim": "Exact counterexample for this handwritten machine only; no learned SCC qualification.",
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory)
    with args.out.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))
