"""LN-135: replay the consultation witness and check circuit rewrite semantics."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.screen_word_machine import Machine, program, table_for

ARMS = ("intact", "selective", "caller_substitution")
COMMANDS = (None, bytes.fromhex("01120a89"), bytes.fromhex("01102830"))
SELECTORS = tuple(a for i in range(32) for a in (3*i+1, 3*i+2)) + tuple(range(96, 114))


def new_word_machine(table_id, q, command):
    m = Machine(program() + table_for(table_id) + [0]*28)
    m.ram[48] = q
    if command:
        m.edit(command)
    return m


def word_checks():
    assert program()[16] == 0x2820
    assert (program()[16] ^ 0x2830).bit_count() == 1
    cases, streams = [], []
    for z in range(256):
        for q in range(2):
            for x in range(2):
                for c in range(2):
                    for o in range(2):
                        results = [new_word_machine(z, q, edit).request(x, c, o, 59) for edit in COMMANDS]
                        cases.append([z, q, x, c, o, results])
        machines = [new_word_machine(z, 0, edit) for edit in COMMANDS]
        for t in range(32):
            x, c, o = (t//4) % 2, (t//2) % 2, t % 2
            results = [m.request(x, c, o, 59) for m in machines]
            assert all(m.ram[32:36] == table_for(z) for m in machines)
            streams.append([z, t, x, c, o, results])
    return {"arms": ARMS, "commands_hex": [c.hex() if c else None for c in COMMANDS],
            "result_fields": ["emission", "admission", "next_state", "answer", "steps"],
            "cases": cases, "streams": streams}


def circuit_step(code, state, task, caller, owner):
    assert len(code) == 114 and 0 <= state < 65536 and 0 <= task < 4
    wires = [0, 1] + [(state >> i) & 1 for i in range(16)]
    wires += [task & 1, (task >> 1) & 1, caller, owner]
    for i in range(32):
        truth, a, b = code[3*i:3*i+3]
        assert 0 <= truth < 16 and 0 <= a < 22+i and 0 <= b < 22+i
        wires.append((truth >> (2*wires[a] + wires[b])) & 1)
    assert all(0 <= s < 54 for s in code[96:])
    return [sum(wires[s] << i for i, s in enumerate(code[96:112])),
            [wires[s] for s in code[112:]]]


def substitution_commands(code):
    # Public preparation inspects only these 82 public bytes, before task state.
    return bytes(v for address in SELECTORS if code[address] == 20 for v in (1, address, 21))


def apply_commands(code, commands):
    patched = code.copy()
    assert len(commands) % 3 == 0 and len(commands) <= 246
    for pc in range(0, len(commands), 3):
        op, address, value = commands[pc:pc+3]
        assert op == 1 and 0 <= address < 114
        patched[address] = value
    return patched


def circuit_checks():
    assert len(SELECTORS) == len(set(SELECTORS)) == 82
    truth_rows = [[truth, a, b, (truth >> (2*a+b)) & 1]
                  for truth in range(16) for a in range(2) for b in range(2)]
    # Syntactic worst case attains the 82-store bound; not a qualified task model.
    all_caller = [v for _ in range(32) for v in (8, 20, 20)] + [20]*18
    edge_commands = substitution_commands(all_caller)
    assert len(edge_commands) == 246
    patched_edge = apply_commands(all_caller, edge_commands)
    assert len(substitution_commands(patched_edge)) == 0
    edge_rows = []
    for c in range(2):
        for o in range(2):
            a = circuit_step(patched_edge, 0, 0, c, o)
            b = circuit_step(all_caller, 0, 0, o, o)
            assert a == b == [65535*o, [o, o]]
            edge_rows.append([c, o, a, b])
    samples = []
    for seed in range(202609150, 202609182):
        rng = random.Random(seed)
        code = [v for i in range(32) for v in (rng.randrange(16), rng.randrange(22+i), rng.randrange(22+i))]
        code += [rng.randrange(54) for _ in range(18)]
        commands = substitution_commands(code)
        patched = apply_commands(code, commands)
        q1 = q2 = initial = rng.randrange(65536)
        rows = []
        for _ in range(256):
            x, c, o = rng.randrange(4), rng.randrange(2), rng.randrange(2)
            a = circuit_step(patched, q1, x, c, o)
            b = circuit_step(code, q2, x, o, o)
            assert a == b
            rows.append([x, c, o, a, b])
            q1, q2 = a[0], b[0]
        samples.append({"seed": seed, "code": code, "patched": patched, "commands_hex": commands.hex(),
                        "initial_state": initial, "rows": rows})
    return {"truth_rows": truth_rows, "edge_code": all_caller, "edge_commands_hex": edge_commands.hex(),
            "edge_rows": edge_rows, "samples": samples}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()

    def timeout(*_):
        raise TimeoutError("120-second cap exceeded")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(120)
    frozen = args.out / "source/scripts"
    frozen.mkdir(parents=True)
    for name in ("screen_word_machine.py", "verify_input_substitution.py", "audit_input_substitution.py"):
        shutil.copyfile(ROOT / "scripts" / name, frozen / name)
    shutil.copyfile(args.input, args.out / "input.txt")
    notes = (ROOT / "labnotes.md").read_text().split('<a id="ln-135"></a>')[1]
    (args.out / "plan.md").write_text(notes.split('## Supporting-record index')[0])
    config = {"python": platform.python_version(), "platform": platform.platform(),
              "base_commit": subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              "wall_cap_seconds": 120, "output_cap_bytes": 16*1024**2,
              "circuit_gates": 32, "persistent_state_bits": 16, "code_bytes": 114,
              "max_stores": 82, "max_command_bytes": 246, "edit_cursor_bits": 8,
              "public_selector_reads": 82, "online_task_reads": 0, "private_data_scratch_bits": 0,
              "task_provisioning": "state only; code public and fixed before task instance",
              "word_request_cap": 59}
    (args.out / "config.json").write_text(json.dumps(config, separators=(',', ':')) + '\n')
    for filename, run in (("word.json", word_checks), ("circuit.json", circuit_checks)):
        (args.out / filename).write_text(json.dumps(run(), separators=(',', ':')) + '\n')
    (args.out / "execution.json").write_text(json.dumps({"completed": True, "seconds": time.monotonic()-start}) + '\n')
    files = [f for f in args.out.rglob('*') if f.is_file() and not any(x.startswith('._') for x in f.parts)]
    hashes = {str(f.relative_to(args.out)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    (args.out / "sha256.json").write_text(json.dumps(hashes, indent=2) + '\n')
    assert sum(f.stat().st_size for f in files) + (args.out / 'sha256.json').stat().st_size <= 16*1024**2
    signal.alarm(0)
    print(json.dumps({"completed": True, "seconds": time.monotonic()-start, "out": str(args.out.resolve())}))


if __name__ == '__main__':
    main()
