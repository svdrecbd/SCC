"""LN-131: exact counterexample screen for a handwritten editable word machine.

This is an implementation witness, not training or a non-malleability proof.
All task/policy logic is in mutable RAM. The host only executes generic opcodes.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import time

LE, ADD, MOV, CONST, CALL, JZ, JNZ, LOAD, DEC, LT, SHR1, AND1, EMIT, HALT, EQ, RET = range(16)
ARMS = ("intact", "selective", "global", "benign")
MASK = 65535


def ins(op, d=0, a=0, b=0):
    return op << 12 | d << 8 | a << 4 | b


def program():
    return [
        ins(ADD, 4, 0, 0), ins(ADD, 4, 4, 1), ins(CONST, 5) | 3,
        ins(CONST, 7), ins(MOV, 8, 5), ins(MOV, 9, 4), ins(CALL) | 30,
        ins(JZ, 10) | 10, ins(LOAD, 7, 5, 2), ins(MOV, 7, 7),
        ins(DEC, 5), ins(CONST, 15) | 4, ins(LT, 10, 5, 15),
        ins(JNZ, 10) | 4, ins(SHR1, 0, 7), ins(AND1, 12, 7),
        ins(MOV, 8, 2), ins(MOV, 9, 3), ins(CALL) | 30,
        ins(MOV, 13, 10), ins(CONST, 14) | 2, ins(JZ, 13) | 23,
        ins(MOV, 14, 12), ins(EMIT, 14), ins(HALT),
        *([ins(MOV, 6, 6)] * 5), ins(EQ, 10, 8, 9), ins(RET),
    ]


def edit_command(address, value):
    # Generic STORE_IMMEDIATE: opcode8, address8, immediate16; fixed before Z.
    return bytes((1, address)) + value.to_bytes(2, "big")


EDITS = {
    "intact": None,
    "selective": edit_command(18, ins(LE, 10, 8, 9)),
    "global": edit_command(30, ins(LE, 10, 8, 9)),
    "benign": edit_command(27, ins(MOV, 6, 0)),
}


class Machine:
    def __init__(self, words):
        if len(words) != 64 or any(not 0 <= w <= MASK for w in words):
            raise ValueError("expected 64 unsigned 16-bit words")
        self.ram = words.copy()
        self.pc = 0

    def edit(self, command):
        if len(command) != 4 or command[0] != 1 or command[1] > 64:
            raise ValueError("invalid one-word edit")
        value = int.from_bytes(command[2:], "big")
        if command[1] == 64:
            self.pc = value
        else:
            self.ram[command[1]] = value

    def execute(self, cap, trace=None):
        # Trace and returned observations are external instrumentation, never
        # accessible to the machine. No task-dependent host value enters here.
        ram = self.ram
        emitted = None
        for step in range(1, cap + 1):
            if not 0 <= self.pc < 64:
                raise ValueError("PC outside RAM")
            word = ram[self.pc]
            if trace is not None:
                trace.append([self.pc, word, ram.copy()])
            self.pc = (self.pc + 1) & MASK
            op, d, a, b = word >> 12, (word >> 8) & 15, (word >> 4) & 15, word & 15
            rd, ra, rb = 48 + d, 48 + a, 48 + b
            if op == LE:
                ram[rd] = int(ram[ra] <= ram[rb])
            elif op == ADD:
                ram[rd] = (ram[ra] + ram[rb]) & MASK
            elif op == MOV:
                ram[rd] = ram[ra]
            elif op == CONST:
                ram[rd] = word & 255
            elif op == CALL:
                ram[59] = self.pc
                self.pc = word & 4095
            elif op in (JZ, JNZ):
                if (ram[rd] == 0) == (op == JZ):
                    self.pc = word & 255
            elif op == LOAD:
                address = b * 16 + ram[ra]
                if not 0 <= address < 64:
                    raise ValueError("LOAD outside RAM")
                ram[rd] = ram[address]
            elif op == DEC:
                ram[rd] = (ram[rd] - 1) & MASK
            elif op == LT:
                ram[rd] = int(ram[ra] < ram[rb])
            elif op == SHR1:
                ram[rd] = ram[ra] >> 1
            elif op == AND1:
                ram[rd] = ram[ra] & 1
            elif op == EMIT:
                if emitted is not None:
                    raise ValueError("more than one emission in this invocation")
                emitted = ram[rd]
            elif op == HALT:
                return [emitted, ram[61], ram[48], ram[60], step]
            elif op == EQ:
                ram[rd] = int(ram[ra] == ram[rb])
            elif op == RET:
                self.pc = ram[59]
        raise TimeoutError("instruction cap exceeded")

    def request(self, symbol, caller, owner, cap, trace=None):
        self.ram[49:52] = [symbol, caller, owner]
        self.pc = 0
        return self.execute(cap, trace)


def table_for(table_id):
    return [(table_id >> (2 * i)) & 3 for i in range(4)]


def make_machine(table_id, q, arm):
    ram = program() + table_for(table_id) + [0] * 28
    ram[48] = q
    machine = Machine(ram)
    if EDITS[arm] is not None:
        machine.edit(EDITS[arm])
    return machine


def check():
    rows, streams, traces = [], [], {}
    for table_id in range(256):
        for q in range(2):
            for symbol in range(2):
                for caller in range(2):
                    for owner in range(2):
                        results = []
                        for arm in ARMS:
                            machine = make_machine(table_id, q, arm)
                            trace = [] if (table_id, q, symbol, caller, owner) == (228, 1, 1, 0, 1) else None
                            results.append(machine.request(symbol, caller, owner, 128 if arm == "global" else 59, trace))
                            assert machine.ram[32:36] == table_for(table_id)
                            if trace is not None:
                                traces[arm] = trace
                        rows.append([table_id, q, symbol, caller, owner, results])
        machines = [make_machine(table_id, 0, arm) for arm in ARMS]
        for t in range(32):
            # Four repeats of every (symbol, caller, owner) triple. Persistent
            # q and scratch across all 32 requests; no per-request RAM reset.
            symbol, caller, owner = (t // 4) % 2, (t // 2) % 2, t % 2
            results = [m.request(symbol, caller, owner, 128 if arm == "global" else 59)
                       for arm, m in zip(ARMS, machines)]
            assert all(m.ram[32:36] == table_for(table_id) for m in machines)
            streams.append([table_id, t, symbol, caller, owner, results])
    return {"arms": ARMS, "result_fields": ["emitted", "admission", "next_q", "answer", "steps"],
            "case_fields": ["table_id", "q", "symbol", "caller", "owner", "results"],
            "stream_fields": ["table_id", "t", "symbol", "caller", "owner", "results"],
            "cases": rows, "streams": streams, "traces": traces}


def save(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("120-second wall cap")))
    signal.alarm(120)
    source = args.out / "source"
    source.mkdir()
    for name in ("screen_word_machine.py", "audit_word_machine.py"):
        shutil.copyfile(root / "scripts" / name, source / name)
    shutil.copyfile(root / "tests/test_word_machine.py", source / "test_word_machine.py")
    notes = (root / "labnotes.md").read_text()
    (args.out / "plan.md").write_text(notes[notes.index('<a id="ln-131"></a>'):notes.index("## Supporting-record index")])
    save(args.out / "config.json", {
        "ram_words": 64, "word_bits": 16, "pc_bits": 16, "total_state_bits": 1040,
        "request_step_cap": 59, "global_diagnostic_cap": 128,
        "edit_description_bits": 32, "edit_step_cap": 16, "edit_scratch_cap_bits": 16,
        "witness_edit_steps": 1, "witness_scratch_bits": 0, "queries": 0,
        "edits_hex": {k: v.hex() if v else None for k, v in EDITS.items()},
        "program_words": program(), "table_ids": list(range(256)),
        "stream_length": 32, "stream_schedule": "symbol=(t//4)%2, caller=(t//2)%2, owner=t%2",
        "wall_cap_seconds": 120, "output_cap_bytes": 16 * 1024**2,
        "python": platform.python_version(), "machine": platform.platform(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    })
    save(args.out / "outputs.json", check())
    save(args.out / "receipt.json", {"completed": True, "seconds": time.monotonic() - started})
    files = [p for p in args.out.rglob("*") if p.is_file() and not p.name.startswith("._")]
    hashes = {str(p.relative_to(args.out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    save(args.out / "sha256.json", hashes)
    assert sum(p.stat().st_size for p in files) + (args.out / "sha256.json").stat().st_size <= 16 * 1024**2
    signal.alarm(0)
    print(json.dumps({"output": str(args.out.resolve()), "seconds": time.monotonic() - started, "completed": True}))


if __name__ == "__main__":
    main()
