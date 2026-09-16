"""Independent audit of an LN-141 irreversible-trajectory screen run. Standard library only.

Recomputes, from the saved public tables and without importing the screen:
hashes; the in-place step for every state; role variance, image sizes and maximum
fibers; the uniform-prior maximum repair spread; record-and-invert costs; the
minimal-automaton class count; and black-pebbling minima for one and two rounds
with a separate depth-first search. Reports discrepancies rather than fixing them.
"""

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path

WORDS, WBITS, ALPHABET = 4, 4, 16
NSTATES = ALPHABET ** WORDS
INPUTS, ROLES = 4, 2


def step(tab, state, rounds):
    words = [(state >> (WBITS * i)) & (ALPHABET - 1) for i in range(WORDS)]
    for _ in range(rounds):
        for i in range(WORDS):
            words[i] = tab[words[(i + 1) % WORDS]][words[i]]
    return sum(w << (WBITS * i) for i, w in enumerate(words))


def full_map(tab, rounds):
    return [step(tab, s, rounds) for s in range(NSTATES)]


def partition_classes(F, rounds):
    cls = [0] * NSTATES
    n = 1
    while True:
        sig = {}
        new = [0] * NSTATES
        for s in range(NSTATES):
            key = (cls[s],) + tuple((F[r][x][s] & (ALPHABET - 1), cls[F[r][x][s]]) for r in range(ROLES) for x in range(INPUTS))
            new[s] = sig.setdefault(key, len(sig))
        if len(sig) == n:
            return n
        cls, n = new, len(sig)


def dag(rounds, roles):
    preds = {i: () for i in range(WORDS)}
    finals = {}
    nid = WORDS
    for role in roles:
        cur = list(range(WORDS))
        for _ in range(rounds):
            new = cur.copy()
            for i in range(WORDS):
                nb = new[0] if i == WORDS - 1 else cur[i + 1]
                preds[nid] = (cur[i], nb)
                new[i] = nid
                nid += 1
            cur = new
        finals[role] = cur
    return preds, finals


def reachable(preds, goal_nodes, emit, capacity):
    """Breadth-first over (held-set, emitted) with in-place overwrite and lost sources."""
    start = frozenset(range(WORDS))
    goal = set(goal_nodes)
    seen = {(start, emit is None)}
    q = deque(seen)
    while q:
        held, flag = q.popleft()
        if flag and goal <= held:
            return True
        for v in preds:
            if v in held:
                nxt = (held - {v}, flag)
            elif v < WORDS or any(p not in held for p in preds[v]):
                continue
            else:
                nflag = flag or v == emit
                options = []
                if len(held) < capacity:
                    options.append((held | {v}, nflag))
                for p in set(preds[v]):
                    options.append(((held - {p}) | {v}, nflag))
                for nxt in options:
                    if nxt not in seen:
                        seen.add(nxt); q.append(nxt)
                continue
            if nxt not in seen:
                seen.add(nxt); q.append(nxt)
    return False


def min_cells(preds, goal_nodes, emit):
    for c in range(WORDS, WORDS + 9):
        if reachable(preds, goal_nodes, emit, c):
            return c
    return None


def audit(directory):
    report = {"hash_mismatches": [], "checks": []}
    for name, digest in json.loads((directory / "sha256.json").read_text()).items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            report["hash_mismatches"].append(name)
    tables = json.loads((directory / "tables.json").read_text())
    results = json.loads((directory / "results.json").read_text())
    for res in results:
        tab = tables[res["variant"]]
        rounds = res["rounds"]
        F = [[full_map(tab[r][x], rounds) for x in range(INPUTS)] for r in range(ROLES)]
        check = {"variant": res["variant"], "rounds": rounds, "ok": True, "notes": []}

        def expect(name, got, want, tol=0.0):
            if isinstance(want, float) or isinstance(got, float):
                good = abs(got - want) <= tol
            else:
                good = got == want
            if not good:
                check["ok"] = False
                check["notes"].append(f"{name}: audit {got} vs saved {want}")

        diff = sum(sum(1 for s in range(NSTATES) if F[0][x][s] != F[1][x][s]) for x in range(INPUTS)) / (INPUTS * NSTATES)
        expect("state_differs_uniform", diff, res["role_variance"]["state_differs_uniform"], 1e-12)
        for r in range(ROLES):
            for x in range(INPUTS):
                counts = {}
                for v in F[r][x]:
                    counts[v] = counts.get(v, 0) + 1
                expect(f"image r{r}x{x}", len(counts), res["images"][f"r{r}x{x}"]["image_size"])
                expect(f"fiber r{r}x{x}", max(counts.values()), res["images"][f"r{r}x{x}"]["max_fiber"])
        spread = 0
        for x in range(INPUTS):
            fibers = {}
            for s in range(NSTATES):
                fibers.setdefault(F[1][x][s], set()).add(F[0][x][s])
            spread = max(spread, max(len(v) for v in fibers.values()))
        expect("uniform max_fiber_spread", spread, res["repair_caller_state_after_owner_step_uniform_prior"]["max_fiber_spread"])
        worst = 0
        for r in range(ROLES):
            for x in range(INPUTS):
                for b in range(ALPHABET):
                    row = tab[r][x][b]
                    fib = max(row.count(v) for v in set(row))
                    worst = max(worst, (fib - 1).bit_length())
        expect("witness_bits_worst_case", worst * WORDS * rounds, res["record_and_invert"]["witness_bits_worst_case"])
        expect("inverse_table_bits", 2 * ROLES * INPUTS * ALPHABET * ALPHABET * WBITS, res["record_and_invert"]["inverse_table_bits"])
        if rounds <= 2:
            expect("minimal_automaton_classes", partition_classes(F, rounds), res["minimal_automaton"]["classes"])
            p1, f1 = dag(rounds, (1,))
            expect("read_only_min_cells", min_cells(p1, list(range(WORDS)), f1[1][0]), res["pebbling"]["read_only_owner_answer_min_cells"])
            p0, f0 = dag(rounds, (0,))
            expect("intact_min_cells", min_cells(p0, f0[0], None), res["pebbling"]["intact_caller_step_min_cells"])
            pj, fj = dag(rounds, (1, 0))
            expect("joint_min_cells", min_cells(pj, fj[0], fj[1][0]), res["pebbling"]["joint_owner_answer_and_caller_state_min_cells"])
        else:
            check["notes"].append("minimal automaton and pebbling replayed only for rounds <= 2 in this audit")
        report["checks"].append(check)
    report["passed"] = not report["hash_mismatches"] and all(c["ok"] for c in report["checks"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("output path must not exist")
    report = audit(args.directory)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"passed": report["passed"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
