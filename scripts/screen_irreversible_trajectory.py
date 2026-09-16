"""LN-141: exact screen of the role-variant in-place irreversible-trajectory candidate.

Two principals share one N=16-bit live state of four nibbles. Each request
(x in 0..3, role r in {0,1}) updates the state in place, word by word, with a
public role-keyed table T_r[x][neighbour][word]; the answer is nibble 0 after
the update. There is no refusal, gate, checker, wipe rule or hidden decoder.
The candidate obstruction is a resource: obtaining the owner-trajectory answer on
a caller request while keeping the caller trajectory intact.

This screen measures, exactly over all 65,536 states where stated:
  role variance, image/fiber structure, durable divergence and its effect on
  reference-scored utility, the best output-only/reader-substitution guess,
  the side-information needed to repair the caller state after an in-place
  owner step (over ALL side-information functions of a given width), the
  non-injectivity of the task's minimal automaton, black-pebbling numbers for
  read-only and joint two-role evaluation with destructible sources, and the
  exact scan/table costs of the record-and-invert attack for this table step.

It is a handwritten finite machine with supplied tables: not learning, not a
lower bound over algebraic shortcuts, and not a claim of catastrophic cognition
failure. Standard library plus NumPy only; no training, GPU or provider access.
"""

import argparse
from collections import deque
import hashlib
import itertools
import json
from pathlib import Path
import platform
import random
import shutil
import signal
import subprocess
import time

import numpy as np

WORDS, WBITS, ALPHABET = 4, 4, 16
NSTATES = ALPHABET ** WORDS  # 65536
INPUTS, ROLES = 4, 2
VARIANTS = ("function", "nearperm")
ROUNDS = (1, 2, 3, 4)
BURN_IN = 16          # steps of exact weight propagation before measuring
HORIZON = 32          # post-attack requests scored against the reference
SAMPLES = 4096        # sampled trajectories for the horizon measurements
SEED = 20260916


def make_tables(variant, seed):
    """T[r][x][b] is a 16-entry map a -> T_r(a, b, x). Public, task-independent."""
    rng = random.Random(f"{variant}-{seed}")
    tables = []
    for _r in range(ROLES):
        per_x = []
        for _x in range(INPUTS):
            per_b = []
            for _b in range(ALPHABET):
                if variant == "function":
                    row = [rng.randrange(ALPHABET) for _ in range(ALPHABET)]
                else:  # random permutation with exactly one merged output pair
                    row = list(range(ALPHABET))
                    rng.shuffle(row)
                    i, j = rng.sample(range(ALPHABET), 2)
                    row[i] = row[j]
                per_b.append(row)
            per_x.append(per_b)
        tables.append(per_x)
    return tables


def unpack(states):
    return [(states >> (WBITS * i)) & (ALPHABET - 1) for i in range(WORDS)]


def pack(words):
    out = np.zeros_like(words[0])
    for i, w in enumerate(words):
        out |= w << (WBITS * i)
    return out


def step_table(tables, r, x, rounds):
    """Vectorised F_r(., x): in-place sequential ring update, `rounds` passes."""
    tab = np.array(tables[r][x], dtype=np.int64)  # [b][a]
    words = unpack(np.arange(NSTATES, dtype=np.int64))
    for _ in range(rounds):
        for i in range(WORDS):
            words[i] = tab[words[(i + 1) % WORDS], words[i]]
    return pack(words)


def answer(states):
    return states & (ALPHABET - 1)


def stationary_weights(F):
    """Exact weight propagation from uniform for BURN_IN steps, uniform (x, r)."""
    w = np.full(NSTATES, 1.0 / NSTATES)
    for _ in range(BURN_IN):
        nxt = np.zeros(NSTATES)
        for r in range(ROLES):
            for x in range(INPUTS):
                np.add.at(nxt, F[r][x], w / (ROLES * INPUTS))
        w = nxt
    return w


def side_information_curve(source_post, target_values, weights):
    """Best success probability, for each side-information width s, of recovering
    target_values(B) from (source_post(B), h(B)) over ALL h with 2^s values.
    Exact: within each source fiber the attacker can single out at most 2^s
    distinct targets, so the optimum keeps the 2^s heaviest target values."""
    order = np.lexsort((target_values, source_post))
    sp, tv, wt = source_post[order], target_values[order], weights[order]
    key = sp.astype(np.int64) * NSTATES + tv
    uniq, idx = np.unique(key, return_index=True)
    mass = np.add.reduceat(wt, idx)
    fibers = uniq // NSTATES
    curve, spreads = [], []
    fib_uniq, fib_idx = np.unique(fibers, return_index=True)
    groups = np.split(mass, fib_idx[1:])
    max_spread = max(len(g) for g in groups)
    for s in range(0, 17):
        keep = 2 ** s
        total = 0.0
        for g in groups:
            if len(g) <= keep:
                total += g.sum()
            else:
                total += np.sort(g)[-keep:].sum()
        curve.append(float(total))
    return {"success_by_side_bits": curve, "max_fiber_spread": int(max_spread),
            "bits_for_exact": int(np.ceil(np.log2(max_spread))) if max_spread > 1 else 0}


def moore_partition(F, ans_of_next):
    """Minimal-automaton classes: refine by (answer, class of successor) on all 8 symbols."""
    cls = np.zeros(NSTATES, dtype=np.int64)
    n = 1
    while True:
        sig = [cls]
        for r in range(ROLES):
            for x in range(INPUTS):
                sig.append(ans_of_next[r][x])
                sig.append(cls[F[r][x]])
        _, new = np.unique(np.stack(sig, 1), axis=0, return_inverse=True)
        new = new.reshape(-1)
        m = int(new.max()) + 1
        if m == n:
            return new, m
        cls, n = new, m


def minimal_automaton_injectivity(F, cls, nclasses):
    merges = 0
    for r in range(ROLES):
        for x in range(INPUTS):
            rep = np.zeros(nclasses, dtype=np.int64) - 1
            np.maximum.at(rep, cls, np.arange(NSTATES))  # one representative per class
            succ = cls[F[r][x][rep]]
            merges += nclasses - len(np.unique(succ))
    return merges


def dag(rounds, roles):
    """Nodes: 0..3 sources; then per role, per round, per word an update node.
    Predecessors follow the sequential in-place ring: word i at round k reads its
    own previous value and the neighbour's value as it stands at that moment."""
    preds = {i: () for i in range(WORDS)}
    finals = {}
    nid = WORDS
    for role in roles:
        cur = list(range(WORDS))
        for _k in range(rounds):
            new = cur.copy()
            for i in range(WORDS):
                preds[nid] = (cur[i], new[(i + 1) % WORDS] if i == WORDS - 1 else cur[(i + 1) % WORDS])
                new[i] = nid
                nid += 1
            # after the pass, word i holds new[i]; the neighbour read for i<3 was the
            # not-yet-updated cur[i+1]; for i==3 it was the already-updated new[0]
            cur = new
        finals[role] = cur
    return preds, finals


def pebble_min_cells(preds, must_hold_final, emit_node, capacity, in_place=True):
    """Black pebbling with destructible sources and optional in-place overwrite.
    State: bitmask of held nodes, flag whether emit_node has ever been held.
    Sources start held; a removed source can never be re-pebbled.
    Returns True if a configuration holding all must_hold_final with the emit flag
    set is reachable using at most `capacity` cells."""
    nodes = sorted(preds)
    start = sum(1 << i for i in range(WORDS))
    goal = sum(1 << v for v in must_hold_final)
    seen = {(start, emit_node is None)}
    queue = deque([(start, emit_node is None)])
    while queue:
        held, flag = queue.popleft()
        if flag and held & goal == goal:
            return True
        count = bin(held).count("1")
        for v in nodes:
            bit = 1 << v
            if held & bit:
                nxt = (held & ~bit, flag)
                if nxt not in seen:
                    seen.add(nxt); queue.append(nxt)
                continue
            if v < WORDS:
                continue  # lost source
            ps = preds[v]
            if any(not held & (1 << p) for p in ps):
                continue
            nflag = flag or v == emit_node
            if count < capacity:
                nxt = (held | bit, nflag)
                if nxt not in seen:
                    seen.add(nxt); queue.append(nxt)
            if in_place:
                for p in set(ps):
                    nxt = ((held & ~(1 << p)) | bit, nflag)
                    if nxt not in seen:
                        seen.add(nxt); queue.append(nxt)
    return False


def pebbling(rounds):
    out = {}
    # read-only single-role evaluation of the owner answer: sources must survive,
    # so forbid overwriting them by treating them as non-removable: emulate by
    # requiring all sources held at the goal.
    preds1, finals1 = dag(rounds, roles=(1,))
    ans1 = finals1[1][0]
    for c in range(WORDS, WORDS + 9):
        if pebble_min_cells(preds1, must_hold_final=list(range(WORDS)), emit_node=ans1, capacity=c):
            out["read_only_owner_answer_min_cells"] = c
            break
    else:
        out["read_only_owner_answer_min_cells"] = None
    # intact in-place caller step
    preds0, finals0 = dag(rounds, roles=(0,))
    for c in range(WORDS, WORDS + 9):
        if pebble_min_cells(preds0, must_hold_final=finals0[0], emit_node=None, capacity=c):
            out["intact_caller_step_min_cells"] = c
            break
    # joint: emit owner answer at some time, end holding the full caller post-state
    predsj, finalsj = dag(rounds, roles=(1, 0))
    out["nodes"] = len(predsj)
    if len(predsj) > 28:
        out["joint_owner_answer_and_caller_state_min_cells"] = "not searched (state space)"
        return out
    for c in range(WORDS, WORDS + 9):
        if pebble_min_cells(predsj, must_hold_final=finalsj[0], emit_node=finalsj[1][0], capacity=c):
            out["joint_owner_answer_and_caller_state_min_cells"] = c
            break
    else:
        out["joint_owner_answer_and_caller_state_min_cells"] = None
    return out


def record_and_invert_costs(tables, rounds):
    """Exact costs of the record-and-invert repair for this table step."""
    worst_bits = 0
    for r in range(ROLES):
        for x in range(INPUTS):
            for b in range(ALPHABET):
                row = tables[r][x][b]
                fib = max(row.count(v) for v in set(row))
                worst_bits = max(worst_bits, int(np.ceil(np.log2(fib))) if fib > 1 else 0)
    updates = WORDS * rounds
    table_bits = ROLES * INPUTS * ALPHABET * ALPHABET * WBITS
    return {
        "word_updates_per_request": updates,
        "intact_table_lookups_per_request": updates,
        "witness_bits_worst_case": worst_bits * updates,
        "scan_lookups_per_request": updates * ALPHABET,
        "scan_time_factor": ALPHABET,
        "public_table_bits": table_bits,
        "inverse_table_bits": table_bits * 2,
    }


def horizon_measurements(F, weights, rng):
    """Sampled post-attack utility against the true-role reference for three attacks."""
    starts = rng.choice(NSTATES, size=SAMPLES, p=weights)
    xs = rng.integers(0, INPUTS, size=(SAMPLES, HORIZON + 1))
    rs = rng.integers(0, ROLES, size=(SAMPLES, HORIZON + 1))
    rs[:, 0] = 0  # the attacked request is a caller request
    S = np.array([0, 1])  # selected forbidden inputs for the selective attack

    Fa = np.array(F)  # [role][input][state]

    def run(policy):
        ref = starts.copy(); att = starts.copy()
        agree, equal = [], []
        owner_hit = None
        for t in range(HORIZON + 1):
            x, r = xs[:, t], rs[:, t]
            ref_next = Fa[r, x, ref]
            r_att = policy(t, x, r)
            att_next = Fa[r_att, x, att]
            if t == 0:
                owner_ans = Fa[1, x, ref] & (ALPHABET - 1)
                owner_hit = float(np.mean((att_next & (ALPHABET - 1)) == owner_ans))
            ref, att = ref_next, att_next
            agree.append(float(np.mean((ref & (ALPHABET - 1)) == (att & (ALPHABET - 1)))))
            equal.append(float(np.mean(ref == att)))
        return {"owner_answer_correct_at_attack": owner_hit,
                "answer_agreement_by_step": agree, "state_equality_by_step": equal,
                "mean_answer_agreement_after_attack": float(np.mean(agree[1:]))}

    return {
        "one_owner_step_then_honest": run(lambda t, x, r: np.where(t == 0, 1, r)),
        "constant_owner": run(lambda t, x, r: np.ones_like(r)),
        "selective_owner_on_S": run(lambda t, x, r: np.where((r == 0) & np.isin(x, S), 1, r)),
        "benign_honest": run(lambda t, x, r: r),
    }


def analyse(variant, rounds, tables):
    F = [[step_table(tables, r, x, rounds) for x in range(INPUTS)] for r in range(ROLES)]
    ans_next = [[answer(F[r][x]) for x in range(INPUTS)] for r in range(ROLES)]
    w = stationary_weights(F)
    out = {"variant": variant, "rounds": rounds}
    out["reachable_support_after_burn_in"] = int(np.count_nonzero(w))
    out["state_entropy_bits_after_burn_in"] = float(-(w[w > 0] * np.log2(w[w > 0])).sum())
    out["role_variance"] = {
        "state_differs_uniform": float(np.mean([np.mean(F[0][x] != F[1][x]) for x in range(INPUTS)])),
        "state_differs_weighted": float(np.mean([(w * (F[0][x] != F[1][x])).sum() for x in range(INPUTS)])),
        "answer_differs_weighted": float(np.mean([(w * (ans_next[0][x] != ans_next[1][x])).sum() for x in range(INPUTS)])),
    }
    out["images"] = {f"r{r}x{x}": {"image_size": int(len(np.unique(F[r][x]))),
                                   "max_fiber": int(np.bincount(F[r][x], minlength=NSTATES).max())}
                     for r in range(ROLES) for x in range(INPUTS)}
    # repair after an in-place owner step: recover F0(B) from (F1(B), h(B)); per x, averaged
    curves = [side_information_curve(F[1][x], F[0][x], w) for x in range(INPUTS)]
    out["repair_caller_state_after_owner_step"] = {
        "success_by_side_bits_mean": [float(np.mean([c["success_by_side_bits"][s] for c in curves])) for s in range(17)],
        "max_fiber_spread": max(c["max_fiber_spread"] for c in curves),
        "bits_for_exact_worst_x": max(c["bits_for_exact"] for c in curves),
    }
    # uniform-prior version, for contrast with the reachable distribution
    u = np.full(NSTATES, 1.0 / NSTATES)
    ucurves = [side_information_curve(F[1][x], F[0][x], u) for x in range(INPUTS)]
    out["repair_caller_state_after_owner_step_uniform_prior"] = {
        "success_by_side_bits_mean": [float(np.mean([c["success_by_side_bits"][s] for c in ucurves])) for s in range(17)],
        "max_fiber_spread": max(c["max_fiber_spread"] for c in ucurves),
    }
    # output-only / pure reader substitution: guess owner answer from caller post-state, no side info
    guess = [side_information_curve(F[0][x], ans_next[1][x], w) for x in range(INPUTS)]
    out["owner_answer_from_caller_post_state"] = {
        "best_guess_no_side_info": float(np.mean([g["success_by_side_bits"][0] for g in guess])),
        "chance": 1.0 / ALPHABET,
        "note": "side-information widths >= 4 bits trivially store the answer itself; only s=0 is meaningful here",
    }
    cls, n = moore_partition(F, ans_next)
    out["minimal_automaton"] = {"classes": int(n),
                                "class_merges_under_some_symbol": int(minimal_automaton_injectivity(F, cls, n))}
    out["pebbling"] = pebbling(rounds)
    out["record_and_invert"] = record_and_invert_costs(tables, rounds)
    rng = np.random.default_rng(SEED + rounds)
    out["horizon"] = horizon_measurements(F, w, rng)
    return out


def save(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wall-cap", type=int, default=600)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    started = time.monotonic()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("wall cap")))
    signal.alarm(args.wall_cap)
    source = args.out / "source"
    source.mkdir()
    for name in ("screen_irreversible_trajectory.py", "audit_irreversible_trajectory.py"):
        shutil.copyfile(root / "scripts" / name, source / name)
    shutil.copyfile(root / "tests/test_irreversible_trajectory.py", source / "test_irreversible_trajectory.py")
    notes = (root / "labnotes.md").read_text()
    (args.out / "plan.md").write_text(notes[notes.index('<a id="ln-141"></a>'):notes.index("## Supporting-record index")])
    tables = {v: make_tables(v, SEED) for v in VARIANTS}
    save(args.out / "tables.json", tables)
    save(args.out / "config.json", {
        "words": WORDS, "word_bits": WBITS, "state_bits": WORDS * WBITS, "inputs": INPUTS, "roles": ROLES,
        "variants": VARIANTS, "rounds": ROUNDS, "burn_in_steps": BURN_IN, "horizon": HORIZON,
        "samples": SAMPLES, "seed": SEED, "answer": "nibble 0 after update", "chance": 1 / ALPHABET,
        "wall_cap_seconds": args.wall_cap, "output_cap_bytes": 16 * 1024**2,
        "python": platform.python_version(), "numpy": np.__version__, "machine": platform.platform(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    })
    results = []
    for variant in VARIANTS:
        for rounds in ROUNDS:
            t0 = time.monotonic()
            res = analyse(variant, rounds, tables[variant])
            res["seconds"] = time.monotonic() - t0
            results.append(res)
            print(json.dumps({"variant": variant, "rounds": rounds, "seconds": res["seconds"]}), flush=True)
    save(args.out / "results.json", results)
    save(args.out / "receipt.json", {"completed": True, "seconds": time.monotonic() - started})
    files = [p for p in args.out.rglob("*") if p.is_file() and not p.name.startswith("._")]
    hashes = {str(p.relative_to(args.out)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    save(args.out / "sha256.json", hashes)
    assert sum(p.stat().st_size for p in files) <= 16 * 1024**2
    signal.alarm(0)
    print(json.dumps({"output": str(args.out.resolve()), "seconds": time.monotonic() - started, "completed": True}))


if __name__ == "__main__":
    main()
