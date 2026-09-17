"""LN-143: wide-word irreversible-trajectory candidate with a compact one-way step.

State: four w-bit words, N=4w. Request (x in 0..3, role r in {0,1}). In-place ring
update, word by word: B[i] <- T(B[i]; key=(B[(i+1)%4], x, r)) where
T(a; k) = E_k(a) XOR a is a Davies-Meyer compression of a SPECK-style ARX block
cipher E_k with R rounds on two w/2-bit halves. Everything is public and editable;
the interpreter is add/rotate/xor. No gate, refusal, checker or wipe rule.

Measured per (w, R):
  exact fiber statistics of T for sampled key contexts over all 2^w inputs
  (witness bits, image fraction); sampled state panels (divergence, utility
  against the true-role reference after one owner step, constant owner,
  selective owner, honest control); an operation ledger for intact execution,
  full scan inversion and preimage tables; the half-word enumeration shortcut
  (exact for R=1 by construction, measured for larger R); and, if python-sat is
  available, SAT preimage solving with conflict counts and full preimage
  enumeration cross-checked against the scan.

This is an exact/sampled measurement of a handwritten machine. SAT conflict
counts are a structure signal at fixed w, not a lower bound; nothing here is a
learned mechanism, an alignment result, or catastrophic cognition failure.
"""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import random
import shutil
import signal
import time
from importlib.metadata import version

import numpy as np

try:
    from pysat.solvers import Cadical195 as Solver
    HAVE_SAT = True
except Exception:  # pragma: no cover
    HAVE_SAT = False

WORDS, INPUTS, ROLES = 4, 4, 2
SEED = 20260916
SCHEMA_VERSION = 2


def params(w):
    h = w // 2
    return {"w": w, "h": h, "alpha": h // 2 + 1, "beta": 2, "mask": (1 << h) - 1, "wmask": (1 << w) - 1}


def constants(R, h, seed=SEED):
    rng = random.Random(f"wide-{h}-{R}-{seed}")
    return [rng.randrange(1 << h) for _ in range(R)]


def rotl(v, s, h, mask):
    s %= h
    return ((v << s) | (v >> (h - s))) & mask


def rotr(v, s, h, mask):
    return rotl(v, h - (s % h), h, mask)


def round_keys(b, x, r, R, p, C):
    """Public key schedule from neighbour word b, input x and role r."""
    h, mask = p["h"], p["mask"]
    bl, bh = b & mask, (b >> h) & mask
    keys = []
    for j in range(R):
        k = rotl(bl, j + 1, h, mask) ^ rotl(bh, 2 * j + 1, h, mask) ^ (((x << 1) | r) * 0x1D) ^ C[j]
        keys.append(k & mask)
    return keys


def T(a, b, x, r, R, p, C):
    """Davies-Meyer step on w-bit words. Vectorised over numpy int64 arrays."""
    h, mask, al, be = p["h"], p["mask"], p["alpha"], p["beta"]
    L, Rh = a & mask, (a >> h) & mask
    for k in round_keys(b, x, r, R, p, C):
        L = (rotr(L, al, h, mask) + Rh) & mask
        L = L ^ k
        Rh = rotl(Rh, be, h, mask) ^ L
    e = (Rh << h) | L
    return e ^ a


def step(B, x, r, R, p, C):
    """B: list of four int64 arrays. In-place sequential ring update."""
    B = list(B)
    for i in range(WORDS):
        B[i] = T(B[i], B[(i + 1) % WORDS], x, r, R, p, C)
    return B


def fiber_statistics(R, p, C, contexts, rng):
    w = p["w"]
    a = np.arange(1 << w, dtype=np.int64)
    rows = []
    for _ in range(contexts):
        b, x, r = int(rng.integers(0, 1 << w)), int(rng.integers(0, INPUTS)), int(rng.integers(0, ROLES))
        v = T(a, np.int64(b), x, r, R, p, C)
        counts = np.bincount(v, minlength=1 << w)
        fib_of_a = counts[v]
        rows.append({"b": b, "x": x, "r": r, "image_size": int(np.count_nonzero(counts)),
                     "max_fiber": int(counts.max()),
                     "mean_witness_bits": float(np.mean(np.log2(fib_of_a))),
                     "worst_witness_bits": int(np.ceil(np.log2(counts.max())))})
    return {"contexts": rows,
            "image_fraction_mean": float(np.mean([c["image_size"] for c in rows]) / (1 << w)),
            "max_fiber_over_contexts": max(c["max_fiber"] for c in rows),
            "mean_witness_bits": float(np.mean([c["mean_witness_bits"] for c in rows])),
            "worst_witness_bits": max(c["worst_witness_bits"] for c in rows),
            "random_function_reference_mean_witness_bits": "about 0.83 for a uniformly random map"}


def state_panels(R, p, C, samples, burn_in, horizon, rng):
    w, wmask = p["w"], p["wmask"]
    B0 = [rng.integers(0, 1 << w, size=samples, dtype=np.int64) for _ in range(WORDS)]
    xs = rng.integers(0, INPUTS, size=(samples, burn_in + horizon + 1))
    rs = rng.integers(0, ROLES, size=(samples, burn_in + horizon + 1))
    rs[:, burn_in] = 0  # attacked request is a caller request
    S = (0, 1)

    def eq(A, Bb):
        m = np.ones(samples, dtype=bool)
        for i in range(WORDS):
            m &= A[i] == Bb[i]
        return m

    def run(policy, forward_only=False):
        ref = [b.copy() for b in B0]; att = [b.copy() for b in B0]
        agree, equal, hit = [], [], None
        for t in range(burn_in + horizon + 1):
            x, r = xs[:, t], rs[:, t]
            ref_n = [np.empty(samples, dtype=np.int64) for _ in range(WORDS)]
            att_n = [np.empty(samples, dtype=np.int64) for _ in range(WORDS)]
            r_att = policy(t, x, r)
            forward_emitted = None
            if forward_only and t == burn_in:
                # Emit before any honest update; these saved answers belong to
                # the evaluator, not an attack bank retained during the update.
                forward_emitted = np.empty(samples, dtype=np.int64)
                for xx in range(INPUTS):
                    m = x == xx
                    if m.any():
                        forward_emitted[m] = T(att[0][m], att[1][m], xx, 1, R, p, C)
            for xx in range(INPUTS):
                for rr in range(ROLES):
                    m = (x == xx) & (r == rr)
                    if m.any():
                        out = step([b[m] for b in ref], xx, rr, R, p, C)
                        for i in range(WORDS):
                            ref_n[i][m] = out[i]
                    m2 = (x == xx) & (r_att == rr)
                    if m2.any():
                        out = step([b[m2] for b in att], xx, rr, R, p, C)
                        for i in range(WORDS):
                            att_n[i][m2] = out[i]
            if t == burn_in:
                owner = [np.empty(samples, dtype=np.int64) for _ in range(WORDS)]
                for xx in range(INPUTS):
                    m = x == xx
                    if m.any():
                        out = step([b[m] for b in ref], xx, 1, R, p, C)
                        owner[0][m] = out[0]
                emitted = att_n[0] if forward_emitted is None else forward_emitted
                hit = float(np.mean(emitted == owner[0]))
            ref, att = ref_n, att_n
            if t >= burn_in:
                agree.append(float(np.mean(ref[0] == att[0])))
                equal.append(float(np.mean(eq(ref, att))))
        return {"owner_answer_correct_at_attack": hit,
                "answer_agreement_after_attack_mean": float(np.mean(agree[1:])),
                "state_equality_at_steps_1_16_64_last": [equal[1], equal[min(16, len(equal) - 1)],
                                                         equal[min(64, len(equal) - 1)], equal[-1]],
                "answer_agreement_by_step_sparse": agree[::8]}

    return {
        "chance": 1.0 / (1 << w),
        "one_owner_step_then_honest": run(lambda t, x, r: np.where(t == burn_in, 1, r)),
        "constant_owner": run(lambda t, x, r: np.ones_like(r)),
        "selective_owner_on_S": run(lambda t, x, r: np.where((r == 0) & np.isin(x, S), 1, r)),
        "benign_honest": run(lambda t, x, r: r),
        "forward_only_owner_then_honest": run(lambda t, x, r: r, forward_only=True),
        "evidence": {"initial_state_words": [b.tolist() for b in B0],
                     "task_inputs": xs.tolist(), "roles": rs.tolist()},
    }


def ledger(R, p):
    w = p["w"]
    return {
        "model": "T-call schedule; not a complete primitive-instruction or peak-memory meter",
        "intact_T_calls_per_request": WORDS,
        "forward_only_T_calls_per_attacked_request": WORDS + 1,
        "forward_only_T_call_factor": (WORDS + 1) / WORDS,
        "ring_passes_per_request": 1,
        "cipher_rounds_per_T": R,
        "scan_forward_evaluations_per_word_update": 1 << w,
        "scan_T_calls_per_request": WORDS * (1 << w),
        "scan_time_factor": 1 << w,
        "half_enumeration_forward_evaluations": 1 << (w // 2),
        "preimage_table_bits_total": (1 << w) * INPUTS * ROLES * (1 << w) * w,
        "round_constant_bits": R * (w // 2),
        "live_state_bits": WORDS * w,
        "full_copy_scratch_bits": WORDS * w,
        "save_restore_extra_word_bits": w,
        "forward_only_extra_persistent_bits": 0,
        "forward_only_extra_workspace_beyond_common_T_bits": 0,
        "workspace_schedule": {
            "cipher_halves_bits": w,
            "materialized_round_keys_bits": R * (w // 2),
            "input_word": "remains in live bank until commit; needed for feed-forward",
            "emission": "consume T return before reusing workspace for honest update",
            "shared": "intact and attack reuse the same T workspace sequentially",
        },
        "peak_machine_bits": None,
        "primitive_instruction_budget": None,
        "admissibility": "not established under any additional deadline or storage cap",
        "unmetered": ["temporary arithmetic values", "control and instruction encoding",
                      "address/input/output registers", "Python/NumPy allocations"],
    }


def half_enumeration_attack(R, p, C, instances, rng):
    """For R=1 the DM equations reduce to one half-word unknown; enumerate 2^(w/2)
    candidates for the high half, derive the low half, verify. Measured for all R."""
    w, h, mask = p["w"], p["h"], p["mask"]
    al, be = p["alpha"], p["beta"]
    success, rows = 0, []
    for _ in range(instances):
        a = int(rng.integers(0, 1 << w)); b = int(rng.integers(0, 1 << w))
        x, r = int(rng.integers(0, INPUTS)), int(rng.integers(0, ROLES))
        v = int(T(np.int64(a), np.int64(b), x, r, R, p, C))
        vL, vR = v & mask, (v >> h) & mask
        k0 = round_keys(b, x, r, R, p, C)[0]
        Rh = np.arange(1 << h, dtype=np.int64)
        # R=1 derivation: L = vR ^ vL ^ rotl(Rh,be) ^ Rh ; check (rotr(L,al)+Rh)^k0 == vL ^ L
        L = vR ^ vL ^ rotl(Rh, be, h, mask) ^ Rh
        cand = (Rh << h) | L
        ok = T(cand, np.int64(b), x, r, R, p, C) == v
        success += int(ok.any())
        rows.append({"a": a, "b": b, "x": x, "r": r, "v": v, "found": bool(ok.any())})
    return {"instances": instances, "preimage_found_fraction": success / instances,
            "candidates_per_instance": 1 << h, "rows": rows}


class CNF:
    def __init__(self):
        self.n = 0
        self.clauses = []

    def var(self):
        self.n += 1
        return self.n

    def const(self, bit):
        v = self.var()
        self.clauses.append([v] if bit else [-v])
        return v

    def xor2(self, a, b):
        o = self.var()
        self.clauses += [[-a, -b, -o], [a, b, -o], [a, -b, o], [-a, b, o]]
        return o

    def and2(self, a, b):
        o = self.var()
        self.clauses += [[-o, a], [-o, b], [-a, -b, o]]
        return o

    def or2(self, a, b):
        o = self.var()
        self.clauses += [[o, -a], [o, -b], [-o, a, b]]
        return o

    def add(self, A, B):
        out, carry = [], None
        for i in range(len(A)):
            s = self.xor2(A[i], B[i])
            if carry is None:
                out.append(s); carry = self.and2(A[i], B[i])
            else:
                out.append(self.xor2(s, carry))
                carry = self.or2(self.and2(A[i], B[i]), self.and2(s, carry))
        return out

    def xor_const(self, A, k):
        return [self.xor2(a, self.const((k >> i) & 1)) for i, a in enumerate(A)]

    def xor_bits(self, A, B):
        return [self.xor2(a, b) for a, b in zip(A, B)]


def rot_bits(bits, s):  # rotl by s on a little-endian bit list
    h = len(bits); s %= h
    return [bits[(i - s) % h] for i in range(h)]


def sat_preimage(R, p, C, b, x, r, v, enumerate_all=False, conflict_budget=2_000_000):
    """CNF for T(a; b,x,r) = v with unknown a; returns solutions and solver stats."""
    h, al, be = p["h"], p["alpha"], p["beta"]
    cnf = CNF()
    L = [cnf.var() for _ in range(h)]; Rh = [cnf.var() for _ in range(h)]
    a_bits = L + Rh
    curL, curR = L, Rh
    for k in round_keys(b, x, r, R, p, C):
        curL = cnf.add(rot_bits(curL, h - (al % h)), curR)
        curL = cnf.xor_const(curL, k)
        curR = cnf.xor_bits(rot_bits(curR, be), curL)
    e_bits = curL + curR
    out = cnf.xor_bits(e_bits, a_bits)
    for i, o in enumerate(out):
        cnf.clauses.append([o] if (v >> i) & 1 else [-o])
    sols, conflicts, termination = [], 0, None
    with Solver(bootstrap_with=cnf.clauses) as s:
        while True:
            s.conf_budget(conflict_budget)
            ok = s.solve_limited()
            st = s.accum_stats()
            conflicts = st.get("conflicts", conflicts)
            if ok is None:
                termination = "budget_exhausted"
                break
            if ok is False:
                termination = "exhausted"
                break
            m = set(l for l in s.get_model() if l > 0)
            a = sum(1 << i for i, vb in enumerate(a_bits) if vb in m)
            sols.append(a)
            if not enumerate_all:
                termination = "solution_found"
                break
            s.add_clause([-vb if vb in m else vb for vb in a_bits])
    return {"solutions": sols, "termination": termination, "conflict_budget_per_solve": conflict_budget,
            "conflicts": int(conflicts), "vars": cnf.n, "clauses": len(cnf.clauses)}


def sat_series(R, p, C, instances, rng):
    w = p["w"]
    rows, agree = [], 0
    t0 = time.monotonic()
    for _ in range(instances):
        a = int(rng.integers(0, 1 << w)); b = int(rng.integers(0, 1 << w))
        x, r = int(rng.integers(0, INPUTS)), int(rng.integers(0, ROLES))
        v = int(T(np.int64(a), np.int64(b), x, r, R, p, C))
        res = sat_preimage(R, p, C, b, x, r, v, enumerate_all=True)
        scan = np.flatnonzero(T(np.arange(1 << w, dtype=np.int64), np.int64(b), x, r, R, p, C) == v)
        same = res["termination"] == "exhausted" and sorted(res["solutions"]) == sorted(int(s) for s in scan)
        agree += int(same)
        rows.append({"b": b, "x": x, "r": r, "v": v, "solutions": res["solutions"],
                     "termination": res["termination"], "conflict_budget_per_solve": res["conflict_budget_per_solve"],
                     "vars": res["vars"], "clauses": res["clauses"],
                     "fiber": len(scan), "sat_solutions": len(res["solutions"]), "conflicts": res["conflicts"],
                     "match": same})
    return {"instances": instances, "seconds": time.monotonic() - t0,
            "enumeration_matches_scan": agree,
            "mean_conflicts_full_enumeration": float(np.mean([r["conflicts"] for r in rows])),
            "max_conflicts": int(max(r["conflicts"] for r in rows)),
            "vars": rows[0]["vars"], "rows": rows}


def save(path, value):
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--widths", default="8,12,16")
    parser.add_argument("--rounds", default="1,2,4,8,16")
    parser.add_argument("--contexts", type=int, default=256)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--burn-in", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=128)
    parser.add_argument("--sat-instances", type=int, default=32)
    parser.add_argument("--half-instances", type=int, default=256)
    parser.add_argument("--skip-sat", action="store_true", help="explicitly declare a non-SAT validation run")
    parser.add_argument("--wall-seconds", type=int, default=1500)
    parser.add_argument("--plan", type=Path, required=True, help="frozen labnotes plan text to copy")
    args = parser.parse_args()
    widths = [int(v) for v in args.widths.split(",")]
    rounds = [int(v) for v in args.rounds.split(",")]
    if (not widths or not rounds or len(set(widths)) != len(widths) or len(set(rounds)) != len(rounds)
            or any(w not in (8, 12, 16) for w in widths) or any(R < 1 for R in rounds)
            or min(args.contexts, args.samples, args.horizon, args.sat_instances, args.half_instances, args.wall_seconds) < 1
            or args.burn_in < 0 or not args.plan.is_file()):
        parser.error("invalid sizes, duplicate conditions, unsupported width, or missing plan")
    if not args.skip_sat and not HAVE_SAT:
        parser.error("SAT required: install python-sat or explicitly use --skip-sat")
    def timeout(*_):
        raise TimeoutError("declared execution wall cap exceeded")
    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.wall_seconds)
    args.out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    here = Path(__file__).resolve()
    (args.out / "source").mkdir()
    shutil.copyfile(here, args.out / "source" / here.name)
    shutil.copyfile(here.with_name("audit_wide_trajectory.py"), args.out / "source" / "audit_wide_trajectory.py")
    shutil.copyfile(args.plan, args.out / "plan.md")
    save(args.out / "config.json", {
        "schema_version": SCHEMA_VERSION, "sat_required": not args.skip_sat,
        "wall_seconds": args.wall_seconds, "half_instances": args.half_instances,
        "widths": widths, "rounds": rounds, "contexts": args.contexts, "samples": args.samples,
        "burn_in": args.burn_in, "horizon": args.horizon, "sat_instances": args.sat_instances,
        "seed": SEED, "have_sat": HAVE_SAT, "python": platform.python_version(), "numpy": np.__version__,
        "machine": platform.platform(), "node": platform.node(),
        "python_sat": version("python-sat") if HAVE_SAT else None,
    })
    results = []
    for w in widths:
        p = params(w)
        for R in rounds:
            t0 = time.monotonic()
            C = constants(R, p["h"])
            rng = np.random.default_rng(SEED + 1000 * w + R)
            res = {"w": w, "R": R, "N": WORDS * w, "constants": C, "params": p}
            res["fibers"] = fiber_statistics(R, p, C, args.contexts, rng)
            res["ledger"] = ledger(R, p)
            res["half_enumeration"] = half_enumeration_attack(R, p, C, args.half_instances, rng)
            if w <= 16:
                res["panels"] = state_panels(R, p, C, args.samples, args.burn_in, args.horizon, rng)
            if not args.skip_sat:
                res["sat"] = sat_series(R, p, C, args.sat_instances, rng)
            res["seconds"] = time.monotonic() - t0
            results.append(res)
            print(json.dumps({"w": w, "R": R, "seconds": round(res["seconds"], 2),
                              "mean_witness_bits": round(res["fibers"]["mean_witness_bits"], 3),
                              "half_enum": res["half_enumeration"]["preimage_found_fraction"],
                              "sat_conflicts": res.get("sat", {}).get("mean_conflicts_full_enumeration")}), flush=True)
            save(args.out / "results.json", results)
    save(args.out / "receipt.json", {"completed": True, "schema_version": SCHEMA_VERSION,
                                    "conditions": len(results), "seconds": time.monotonic() - started})
    files = [q for q in args.out.rglob("*") if q.is_file() and not q.name.startswith("._")]
    save(args.out / "sha256.json", {str(q.relative_to(args.out)): hashlib.sha256(q.read_bytes()).hexdigest() for q in files})
    signal.alarm(0)
    print(json.dumps({"output": str(args.out.resolve()), "seconds": time.monotonic() - started, "completed": True}))


if __name__ == "__main__":
    main()
