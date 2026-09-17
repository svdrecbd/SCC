"""Independent audit of an LN-143 wide-word run. Standard library only; no runner import.

Requires the version-2 evidence contract and complete configured inventory.
Recomputes fibers, SAT solution sets and trajectory panel scores from saved inputs.
An optional context limit is reported as partial fiber coverage; SAT and trajectory
replay remain complete. Legacy runs fail closed without modifying their evidence.
"""

import argparse
import hashlib
import json
import math
import signal
from pathlib import Path

WORDS, INPUTS, ROLES = 4, 4, 2


def rotl(v, s, h, mask):
    s %= h
    return ((v << s) | (v >> (h - s))) & mask


def rotr(v, s, h, mask):
    return rotl(v, h - (s % h), h, mask)


def keys(b, x, r, R, p, C):
    h, mask = p["h"], p["mask"]
    bl, bh = b & mask, (b >> h) & mask
    return [(rotl(bl, j + 1, h, mask) ^ rotl(bh, 2 * j + 1, h, mask) ^ (((x << 1) | r) * 0x1D) ^ C[j]) & mask
            for j in range(R)]


def T(a, b, x, r, R, p, C):
    h, mask = p["h"], p["mask"]
    L, Rh = a & mask, (a >> h) & mask
    for k in keys(b, x, r, R, p, C):
        L = ((rotr(L, p["alpha"], h, mask) + Rh) & mask) ^ k
        Rh = rotl(Rh, p["beta"], h, mask) ^ L
    return ((Rh << h) | L) ^ a


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same(actual, expected, label):
    if isinstance(expected, dict):
        require(isinstance(actual, dict), label)
        for key, value in expected.items():
            require(key in actual, f"{label}: missing {key}")
            same(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), label)
        for i, (a, b) in enumerate(zip(actual, expected)):
            same(a, b, f"{label}[{i}]")
    elif isinstance(expected, float):
        require(isinstance(actual, (int, float)) and math.isfinite(actual)
                and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), label)
    else:
        require(actual == expected, label)


def step(B, x, r, R, p, C):
    B = list(B)
    for i in range(4):
        B[i] = T(B[i], B[(i+1) % 4], x, r, R, p, C)
    return B


ARMS = ("one_owner_step_then_honest", "constant_owner", "selective_owner_on_S",
        "benign_honest", "forward_only_owner_then_honest")


def replay_panels(res, config):
    panels = res["panels"]
    require(set(panels) == set(ARMS) | {"chance", "evidence"}, "panel inventory")
    w, R, p, C = res["w"], res["R"], res["params"], res["constants"]
    n, burn, horizon = config["samples"], config["burn_in"], config["horizon"]
    ev = panels["evidence"]
    B0, xs, roles = ev["initial_state_words"], ev["task_inputs"], ev["roles"]
    total = burn + horizon + 1
    require(len(B0) == 4 and all(len(b) == n for b in B0), "initial state shape")
    require(all(type(v) is int and 0 <= v < 1 << w for b in B0 for v in b), "initial state range")
    require(len(xs) == len(roles) == n, "panel sample count")
    require(all(len(x) == len(r) == total for x, r in zip(xs, roles)), "panel request count")
    require(all(type(v) is int and 0 <= v < 4 for row in xs for v in row), "task input range")
    require(all(type(v) is int and v in (0, 1) for row in roles for v in row), "role range")
    require(all(r[burn] == 0 for r in roles), "attack must be a caller request")
    same(panels["chance"], 1.0 / (1 << w), "chance")
    for arm in ARMS:
        agree, equal, hit = [0]*(horizon+1), [0]*(horizon+1), 0
        for j in range(n):
            ref = [b[j] for b in B0]; att = list(ref)
            for t, (x, r) in enumerate(zip(xs[j], roles[j])):
                ra = r
                if arm == "constant_owner" or (arm == "one_owner_step_then_honest" and t == burn):
                    ra = 1
                if arm == "selective_owner_on_S" and r == 0 and x in (0, 1):
                    ra = 1
                ref_next, att_next = step(ref,x,r,R,p,C), step(att,x,ra,R,p,C)
                if t == burn:
                    owner = step(ref,x,1,R,p,C)[0]
                    emitted = (T(att[0],att[1],x,1,R,p,C)
                               if arm == "forward_only_owner_then_honest" else att_next[0])
                    hit += emitted == owner
                ref, att = ref_next, att_next
                if t >= burn:
                    agree[t-burn] += ref[0] == att[0]
                    equal[t-burn] += ref == att
        same(panels[arm], {
            "owner_answer_correct_at_attack": hit/n,
            "answer_agreement_after_attack_mean": sum(agree[1:])/(n*horizon),
            "state_equality_at_steps_1_16_64_last": [equal[i]/n for i in (1,min(16,horizon),min(64,horizon),horizon)],
            "answer_agreement_by_step_sparse": [v/n for v in agree[::8]],
        }, arm)
    same(panels["benign_honest"]["answer_agreement_after_attack_mean"],1.0,"honest control")
    same(panels["forward_only_owner_then_honest"], {
        "owner_answer_correct_at_attack":1.0, "answer_agreement_after_attack_mean":1.0,
        "state_equality_at_steps_1_16_64_last":[1.0]*4,
    }, "forward bypass")
    return n * total * len(ARMS)


def check_condition(res, config, max_contexts):
    w, R, p, C = res["w"], res["R"], res["params"], res["constants"]
    h = w//2
    same(p,{"w":w,"h":h,"alpha":h//2+1,"beta":2,"mask":(1<<h)-1,"wmask":(1<<w)-1},"params")
    require(res["N"] == 4*w and len(C) == R and all(type(c) is int and 0 <= c < 1<<h for c in C), "machine shape")
    contexts = res["fibers"]["contexts"]
    require(len(contexts) == config["contexts"], "fiber context count")
    limit = len(contexts) if max_contexts is None else min(max_contexts, len(contexts))
    for ctx in contexts:
        require(type(ctx["b"]) is int and 0 <= ctx["b"] < 1<<w
                and ctx["x"] in range(4) and ctx["r"] in (0,1), "fiber context range")
    for ctx in contexts[:limit]:
        counts = {}
        for a in range(1<<w):
            v = T(a,ctx["b"],ctx["x"],ctx["r"],R,p,C)
            counts[v] = counts.get(v,0)+1
        same(ctx,{"image_size":len(counts),"max_fiber":max(counts.values()),
                  "mean_witness_bits":sum(v*math.log2(v) for v in counts.values())/(1<<w),
                  "worst_witness_bits":math.ceil(math.log2(max(counts.values())))},"fiber replay")
    same(res["fibers"],{
        "image_fraction_mean":sum(c["image_size"] for c in contexts)/len(contexts)/(1<<w),
        "max_fiber_over_contexts":max(c["max_fiber"] for c in contexts),
        "mean_witness_bits":sum(c["mean_witness_bits"] for c in contexts)/len(contexts),
        "worst_witness_bits":max(c["worst_witness_bits"] for c in contexts)},"fiber aggregate")
    same(res["ledger"],{
        "intact_T_calls_per_request":4,"forward_only_T_calls_per_attacked_request":5,
        "forward_only_T_call_factor":1.25,"ring_passes_per_request":1,"cipher_rounds_per_T":R,
        "scan_forward_evaluations_per_word_update":1<<w,"scan_T_calls_per_request":4*(1<<w),
        "scan_time_factor":1<<w,"half_enumeration_forward_evaluations":1<<h,
        "preimage_table_bits_total":8*w*(1<<(2*w)),"round_constant_bits":R*h,
        "live_state_bits":4*w,"full_copy_scratch_bits":4*w,"save_restore_extra_word_bits":w,
        "forward_only_extra_persistent_bits":0,"forward_only_extra_workspace_beyond_common_T_bits":0,
        "workspace_schedule":{"cipher_halves_bits":w,"materialized_round_keys_bits":R*h},
        "peak_machine_bits":None,"primitive_instruction_budget":None},"ledger")
    half = res["half_enumeration"]
    require(half["instances"] == config["half_instances"] and half["candidates_per_instance"] == 1<<h,
            "half-enumeration shape")
    require(0 <= half["preimage_found_fraction"] <= 1, "half-enumeration fraction")
    require(len(half["rows"]) == config["half_instances"], "half-enumeration row count")
    hits = 0
    for row in half["rows"]:
        require(all(type(row[k]) is int and 0 <= row[k] < 1<<w for k in ("a","b","v"))
                and row["x"] in range(4) and row["r"] in (0,1), "half-enumeration context range")
        require(T(row["a"],row["b"],row["x"],row["r"],R,p,C) == row["v"], "half target")
        vL, vR = row["v"] & ((1<<h)-1), row["v"] >> h
        found = any(T((high<<h) | (vR ^ vL ^ rotl(high,p["beta"],h,(1<<h)-1) ^ high),
                      row["b"],row["x"],row["r"],R,p,C) == row["v"] for high in range(1<<h))
        require(row["found"] is found, "half-enumeration result mismatch")
        hits += found
    same(half["preimage_found_fraction"],hits/len(half["rows"]),"half-enumeration aggregate")
    if R == 1:
        same(half["preimage_found_fraction"],1.0,"R=1 positive control")
    sat_replayed = 0
    if config["sat_required"]:
        sat = res["sat"]; rows = sat["rows"]
        require(sat["instances"] == len(rows) == config["sat_instances"], "SAT inventory")
        for row in rows:
            require(row["termination"] == "exhausted", "SAT enumeration incomplete")
            require(type(row["b"]) is int and 0 <= row["b"] < 1<<w and 0 <= row["v"] < 1<<w
                    and row["x"] in range(4) and row["r"] in (0,1), "SAT context range")
            sols = row["solutions"]
            require(all(type(a) is int and 0 <= a < 1<<w for a in sols) and len(sols) == len(set(sols)), "invalid SAT solutions")
            expected = [a for a in range(1<<w) if T(a,row["b"],row["x"],row["r"],R,p,C) == row["v"]]
            require(sorted(sols) == expected and len(expected) > 0, "SAT solution set disagrees with independent scan")
            same(row,{"fiber":len(expected),"sat_solutions":len(expected),"match":True},"SAT row")
            require(row["vars"] == sat["vars"] and row["conflicts"] >= 0 and row["clauses"] > 0
                    and row["conflict_budget_per_solve"] > 0,"SAT telemetry shape")
            sat_replayed += 1
        same(sat,{"enumeration_matches_scan":len(rows),
                  "mean_conflicts_full_enumeration":sum(r["conflicts"] for r in rows)/len(rows),
                  "max_conflicts":max(r["conflicts"] for r in rows)},"SAT aggregate")
    else:
        require("sat" not in res, "unexpected SAT evidence in non-SAT run")
    transitions = replay_panels(res,config)
    return {"w":w,"R":R,"ok":True,"contexts_replayed":limit,"contexts_available":len(contexts),
            "sat_instances_replayed":sat_replayed,"panel_transitions_replayed":transitions}


def audit(directory, max_contexts=None):
    directory = Path(directory)
    report = {"passed":False,"hash_mismatches":[],"errors":[],"checks":[],
              "scope":"independent fiber/SAT-solution/half-enumeration/panel replay; solver telemetry is checked for consistency, not independently rerun"}
    try:
        require(max_contexts is None or type(max_contexts) is int and max_contexts > 0, "context limit must be positive")
        manifest = json.loads((directory/"sha256.json").read_text())
        require(isinstance(manifest,dict) and manifest, "empty manifest")
        required = {"config.json","receipt.json","results.json","plan.md",
                    "source/run_wide_trajectory.py","source/audit_wide_trajectory.py"}
        require(required <= set(manifest), "manifest missing required evidence/source")
        for name,digest in manifest.items():
            path = directory/name
            require(not Path(name).is_absolute() and path.resolve().is_relative_to(directory.resolve()), "unsafe manifest path")
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                report["hash_mismatches"].append(name)
        require(not report["hash_mismatches"], "manifest hash verification failed")
        config = json.loads((directory/"config.json").read_text())
        receipt = json.loads((directory/"receipt.json").read_text())
        results = json.loads((directory/"results.json").read_text())
        require(config["schema_version"] == receipt["schema_version"] == 2, "unsupported evidence version; legacy evidence is incomplete")
        require(receipt["completed"] is True, "missing completion")
        require(type(config["sat_required"]) is bool and (not config["sat_required"] or config["have_sat"] is True), "SAT contract")
        widths, rounds = config["widths"],config["rounds"]
        require(widths and rounds and len(widths) == len(set(widths)) and len(rounds) == len(set(rounds)), "empty or duplicate configured inventory")
        require(all(type(w) is int and w in (8,12,16) for w in widths)
                and all(type(r) is int and r > 0 for r in rounds), "invalid condition sizes")
        require(all(type(config[k]) is int and config[k] > 0 for k in ("contexts","samples","horizon","sat_instances","half_instances","wall_seconds"))
                and type(config["burn_in"]) is int and config["burn_in"] >= 0, "invalid sample configuration")
        expected = {(w,r) for w in widths for r in rounds}
        require(isinstance(results,list) and len(results) == receipt["conditions"] == len(expected), "incomplete result inventory")
        require({(r["w"],r["R"]) for r in results} == expected, "missing, duplicate or unexpected condition")
        require(isinstance(receipt["seconds"],(int,float)) and 0 <= receipt["seconds"] <= config["wall_seconds"], "wall contract")
        report["sat_required"] = config["sat_required"]
        for res in results:
            report["checks"].append(check_condition(res,config,max_contexts))
        report["fiber_coverage"] = "full" if all(c["contexts_replayed"] == c["contexts_available"] for c in report["checks"]) else "partial"
        report["passed"] = True
    except (OSError, ValueError, KeyError, TypeError, IndexError, OverflowError) as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--max-contexts",type=int,default=None,help="optional partial fiber replay; default all contexts")
    parser.add_argument("--wall-seconds",type=int,default=1500)
    args = parser.parse_args()
    if args.wall_seconds <= 0:
        parser.error("wall cap must be positive")
    def timeout(*_):
        raise TimeoutError("audit wall cap exceeded")
    signal.signal(signal.SIGALRM,timeout)
    signal.alarm(args.wall_seconds)
    report = audit(args.directory,args.max_contexts)
    with args.out.open("x") as f:
        json.dump(report,f,indent=2); f.write("\n")
    signal.alarm(0)
    print(json.dumps({"passed":report["passed"],"out":str(args.out)}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
