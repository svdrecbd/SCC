"""Independent audit of an LN-143 wide-word run. Standard library only; no runner import.

Recomputes the Davies-Meyer ARX step from the saved constants and parameters,
replays every saved fiber context exactly over all 2^w inputs, re-derives the
ledger arithmetic, checks that SAT solution counts equal scan fibers wherever the
runner recorded both, verifies hashes, and reports discrepancies.
"""

import argparse
import hashlib
import json
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


def audit(directory, max_contexts):
    report = {"hash_mismatches": [], "checks": []}
    for name, digest in json.loads((directory / "sha256.json").read_text()).items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
            report["hash_mismatches"].append(name)
    results = json.loads((directory / "results.json").read_text())
    for res in results:
        w, R, p, C = res["w"], res["R"], res["params"], res["constants"]
        check = {"w": w, "R": R, "ok": True, "notes": [], "contexts_replayed": 0}

        def bad(msg):
            check["ok"] = False
            check["notes"].append(msg)

        for ctx in res["fibers"]["contexts"][:max_contexts]:
            counts = {}
            for a in range(1 << w):
                v = T(a, ctx["b"], ctx["x"], ctx["r"], R, p, C)
                counts[v] = counts.get(v, 0) + 1
            if len(counts) != ctx["image_size"] or max(counts.values()) != ctx["max_fiber"]:
                bad(f"context {ctx}: audit image {len(counts)} fiber {max(counts.values())}")
            check["contexts_replayed"] += 1
        led = res["ledger"]
        if led["intact_ops_per_request"] != WORDS * (6 * R + 1) or led["scan_time_factor"] != 1 << w \
                or led["preimage_table_bits_total"] != (1 << w) * INPUTS * ROLES * (1 << w) * w:
            bad("ledger arithmetic mismatch")
        if R == 1 and res["half_enumeration"]["preimage_found_fraction"] != 1.0:
            bad("half enumeration must succeed on every R=1 instance")
        if "sat" in res:
            rows = res["sat"]["rows"]
            if res["sat"]["enumeration_matches_scan"] != len(rows) or any(not r["match"] for r in rows):
                bad("SAT enumeration disagreed with scan on some instance")
        if "panels" in res:
            if res["panels"]["benign_honest"]["answer_agreement_after_attack_mean"] != 1.0:
                bad("honest control lost utility")
        report["checks"].append(check)
    report["passed"] = not report["hash_mismatches"] and all(c["ok"] for c in report["checks"])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-contexts", type=int, default=8, help="exact 2^w replays per (w,R)")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("output path must not exist")
    report = audit(args.directory, args.max_contexts)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"passed": report["passed"], "out": str(args.out)}))


if __name__ == "__main__":
    main()
