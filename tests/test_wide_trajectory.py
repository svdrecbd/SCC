"""Implementation checks for the LN-143 wide-word runner (not SCC evidence)."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_wide_trajectory import (  # noqa: E402
    CNF, HAVE_SAT, T, constants, half_enumeration_attack, params, rotl, rotr, round_keys, sat_preimage,
)


def test_rotations_are_inverse_and_masked():
    p = params(16)
    v = np.int64(0b1010_0000_0110_0001 & p["mask"])
    assert rotr(rotl(v, 3, p["h"], p["mask"]), 3, p["h"], p["mask"]) == v
    assert rotl(np.int64(0xFF), 4, 8, 0xFF) == 0xFF


def test_cipher_core_is_a_permutation_so_only_feedforward_loses_information():
    # E_k(a) = T(a) ^ a must be a bijection on w bits for every key context
    w = 8
    p = params(w); C = constants(4, p["h"])
    a = np.arange(1 << w, dtype=np.int64)
    for b, x, r in [(0, 0, 0), (137, 2, 1), (255, 3, 0)]:
        e = T(a, np.int64(b), x, r, 4, p, C) ^ a
        assert len(np.unique(e)) == 1 << w


def test_half_enumeration_recovers_every_preimage_for_one_round():
    p = params(12); C = constants(1, p["h"])
    out = half_enumeration_attack(1, p, C, 64, np.random.default_rng(1))
    assert out["preimage_found_fraction"] == 1.0 and out["candidates_per_instance"] == 1 << 6


def test_round_keys_depend_on_role_and_input():
    p = params(16); C = constants(2, p["h"])
    assert round_keys(5, 1, 0, 2, p, C) != round_keys(5, 1, 1, 2, p, C)
    assert round_keys(5, 1, 0, 2, p, C) != round_keys(5, 2, 0, 2, p, C)


def test_cnf_adder_matches_integer_addition():
    pytest.importorskip("pysat")
    from pysat.solvers import Cadical195
    cnf = CNF()
    A = [cnf.var() for _ in range(4)]; B = [cnf.var() for _ in range(4)]
    S = cnf.add(A, B)
    for a, b in [(3, 5), (15, 1), (9, 9)]:
        units = [(v if (a >> i) & 1 else -v) for i, v in enumerate(A)] + [(v if (b >> i) & 1 else -v) for i, v in enumerate(B)]
        with Cadical195(bootstrap_with=cnf.clauses + [[u] for u in units]) as s:
            assert s.solve()
            m = set(l for l in s.get_model() if l > 0)
            assert sum(1 << i for i, v in enumerate(S) if v in m) == (a + b) & 15


@pytest.mark.skipif(not HAVE_SAT, reason="python-sat not installed")
def test_sat_preimage_enumeration_matches_scan():
    w = 8
    p = params(w); C = constants(2, p["h"])
    a = np.arange(1 << w, dtype=np.int64)
    b, x, r = 77, 1, 1
    v = int(T(np.int64(33), np.int64(b), x, r, 2, p, C))
    scan = sorted(int(s) for s in np.flatnonzero(T(a, np.int64(b), x, r, 2, p, C) == v))
    res = sat_preimage(2, p, C, b, x, r, v, enumerate_all=True)
    assert sorted(res["solutions"]) == scan and 33 in scan
