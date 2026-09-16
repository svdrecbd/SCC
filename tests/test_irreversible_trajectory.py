"""Implementation checks for the LN-141 irreversible-trajectory screen (not SCC evidence)."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.screen_irreversible_trajectory import (  # noqa: E402
    ALPHABET, WORDS, dag, make_tables, pebble_min_cells, side_information_curve,
    step_table, NSTATES,
)


def scalar_step(tab, state, rounds):
    words = [(state >> (4 * i)) & 15 for i in range(WORDS)]
    for _ in range(rounds):
        for i in range(WORDS):
            words[i] = tab[words[(i + 1) % WORDS]][words[i]]
    return sum(w << (4 * i) for i, w in enumerate(words))


def test_tables_shapes_and_nearperm_has_exactly_one_collision():
    for variant in ("function", "nearperm"):
        t = make_tables(variant, 1)
        assert len(t) == 2 and len(t[0]) == 4 and len(t[0][0]) == 16 and len(t[0][0][0]) == 16
    t = make_tables("nearperm", 1)
    for r in range(2):
        for x in range(4):
            for b in range(16):
                assert len(set(t[r][x][b])) == 15


def test_vectorised_step_matches_scalar_reference():
    t = make_tables("function", 3)
    rng = np.random.default_rng(0)
    for rounds in (1, 2, 3):
        F = step_table(t, 1, 2, rounds)
        for s in rng.integers(0, NSTATES, size=64):
            assert int(F[s]) == scalar_step(t[1][2], int(s), rounds)


def test_dag_one_round_ring_reads_updated_neighbour_only_at_last_word():
    preds, finals = dag(1, (0,))
    assert preds[4] == (0, 1) and preds[5] == (1, 2) and preds[6] == (2, 3) and preds[7] == (3, 4)
    assert finals[0] == [4, 5, 6, 7]


def test_pebbling_minima_for_one_round():
    p0, f0 = dag(1, (0,))
    assert pebble_min_cells(p0, f0[0], None, capacity=4)
    p1, f1 = dag(1, (1,))
    assert not pebble_min_cells(p1, list(range(WORDS)), f1[1][0], capacity=4)
    assert pebble_min_cells(p1, list(range(WORDS)), f1[1][0], capacity=5)
    pj, fj = dag(1, (1, 0))
    assert not pebble_min_cells(pj, fj[0], fj[1][0], capacity=4)
    assert pebble_min_cells(pj, fj[0], fj[1][0], capacity=5)


def test_side_information_curve_exact_on_toy():
    # two source fibers; targets {0,1,2} in fiber A with weights .2,.2,.1; {5} in fiber B .5
    src = np.array([7, 7, 7, 9] + [0] * (NSTATES - 4))
    tgt = np.array([0, 1, 2, 5] + [0] * (NSTATES - 4))
    w = np.array([0.2, 0.2, 0.1, 0.5] + [0.0] * (NSTATES - 4))
    out = side_information_curve(src, tgt, w)
    assert abs(out["success_by_side_bits"][0] - 0.7) < 1e-12   # best single guess per fiber
    assert abs(out["success_by_side_bits"][1] - 0.9) < 1e-12   # two guesses per fiber
    assert abs(out["success_by_side_bits"][2] - 1.0) < 1e-12
    assert out["max_fiber_spread"] == 3 and out["bits_for_exact"] == 2
