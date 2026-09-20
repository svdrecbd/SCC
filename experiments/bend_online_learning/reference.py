"""Closed-form states, exact masses and independent binomial learning curves."""
from fractions import Fraction
from math import comb, log2

COUNT_MODES = {'count_repair', 'complemented_count', 'complemented_count_repaired', 'repair_after_four'}


def expected(mode, t, history, row_id):
    active, h = t, history
    if mode in ('reset_after_four', 'repair_after_four') and t >= 4:
        active, h = t - 4, history >> 4
    if mode == 'repair_after_four' and t < 4:
        active, h = 0, 0
    if mode == 'input_erased_boundary':
        h = 0
    ones = h.bit_count()
    if mode in COUNT_MODES:
        delta = 2 * ones - active
        state = [16 - delta if mode.startswith('complemented') else 16 + delta]
        native = int(state[0] > 16)
        pred = int(delta > 0) if mode == 'complemented_count_repaired' else native
        calls = 0
    else:
        state = [1, 1] if mode == 'neutralized_update' else [3 ** (active - ones), 3 ** ones]
        native = pred = int(state[1] > state[0])
        calls = 0 if mode == 'neutralized_update' else 2 * t
    return dict(id=row_id, mode=mode, t=t, history=history, state=state,
                native_pred=native, pred=pred, permission=1 if mode == 'live_allow' else 1 - pred,
                likelihood_calls=calls)


def information(masses, denominator):
    return sum(m / denominator * log2(2 * m / sum(pair))
               for pair in masses.values() for m in pair if m)


def fraction(x):
    x = Fraction(x)
    return [x.numerator, x.denominator]


def summarize(rows, mode, t):
    denominator = 2 * 4 ** t
    state_mass, raw_mass = {}, {}
    native = predicted = unsafe = 0
    for row in rows:
        history = row['history']
        ones = history.bit_count()
        masses = (3 ** (t - ones), 3 ** ones)
        state_mass.setdefault(tuple(row['state']), [0, 0])
        raw = 0 if mode == 'input_erased_boundary' else history
        raw_mass.setdefault(raw, [0, 0])
        for theta, mass in enumerate(masses):
            state_mass[tuple(row['state'])][theta] += mass
            raw_mass[raw][theta] += mass
            native += mass * (row['native_pred'] == theta)
            predicted += mass * (row['pred'] == theta)
        unsafe += masses[1] * row['permission']
    assert sum(sum(m) for m in state_mass.values()) == denominator
    accuracy = Fraction(predicted, denominator)
    return dict(mode=mode, t=t, state_target_masses=[dict(state=list(k), masses=v)
                for k, v in sorted(state_mass.items())], denominator=denominator,
                native_accuracy=fraction(Fraction(native, denominator)),
                repaired_accuracy=fraction(accuracy),
                best_state_accuracy=fraction(Fraction(sum(max(p) for p in state_mass.values()), denominator)),
                next_observation_accuracy=fraction(Fraction(1, 4) + accuracy / 2),
                unsafe_permission=fraction(Fraction(unsafe, 4 ** t)),
                state_information_bits=information(state_mass, denominator),
                accessible_transcript_information_bits=information(raw_mass, denominator))


def binomial_accuracy(t):
    return fraction(Fraction(sum(comb(t, k) * max(3 ** k, 3 ** (t - k))
                                 for k in range(t + 1)), 2 * 4 ** t))


def binomial_information(t):
    # Group only by sufficient number of ones; separate from per-history rollout.
    masses = {k: [comb(t, k) * 3 ** (t - k), comb(t, k) * 3 ** k] for k in range(t + 1)}
    return information(masses, 2 * 4 ** t)
