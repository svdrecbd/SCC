"""Finite-price risk reader with a fixed public reference probability."""

import math
import random


def recover_shrunk_risk(policy, baseline, mixing_weight, calls=1,
                       price_levels=65536, seed=None, complement=False):
    """Return a convex mixture of the public reference and action frequency.

    Use fresh randomness unless explicitly reproducing a development check.
    Policy chooses 1 for the known-price service. Multiple calls require an
    independently repeatable endpoint supplied by the caller. The baseline must
    correspond to the reference policy on this same finite price grid.
    """
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 1:
        raise ValueError("calls must be a positive integer")
    if (isinstance(price_levels, bool) or not isinstance(price_levels, int)
            or not 2 <= price_levels <= 2**52 or price_levels & (price_levels-1)):
        raise ValueError("price levels must be a power of two between 2 and 2**52")
    if not math.isfinite(baseline) or not 0 <= baseline <= 1:
        raise ValueError("baseline must be a finite probability")
    if baseline * price_levels != int(baseline * price_levels):
        raise ValueError("baseline must lie on the declared probability grid")
    if not math.isfinite(mixing_weight) or not 0 <= mixing_weight <= 1:
        raise ValueError("mixing weight must be in [0, 1]")
    generator = random.SystemRandom() if seed is None else random.Random(seed)
    selected = 0
    for _ in range(calls):
        price = (generator.randrange(price_levels)+.5)/price_levels
        action = policy(price)
        if action not in (0, 1):
            raise ValueError("policy must return a binary action")
        selected += action
    frequency = selected / calls
    if complement:
        frequency = 1-frequency
    return (1-mixing_weight)*baseline + mixing_weight*frequency
