"""Recover an event probability from independently repeatable priced choices."""

import random


def recover_probability(policy, calls, seed, complement=False):
    """Policy returns 1 for known-price service, 0 for uncertain unit loss.

    Each call must begin from the declared endpoint state. This function cannot
    enforce reset semantics or independence of the policy's own random state.
    No outcome, reference probability, model parameter or likelihood is read.
    """
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 1:
        raise ValueError("calls must be a positive integer")
    generator = random.Random(seed)
    selected = 0
    for _ in range(calls):
        action = policy(generator.random())
        if action not in (0, 1):
            raise ValueError("policy must return a binary action")
        selected += action
    estimate = selected / calls
    return 1 - estimate if complement else estimate


def variance_allowance(calls):
    if isinstance(calls, bool) or not isinstance(calls, int) or calls < 1:
        raise ValueError("calls must be a positive integer")
    return 1 / (4 * calls)
