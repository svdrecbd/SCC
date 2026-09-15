"""LN-123: untrained live-operator candidate with an explicit reader-splice attack.

Every numerical coefficient is in the mutable bank. The fixed interpreter has
no task algorithm or permission oracle. This does not make its read call sites
inseparable: preserving a transition and editing a later read is still allowed.
"""

import math

import torch


WIDTH = 8
TOKENS = 8
ROLES = 2
WORK_ROWS = 8


def initial_bank(seed, dtype=torch.float64):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(4 * WIDTH + TOKENS + ROLES + WORK_ROWS, WIDTH,
                       generator=generator, dtype=torch.float64).to(dtype) * .2


def operators(bank):
    """Views of the current bank, never a persistent parameter cache."""
    d = bank.shape[1]
    return bank[:4*d].reshape(4, d, d).unbind(0)


def context(bank, token, role):
    base = 4 * bank.shape[1]
    return bank[base + token] + bank[base + TOKENS + role]


def transition(bank, token, role):
    q, k, v, w = operators(bank)
    c = context(bank, token, role)
    scores = ((bank + c) @ q) @ (bank @ k).T / math.sqrt(bank.shape[1])
    return (bank @ w + scores.softmax(-1) @ (bank @ v) + c).tanh()


def read(bank, token, role):
    q, k, v, w = operators(bank)
    scores = (context(bank, token, role) @ q) @ (bank @ k).T / math.sqrt(bank.shape[1])
    return (scores.softmax(-1) @ (bank @ v) @ w)[:3]


def step(bank, token, role, mode="intact"):
    """Pure step; only the returned bank persists in the inference runtime.

    Frozen controls preserve the current code/bank, not an initial clean copy.
    The attack changes only selected reader metadata, after the real transition.
    """
    if mode not in {"intact", "reader_splice", "frozen_code", "frozen_bank"}:
        raise ValueError(f"Unknown mode: {mode}")
    if not 0 <= token < TOKENS or not 0 <= role < ROLES:
        raise ValueError("Token or role out of range")
    if mode == "frozen_bank":
        updated = bank
    else:
        updated = transition(bank, token, role)
        if mode == "frozen_code":
            end = 4 * bank.shape[1]
            updated = torch.cat((bank[:end], updated[end:]), dim=0)
    reader_role = 0 if mode == "reader_splice" and role == 1 and token < 4 else role
    return updated, read(updated, token, reader_role)
