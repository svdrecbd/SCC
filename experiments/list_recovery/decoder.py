"""GL list recovery; decoder never receives a solution, verifier or error rule."""
from fractions import Fraction
import random
import numpy as np


def choose_k(n, epsilon, failure):
    epsilon, failure = Fraction(epsilon), Fraction(failure)
    if not 0 < epsilon <= Fraction(1, 2) or not 0 < failure < 1:
        raise ValueError('invalid advantage/failure target')
    k = 1
    while Fraction(n, 4*(2**k-1)*epsilon**2) > failure:
        k += 1
    return k


def subset_masks(n, k, seed):
    if not 1 <= n <= 64 or not 1 <= k <= 20:
        raise ValueError('implementation supports n<=64 and k<=20')
    rng = random.Random(seed)
    basis = [rng.getrandbits(n) for _ in range(k)]
    masks = np.zeros(2**k, dtype=np.uint64)
    for j, value in enumerate(basis):
        width = 2**j
        masks[width:2*width] = masks[:width] ^ np.uint64(value)
    return basis, masks


def walsh(values):
    values = np.array(values, dtype=np.int64, copy=True)
    width = 1
    while width < len(values):
        view = values.reshape(-1, 2*width)
        left, right = view[:, :width].copy(), view[:, width:].copy()
        view[:, :width] = left + right
        view[:, width:] = left - right
        width *= 2
    return values


def decode(n, oracle, k, seed, trace_path):
    basis, masks = subset_masks(n, k, seed)
    length = len(masks)
    candidates = np.zeros(length, dtype=np.uint64)
    packed = np.zeros((n, (length-1+7)//8), dtype=np.uint8)
    for i in range(n):
        responses = np.asarray(oracle(masks[1:] ^ np.uint64(1 << i)))
        if responses.shape != (length-1,) or not np.all((responses == 0) | (responses == 1)):
            raise ValueError('oracle must return one bit per query')
        packed[i] = np.packbits(responses.astype(np.uint8), bitorder='little')
        votes = np.zeros(length, dtype=np.int64)
        votes[1:] = 1 - 2*responses.astype(np.int64)
        scores = walsh(votes)
        # length-1 is odd, so all integer scores are nonzero.
        assert np.all(scores != 0)
        candidates |= (scores < 0).astype(np.uint64) << np.uint64(i)
    np.savez_compressed(trace_path, responses=packed, candidates=candidates)
    unique = sorted(map(int, np.unique(candidates)))
    return dict(n=n, k=k, decoder_seed=seed, basis=basis, candidates=unique,
                logical_oracle_queries=n*(length-1), oracle_batches=n,
                transform_add_sub=n*length*k, candidate_bit_placements=n*length,
                subset_xors=length-1, transcript_payload_bytes=packed.nbytes,
                candidate_array_bytes=candidates.nbytes, mask_array_bytes=masks.nbytes,
                transform_array_bytes=8*length, raw_candidate_slots=length)


def select(candidates, rows, rhs):
    matches = []
    for candidate in candidates:
        value = sum(((row & candidate).bit_count() & 1) << i for i, row in enumerate(rows))
        if value == rhs:
            matches.append(candidate)
    return dict(matches=matches, candidate_checks=len(candidates),
                verifier_row_parities=len(candidates)*len(rows))
