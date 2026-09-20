"""Public-input binary linear repair. No reference inverse enters recover()."""


def apply(rows, value, meter):
    meter['row_parities'] += len(rows)
    return sum(((row & value).bit_count() & 1) << i for i, row in enumerate(rows))


def valid(rows, rhs, candidate, meter):
    if type(candidate) is not int or not 0 <= candidate < 1 << len(rows):
        return False
    meter['candidate_checks'] += 1
    return apply(rows, candidate, meter) == rhs


def recover(rows, rhs, oracle, masks, meter, radius_one=False):
    """Return verified full solution, or None; caller randomizes fallback bit.

    radius_one enumerates a complete Hamming-one ball about each valid word.
    This is a public algorithm, not an assumed immutable trusted component.
    """
    for mask in masks:
        shifted = rhs ^ apply(rows, mask, meter)
        meter['oracle_calls'] += 1
        candidate = oracle(shifted)
        candidates = [candidate]
        if radius_one and type(candidate) is int and 0 <= candidate < 1 << len(rows):
            candidates += [candidate ^ (1 << i) for i in range(len(rows))]
        for proposed in candidates:
            if valid(rows, shifted, proposed, meter):
                return proposed ^ mask
    return None


def solve(rows, rhs, meter):
    """Direct Gaussian elimination, recomputed per RHS; packed-row meter."""
    n = len(rows)
    work = [row | (((rhs >> i) & 1) << n) for i, row in enumerate(rows)]
    for col in range(n):
        pivot = None
        for i in range(col, n):
            meter['pivot_tests'] += 1
            if (work[i] >> col) & 1:
                pivot = i
                break
        if pivot is None:
            raise ValueError('singular matrix')
        work[col], work[pivot] = work[pivot], work[col]
        for i in range(n):
            meter['elimination_tests'] += 1
            if i != col and ((work[i] >> col) & 1):
                work[i] ^= work[col]
                meter['row_xors'] += 1
    return sum(((work[i] >> n) & 1) << i for i in range(n))


def new_meter():
    return dict(row_parities=0, candidate_checks=0, oracle_calls=0,
                pivot_tests=0, elimination_tests=0, row_xors=0)
