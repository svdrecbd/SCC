"""Balanced permission designs and least-squares recovery of class probabilities."""
import numpy as np


def quadratic_character(value, prime):
    value %= prime
    if value == 0:
        return 0
    residue = pow(int(value), (prime-1)//2, prime)
    return 1 if residue == 1 else -1


def hadamard_matrix(order):
    if order in (2, 4, 8):
        matrix = np.ones((1, 1), dtype=np.int64)
        while len(matrix) < order:
            matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    elif order in (12, 20):
        prime = order-1
        character = np.array([[quadratic_character(left-right, prime)
                               for right in range(prime)] for left in range(prime)], dtype=np.int64)
        matrix = np.ones((order, order), dtype=np.int64)
        matrix[1:, 0] = -1
        matrix[1:, 1:] = character+np.eye(prime, dtype=np.int64)
    elif order == 100:
        elements = [(left, right) for left in range(7) for right in range(7)]
        conference = np.ones((50, 50), dtype=np.int64)
        conference[0, 0] = 0
        for row, (left_a, left_b) in enumerate(elements):
            for column, (right_a, right_b) in enumerate(elements):
                delta_a, delta_b = (left_a-right_a)%7, (left_b-right_b)%7
                norm = (delta_a*delta_a-3*delta_b*delta_b)%7
                conference[row+1, column+1] = quadratic_character(norm, 7)
        assert np.array_equal(conference @ conference.T, 49*np.eye(50, dtype=np.int64))
        identity = np.eye(50, dtype=np.int64)
        matrix = np.block([[conference+identity, conference-identity],
                           [conference-identity, -conference-identity]])
    else:
        raise ValueError("unsupported construction order")
    matrix = matrix * matrix[:1, :]
    matrix = matrix * matrix[:, :1]
    assert np.array_equal(matrix @ matrix.T, order*np.eye(order, dtype=np.int64))
    return matrix


def permission_design(order):
    rows = hadamard_matrix(order)[1:]
    return (np.concatenate([rows, -rows], axis=0)+1)//2


def project_probability_simplex(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("finite matrix of class scores required")
    ordered = np.sort(values, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1)-1
    indices = np.arange(1, values.shape[1]+1)
    positive = ordered-cumulative/indices > 0
    support = positive.sum(axis=1)-1
    threshold = cumulative[np.arange(len(values)), support]/(support+1)
    return np.maximum(values-threshold[:, None], 0)


def recover_probabilities(reports, masks):
    reports = np.asarray(reports, dtype=np.float64)
    masks = np.asarray(masks, dtype=np.float64)
    count = masks.shape[1]
    if reports.ndim != 2 or reports.shape[1] != len(masks):
        raise ValueError("incompatible report dimensions")
    if not np.isfinite(reports).all() or np.any((reports < 0) | (reports > 1)):
        raise ValueError("reports must lie between zero and one")
    # The caller supplies the validated balanced design; this is not a decoder
    # for arbitrary sets of masks.
    centered = masks-.5
    coefficient = count/(4*(count-1))
    raw = np.full((len(reports), count), 1/count)+(reports-.5) @ centered/(len(masks)*coefficient)
    return project_probability_simplex(raw), raw
