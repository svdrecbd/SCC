"""Explicit binary-extension field arithmetic and bounded polynomial recovery."""
import numpy as np


class BinaryExtensionField:
    def __init__(self, degree, modulus):
        self.degree = degree
        self.order = 1 << degree
        self.exponents = np.empty(self.order-1, dtype=np.int64)
        self.logarithms = np.zeros(self.order, dtype=np.int64)
        value = 1
        visited = set()
        for exponent in range(self.order-1):
            if value in visited or not 1 <= value < self.order:
                raise ValueError('The declared polynomial does not give the required primitive cycle.')
            visited.add(value)
            self.exponents[exponent] = value
            self.logarithms[value] = exponent
            value <<= 1
            if value & self.order:
                value ^= modulus
        if value != 1 or visited != set(range(1, self.order)):
            raise ValueError('The nonzero multiplicative cycle is incomplete.')
        self.modulus = modulus

    def multiply(self, left, right):
        left, right = np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)
        result = self.exponents[(self.logarithms[left]+self.logarithms[right]) % (self.order-1)]
        result = np.where((left == 0) | (right == 0), 0, result)
        return int(result) if result.ndim == 0 else result

    def inverse(self, values):
        values = np.asarray(values, dtype=np.int64)
        if np.any(values == 0):
            raise ValueError('Zero has no multiplicative inverse.')
        result = self.exponents[(-self.logarithms[values]) % (self.order-1)]
        return int(result) if result.ndim == 0 else result

    def evaluate(self, coefficients, points):
        points = np.asarray(points, dtype=np.int64)
        result = np.zeros_like(points)
        for coefficient in reversed(coefficients):
            result = self.multiply(result, points) ^ int(coefficient)
        return result

    def interpolation(self, count, anchors):
        if not 1 <= anchors < count <= self.order:
            raise ValueError('Require distinct field positions and a proper anchor subset.')
        points = np.arange(count, dtype=np.int64)
        result = np.ones((count, anchors), dtype=np.int64)
        for column in range(anchors):
            denominator = 1
            for other in range(anchors):
                if other == column:
                    continue
                result[:, column] = self.multiply(result[:, column], points ^ other)
                denominator = self.multiply(denominator, column ^ other)
            result[:, column] = self.multiply(result[:, column], self.inverse(denominator))
        assert np.array_equal(result[:anchors], np.eye(anchors, dtype=np.int64))
        return result

    def linear_image(self, matrix, vector):
        matrix = np.asarray(matrix, dtype=np.int64)
        vector = np.asarray(vector, dtype=np.int64)
        return np.bitwise_xor.reduce(self.multiply(matrix, vector), axis=-1)


def divide_polynomials(field, numerator, denominator):
    remainder = list(map(int, numerator))
    divisor = list(map(int, denominator))
    while len(remainder) > 1 and remainder[-1] == 0:
        remainder.pop()
    while len(divisor) > 1 and divisor[-1] == 0:
        divisor.pop()
    if divisor == [0]:
        raise ValueError('A zero polynomial cannot be a divisor.')
    quotient = [0]*max(1, len(remainder)-len(divisor)+1)
    inverse_lead = field.inverse(divisor[-1])
    while len(remainder) >= len(divisor) and any(remainder):
        offset = len(remainder)-len(divisor)
        coefficient = field.multiply(remainder[-1], inverse_lead)
        quotient[offset] = coefficient
        for index, value in enumerate(divisor):
            remainder[offset+index] ^= field.multiply(coefficient, value)
        while len(remainder) > 1 and remainder[-1] == 0:
            remainder.pop()
    while len(quotient) > 1 and quotient[-1] == 0:
        quotient.pop()
    return quotient, remainder


def recover_polynomial(field, points, observations, dimension, error_limit):
    points = np.asarray(points, dtype=np.int64)
    observations = np.asarray(observations, dtype=np.int64)
    unknowns = dimension+2*error_limit
    if len(points) < unknowns or len(set(points.tolist())) != len(points):
        raise ValueError('Insufficient distinct observations for the declared error bound.')
    powers = np.ones((len(points), dimension+error_limit), dtype=np.int64)
    for index in range(1, powers.shape[1]):
        powers[:, index] = field.multiply(powers[:, index-1], points)
    right = field.multiply(observations, powers[:, error_limit])
    matrix = np.concatenate((powers,
                             field.multiply(observations[:, None], powers[:, :error_limit]),
                             right[:, None]), axis=1)
    pivot_columns = []
    for column in range(unknowns):
        row = len(pivot_columns)
        candidates = np.flatnonzero(matrix[row:, column])
        if not len(candidates):
            continue
        pivot = row+int(candidates[0])
        matrix[[row, pivot]] = matrix[[pivot, row]]
        matrix[row] = field.multiply(matrix[row], field.inverse(matrix[row, column]))
        coefficients = matrix[:, column].copy()
        coefficients[row] = 0
        matrix ^= field.multiply(coefficients[:, None], matrix[row][None, :])
        pivot_columns.append(column)
    for row in range(len(pivot_columns), len(points)):
        if not matrix[row, :unknowns].any() and matrix[row, -1] != 0:
            return None
    solution = np.zeros(unknowns, dtype=np.int64)
    for row, column in enumerate(pivot_columns):
        solution[column] = matrix[row, -1]
    numerator = solution[:dimension+error_limit].tolist()
    denominator = solution[dimension+error_limit:].tolist()+[1]
    quotient, remainder = divide_polynomials(field, numerator, denominator)
    if any(remainder) or len(quotient) > dimension:
        return None
    errors = int(np.count_nonzero(field.evaluate(quotient, points) != observations))
    return quotient if errors <= error_limit else None
