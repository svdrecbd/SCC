"""Exact conditional sampling from a noisy affine polynomial code."""
import numpy as np


def interpolation_matrix(field, points, anchors):
    points = np.asarray(points, dtype=np.int64)
    anchors = np.asarray(anchors, dtype=np.int64)
    if len(set(anchors.tolist())) != len(anchors) or not len(anchors):
        raise ValueError('Interpolation anchors must be nonempty and distinct.')
    matrix = np.ones((len(points), len(anchors)), dtype=np.int64)
    for column, anchor in enumerate(anchors):
        denominator = 1
        for other in anchors:
            if other != anchor:
                matrix[:, column] = field.multiply(matrix[:, column], points ^ other)
                denominator = field.multiply(denominator, anchor ^ other)
        matrix[:, column] = field.multiply(matrix[:, column], field.inverse(denominator))
    return matrix


def draw_noise(generator, order, count, correlation):
    values = generator.integers(order, size=count, dtype=np.int64)
    values[generator.random(count) < correlation] = 0
    return values


class ConditionalCodeSampler:
    def __init__(self, field, coordinate_count, dimension, syndrome, observed):
        if not 1 <= dimension < coordinate_count <= field.order:
            raise ValueError('Require 1 <= dimension < coordinate count <= field order.')
        self.field = field
        self.coordinate_count = coordinate_count
        self.dimension = dimension
        self.observed = np.asarray(observed, dtype=np.int64)
        if (len(self.observed) > dimension
                or len(set(self.observed.tolist())) != len(self.observed)
                or np.any(self.observed < 0) or np.any(self.observed >= coordinate_count)):
            raise ValueError('History must contain at most dimension distinct valid positions.')
        syndrome = np.asarray(syndrome, dtype=np.int64)
        if syndrome.shape != (coordinate_count-dimension,) or np.any(syndrome < 0) or np.any(syndrome >= field.order):
            raise ValueError('Invalid syndrome.')
        self.offset = np.concatenate((np.zeros(dimension, dtype=np.int64), syndrome))
        observed_set = set(self.observed.tolist())
        self.unobserved = np.array([position for position in range(coordinate_count)
                                    if position not in observed_set], dtype=np.int64)
        self.anchors = np.concatenate((self.observed, self.unobserved[:dimension-len(self.observed)]))
        self.matrix = interpolation_matrix(field, np.arange(coordinate_count), self.anchors)

    def construct(self, values, observed_noise, free_values, remaining_noise):
        values, observed_noise, free_values, remaining_noise = [np.asarray(item, dtype=np.int64)
            for item in (values, observed_noise, free_values, remaining_noise)]
        expected = (len(self.observed), len(self.observed), self.dimension-len(self.observed), len(self.unobserved))
        for item, count in zip((values, observed_noise, free_values, remaining_noise), expected):
            if item.shape != (count,) or np.any(item < 0) or np.any(item >= self.field.order):
                raise ValueError('Invalid sampling input.')
        anchor_values = np.concatenate((values ^ observed_noise ^ self.offset[self.observed], free_values))
        source = self.field.linear_image(self.matrix, anchor_values) ^ self.offset
        output = source.copy()
        output[self.observed] = values
        output[self.unobserved] ^= remaining_noise
        return source, output

    def sample(self, values, correlation, generator):
        if not 0 <= correlation <= 1:
            raise ValueError('Correlation must lie in [0,1].')
        return self.construct(values,
            draw_noise(generator, self.field.order, len(self.observed), correlation),
            generator.integers(self.field.order, size=self.dimension-len(self.observed)),
            draw_noise(generator, self.field.order, len(self.unobserved), correlation))
