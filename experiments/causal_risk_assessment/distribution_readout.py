"""Read the released predictor's uniform-interior, half-normal-tail mixture."""
import numpy as np
from scipy.special import ndtr, ndtri


class ConsequenceDistribution:
    def __init__(self, borders, probabilities):
        self.borders = np.asarray(borders, dtype=float)
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.widths = np.diff(self.borders)
        if self.borders.ndim != 1 or len(self.widths) < 2 or np.any(self.widths <= 0):
            raise ValueError('Strictly increasing borders and at least two components required')
        if self.probabilities.shape[-1] != len(self.widths):
            raise ValueError('Probability component count differs from border count')
        if not np.all(np.isfinite(self.probabilities)) or np.any(self.probabilities < 0):
            raise ValueError('Finite nonnegative probabilities required')
        if not np.allclose(self.probabilities.sum(axis=-1), 1, atol=2e-6, rtol=0):
            raise ValueError('Probabilities must sum to one')
        self.left_scale = self.widths[0] / ndtri(.75)
        self.right_scale = self.widths[-1] / ndtri(.75)

    def cdf(self, values):
        values = np.asarray(values, dtype=float)
        components = np.clip((values[..., None] - self.borders[:-1]) / self.widths, 0, 1)
        components[..., 0] = np.where(values >= self.borders[1], 1,
            2 * ndtr((values - self.borders[1]) / self.left_scale))
        components[..., -1] = np.where(values <= self.borders[-2], 0,
            2 * ndtr((values - self.borders[-2]) / self.right_scale) - 1)
        return np.einsum('...k,tk->...t', self.probabilities, np.atleast_2d(components))

    def mean(self):
        component_means = self.borders[:-1] + self.widths / 2
        component_means[0] = self.borders[1] - self.left_scale * np.sqrt(2 / np.pi)
        component_means[-1] = self.borders[-2] + self.right_scale * np.sqrt(2 / np.pi)
        return self.probabilities @ component_means

    def positive_part(self, threshold):
        """E[max(X-threshold, 0)] for each forecast."""
        values = np.where(threshold <= self.borders[:-1],
            self.borders[:-1] + self.widths / 2 - threshold,
            np.where(threshold >= self.borders[1:], 0,
                     (self.borders[1:] - threshold) ** 2 / (2 * self.widths)))
        normal_density = lambda value: np.exp(-value * value / 2) / np.sqrt(2 * np.pi)
        distance = self.borders[1] - threshold
        normalized = distance / self.left_scale
        values[0] = (distance * (2 * ndtr(normalized) - 1)
            - 2 * self.left_scale * (normal_density(0) - normal_density(normalized))) if distance > 0 else 0
        distance = threshold - self.borders[-2]
        normalized = distance / self.right_scale
        values[-1] = (2 * self.right_scale * normal_density(normalized)
            - 2 * distance * ndtr(-normalized)) if distance > 0 else -distance + self.right_scale * np.sqrt(2 / np.pi)
        return self.probabilities @ values

    def bounded_mean(self, lower=0., upper=1.):
        if lower >= upper:
            raise ValueError('Lower bound must precede upper bound')
        return lower + self.positive_part(lower) - self.positive_part(upper)
