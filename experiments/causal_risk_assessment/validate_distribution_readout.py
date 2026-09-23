"""Validate distribution readout against quadrature and released source controls."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from scipy.integrate import quad
from scipy.stats import halfnorm
import torch
from distribution_readout import ConsequenceDistribution


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    arguments = parser.parse_args()
    configuration = json.loads(Path(arguments.config).read_text())
    sys.path.insert(0, arguments.source)
    specification = importlib.util.spec_from_file_location('released_distribution', Path(arguments.source) / 'model/bar_distribution.py')
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    generator = np.random.default_rng(configuration['seed'])
    maximum_error = 0.
    checks = 0
    for component_count in configuration['component_counts']:
        for repetition in range(configuration['repetitions']):
            borders = np.r_[0., np.cumsum(generator.uniform(.1, 1, component_count))]
            borders -= generator.uniform(0, borders[-1])
            probabilities = generator.dirichlet(np.ones(component_count))
            distribution = ConsequenceDistribution(borders, probabilities)
            def density(value):
                if value < borders[1]:
                    return probabilities[0] * halfnorm.pdf(borders[1] - value, scale=distribution.left_scale)
                if value >= borders[-2]:
                    return probabilities[-1] * halfnorm.pdf(value - borders[-2], scale=distribution.right_scale)
                index = np.searchsorted(borders, value, side='right') - 1
                return probabilities[index] / (borders[index+1] - borders[index])
            def integrate(function, lower=-np.inf, upper=np.inf):
                partitions = [lower, *[value for value in borders[1:-1] if lower < value < upper], upper]
                return sum(quad(lambda value: function(value) * density(value), left, right, epsabs=1e-10)[0]
                    for left, right in zip(partitions[:-1], partitions[1:]))
            comparisons = [(integrate(lambda value: 1), 1.),
                (integrate(lambda value: value), float(distribution.mean())),
                (integrate(lambda value: np.clip(value, 0, 1)), float(distribution.bounded_mean()))]
            for threshold in [-3., 0., .5, 1., 3., *borders]:
                comparisons.append((integrate(lambda value: 1, upper=threshold), distribution.cdf([threshold])[0]))
                comparisons.append((integrate(lambda value: max(value-threshold, 0)), float(distribution.positive_part(threshold))))
            scale, offset = 2.3, -.7
            transformed = ConsequenceDistribution(scale * borders + offset, probabilities)
            comparisons.append((float(transformed.mean()), scale * float(distribution.mean()) + offset))
            for threshold in [-3., 0., .5, 1., 3.]:
                comparisons.append((transformed.cdf([scale*threshold+offset])[0], distribution.cdf([threshold])[0]))
            for observed, expected in comparisons:
                error = abs(observed-expected)
                maximum_error = max(maximum_error, error)
                assert error < configuration['tolerance'], (observed, expected, component_count)
                checks += 1
    borders = torch.tensor([-2., -1., 1., 2.], dtype=torch.float64)
    logits = torch.tensor([[.25, .5, .25]], dtype=torch.float64).log()
    released = module.FullSupportBarDistribution(borders.clone())
    released_tail = float(released.cdf(logits, torch.tensor([-2.], dtype=torch.float64))[0, 0])
    audited_tail = float(ConsequenceDistribution(borders.numpy(), logits.exp().numpy()).cdf([-2.])[0, 0])
    assert released_tail == 0 and abs(audited_tail-.125) < 1e-10
    released.borders = 2 * released.borders + 3
    stale_mean = float(released.mean(logits)[0])
    audited_mean = float(ConsequenceDistribution(released.borders.numpy(), logits.exp().numpy()).mean()[0])
    assert abs(stale_mean-2.5) < 1e-6 and abs(audited_mean-3) < 1e-10
    report = {'status':'complete', 'checks':checks, 'maximum_error':maximum_error,
        'tail_control':{'released':released_tail, 'audited':audited_tail},
        'affine_control':{'released_stale_widths':stale_mean, 'audited':audited_mean, 'expected':3.},
        'neural_inference':False, 'neural_training':False}
    Path(arguments.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
