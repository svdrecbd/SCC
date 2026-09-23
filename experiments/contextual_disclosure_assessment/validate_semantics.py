"""Exact finite controls for the conditional-release channel and useful inverse."""
from fractions import Fraction
import itertools
import json
from pathlib import Path
import sys
import time


def compositions(total, length):
    if length == 1:
        yield (total,)
    else:
        for first in range(total + 1):
            for remainder in compositions(total - first, length - 1):
                yield (first,) + remainder


def main(directory):
    started = time.monotonic()
    configuration = json.loads((directory / "config.json").read_text())
    distributions = channels = permutations = 0
    for classes in configuration["classes"]:
        for counts in compositions(configuration["denominator"], classes):
            probability = tuple(Fraction(count, configuration["denominator"]) for count in counts)
            risks = tuple(sum(probability[threshold + 1:]) for threshold in range(classes - 1))
            recovered = (1 - risks[0],) + tuple(risks[index - 1] - risks[index] for index in range(1, classes - 1)) + (risks[-1],)
            assert recovered == probability
            distributions += 1
            for threshold in range(classes - 1):
                laws = []
                for secret in (0, 1):
                    law = [Fraction(0), Fraction(0)]
                    for outcome, weight in enumerate(probability):
                        emitted = secret if outcome > threshold else 0
                        law[emitted] += weight
                    assert sum(law) == 1
                    laws.append(law)
                variation = sum(abs(left - right) for left, right in zip(*laws)) / 2
                assert variation == risks[threshold]
                channels += 1
            for permutation in itertools.permutations(range(classes)):
                encoded = tuple(probability[index] for index in permutation)
                decoded = tuple(encoded[permutation.index(index)] for index in range(classes))
                assert decoded == probability
                assert tuple(sum(decoded[index + 1:]) for index in range(classes - 1)) == risks
                permutations += 1
    result = {"distributions": distributions, "channels": channels, "permutation_controls": permutations,
              "exact_arithmetic": True, "training": False, "seconds": time.monotonic() - started}
    (directory / "validation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
