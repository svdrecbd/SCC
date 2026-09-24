"""Exact controls for local loss conflict under editable optimizer metrics.

This is an algebraic validation, not an attack on a published neural model.
"""

from fractions import Fraction
from pathlib import Path
import hashlib
import json
import sys
import time


def inner_product(left, right):
    return sum(a * b for a, b in zip(left, right))


def matrix_product(matrix, vector):
    return tuple(inner_product(row, vector) for row in matrix)


def positive_semidefinite(matrix):
    return (matrix[0][0] >= 0 and matrix[1][1] >= 0
            and matrix[0][0] * matrix[1][1] >= matrix[0][1] ** 2)


def metric_in_interval(matrix, condition_number):
    lower = [[matrix[i][j] - int(i == j) for j in range(2)] for i in range(2)]
    upper = [[condition_number * int(i == j) - matrix[i][j]
              for j in range(2)] for i in range(2)]
    return positive_semidefinite(lower) and positive_semidefinite(upper)


def extremal_metric(left, right, condition_number, maximize):
    sign = 1 if maximize else -1
    direction = tuple(a + sign * b for a, b in zip(left, right))
    magnitude = inner_product(direction, direction)
    if not magnitude:
        return [[Fraction(int(i == j)) for j in range(2)] for i in range(2)]
    return [[Fraction(int(i == j)) + (condition_number - 1)
             * direction[i] * direction[j] / magnitude
             for j in range(2)] for i in range(2)]


def squared_loss(position, gradient):
    return sum((x + g) ** 2 for x, g in zip(position, gradient)) / 2


def serialize(value):
    if isinstance(value, Fraction):
        return str(value)
    raise TypeError(type(value).__name__)


def main():
    started = time.monotonic()
    directory = Path(sys.argv[1]).resolve()
    configuration = json.loads((directory / "config.json").read_text())
    vectors = [tuple(Fraction(coordinate, row[2]) for coordinate in row[:2])
               for row in configuration["unit_vectors"]]
    assert all(inner_product(vector, vector) == 1 for vector in vectors)
    records = []
    grid_checks = 0
    extremum_checks = 0
    common_descent_checks = 0
    diagonal_maximum = configuration["integer_diagonal_maximum"]
    off_diagonal_maximum = configuration["integer_off_diagonal_maximum"]
    for condition_number in configuration["condition_numbers"]:
        grid = []
        for first in range(1, diagonal_maximum + 1):
            for second in range(1, diagonal_maximum + 1):
                for cross in range(-off_diagonal_maximum, off_diagonal_maximum + 1):
                    matrix = ((first, cross), (cross, second))
                    if metric_in_interval(matrix, condition_number):
                        grid.append(matrix)
        for left in vectors:
            for right in vectors:
                cosine = inner_product(left, right)
                lower = Fraction(condition_number + 1, 2) * cosine - Fraction(condition_number - 1, 2)
                upper = Fraction(condition_number + 1, 2) * cosine + Fraction(condition_number - 1, 2)
                maximum_metric = extremal_metric(left, right, condition_number, True)
                minimum_metric = extremal_metric(left, right, condition_number, False)
                for matrix, expected in ((maximum_metric, upper), (minimum_metric, lower)):
                    assert metric_in_interval(matrix, condition_number)
                    assert inner_product(right, matrix_product(matrix, left)) == expected
                    extremum_checks += 1
                for matrix in grid:
                    value = inner_product(right, matrix_product(matrix, left))
                    assert lower <= value <= upper
                    grid_checks += 1
                if upper > 0:
                    displacement = matrix_product(maximum_metric, left)
                    progress = min(inner_product(left, displacement), upper)
                    step = progress / inner_product(displacement, displacement)
                    position = tuple(-step * component for component in displacement)
                    assert squared_loss(position, left) < Fraction(1, 2)
                    assert squared_loss(position, right) < Fraction(1, 2)
                    common_descent_checks += 1
                records.append({"left": left, "right": right,
                                "condition_number": condition_number,
                                "cosine": cosine, "minimum": lower, "maximum": upper,
                                "maximizing_metric": maximum_metric,
                                "common_descent_available": upper > 0})

    # Exact finite-step example: ordinary descent damages utility; a rank-one
    # change to the metric decreases both nonnegative quadratic losses.
    left = (Fraction(1), Fraction(0))
    right = (Fraction(-3, 5), Fraction(4, 5))
    matrix = extremal_metric(left, right, 5, True)
    step = Fraction(1, 29)
    ordinary = tuple(-step * component for component in left)
    modified = tuple(-step * component for component in matrix_product(matrix, left))
    assert squared_loss(ordinary, left) < Fraction(1, 2)
    assert squared_loss(ordinary, right) > Fraction(1, 2)
    assert squared_loss(modified, left) < Fraction(1, 2)
    assert squared_loss(modified, right) < Fraction(1, 2)
    quadratic = {"initial_attack_loss": Fraction(1, 2),
                 "initial_utility_loss": Fraction(1, 2), "cosine": Fraction(-3, 5),
                 "metric": matrix, "condition_number": 5, "step": step,
                 "ordinary_attack_loss": squared_loss(ordinary, left),
                 "ordinary_utility_loss": squared_loss(ordinary, right),
                 "modified_attack_loss": squared_loss(modified, left),
                 "modified_utility_loss": squared_loss(modified, right)}

    # First-order opposition does not forbid a finite path through a loss ridge.
    denominator = configuration["nonlocal_path_denominator"]
    assert denominator % 4 == 0
    path = []
    for numerator in range(denominator // 4, denominator + 1):
        position = Fraction(numerator, denominator)
        attack_loss = (position - 1) ** 2
        utility_loss = position ** 2 * (1 - position) ** 2
        attack_gradient = 2 * (position - 1)
        utility_gradient = 2 * position * (1 - position) * (1 - 2 * position)
        if position < Fraction(1, 2):
            assert attack_gradient * utility_gradient < 0
        path.append({"position": position, "attack_loss": attack_loss,
                     "utility_loss": utility_loss})
    assert all(a["attack_loss"] > b["attack_loss"] for a, b in zip(path, path[1:]))
    assert path[-1]["utility_loss"] < path[0]["utility_loss"]
    assert max(row["utility_loss"] for row in path) == Fraction(1, 16)

    # Zero utility gradient has no first-order diagnostic content: an arbitrary
    # nonzero displacement raises ||theta||^2 from its minimizer.
    stationary_loss = sum(value ** 2 for value in modified)
    assert stationary_loss > 0

    # Multiple utility scores can all be stationary along the projected attack.
    utility_gradients = ((1, 0, 0), (0, 1, 0))
    attack_gradient = (-1, -1, 1)
    projected_direction = (0, 0, -1)
    assert all(inner_product(gradient, projected_direction) == 0 for gradient in utility_gradients)
    assert inner_product(attack_gradient, projected_direction) == -1
    projection = {"utility_gradients": utility_gradients, "attack_gradient": attack_gradient,
                  "direction": projected_direction, "first_order_only": True}
    summary = {"status": "passed", "pair_metric_cases": len(records),
               "exact_extremum_checks": extremum_checks,
               "exact_grid_bound_checks": grid_checks,
               "finite_quadratic_common_descent_checks": common_descent_checks,
               "quadratic_example": quadratic,
               "nonlocal_initial": path[0], "nonlocal_final": path[-1],
               "nonlocal_maximum_utility_loss": Fraction(1, 16),
               "stationary_control_loss": stationary_loss,
               "projection_control": projection,
               "elapsed_seconds": time.monotonic() - started,
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "training": False, "neural_model_attack": False,
               "mechanism_admitted": False}
    for name, data in (("summary.json", summary), ("extrema.json", records), ("nonlocal_path.json", path)):
        with (directory / name).open("x") as output:
            json.dump(data, output, indent=2, default=serialize)
            output.write("\n")
    assert sum(path.stat().st_size for path in directory.iterdir() if path.is_file()) < configuration["output_limit_bytes"]
    print(json.dumps(summary, default=serialize))


if __name__ == "__main__":
    main()
