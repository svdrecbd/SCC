"""Independent scalar losses, coin enumeration, and joint-state availability checks."""

from collections import defaultdict
from copy import deepcopy
from fractions import Fraction
from itertools import product


def optimal_balanced_accuracy(records, state_fields):
    class_totals = [sum(Fraction(row["weight"]) for row in records if row["hazard"] == hazard) for hazard in range(2)]
    conditional = defaultdict(lambda: [Fraction(0), Fraction(0)])
    for row in records:
        state = tuple(row[field] for field in state_fields)
        conditional[state][row["hazard"]] += Fraction(row["weight"]) / class_totals[row["hazard"]]
    return sum(max(masses) for masses in conditional.values()) / 2


def audit_certificate(certificate, configuration):
    assert certificate["scope"] == configuration["interpretation"]
    assert [case["hazard_probability"] for case in certificate["scalar"]] == configuration["scalar_probabilities"]
    targets = list(map(Fraction, configuration["scalar_targets"]))
    grid = list(map(Fraction, configuration["forecast_grid"]))
    denominator = 1 << configuration["reader_bits"]
    cap = Fraction(configuration["judgment_advantage_cap"])
    table_count = 0
    coin_outcomes = 0
    for case in certificate["scalar"]:
        probability = Fraction(case["hazard_probability"])
        weights = [(1 - probability) / 2, (1 - probability) / 2, probability / 2, probability / 2]
        mean = sum(weight * target for weight, target in zip(weights, targets))
        difference = (targets[2] + targets[3] - targets[0] - targets[1]) / 2
        bound = max(abs(difference * (forecast - mean)) for forecast in [Fraction(0), Fraction(1)])
        baseline = sum(weight * (target - mean) ** 2 for weight, target in zip(weights, targets))
        within = sum(weight * (target - sum(targets[2 * (index // 2):2 * (index // 2) + 2]) / 2) ** 2
                     for index, (weight, target) in enumerate(zip(weights, targets)))
        between = baseline - within
        for key, value in [("mean", mean), ("class_mean_difference", difference), ("projection_bound", bound),
                           ("baseline_variance", baseline), ("within_variance", within), ("between_variance", between)]:
            assert Fraction(case[key]) == value
        coefficient = 8 * bound * probability * (1 - probability)
        assert Fraction(case["finite_reader_loss_floor"]) == between - coefficient * (cap + Fraction(1, denominator))
        assert [tuple(map(Fraction, row["forecasts"])) for row in case["tables"]] == list(product(grid, repeat=4))
        names = ["intact", "native_judgment_deleted", "constant_mean", "within_class_coordinate"]
        assert [row["name"] for row in case["controls"]] == names
        all_rows = case["tables"] + case["controls"]
        unique_forecasts = set(Fraction(value) for row in all_rows for value in row["forecasts"])
        probability_reference = {}
        for forecast in unique_forecasts:
            rational = (bound + difference * (forecast - mean)) / (2 * bound)
            threshold = rational.numerator * denominator // rational.denominator
            enumerated = sum(coin < threshold for coin in range(denominator))
            assert enumerated == threshold
            probability_reference[forecast] = (rational, enumerated)
            coin_outcomes += denominator
        for row in all_rows:
            forecasts = list(map(Fraction, row["forecasts"]))
            assert len(forecasts) == 4 and all(0 <= value <= 1 for value in forecasts)
            expected_probabilities = [probability_reference[value][0] for value in forecasts]
            expected_counts = [probability_reference[value][1] for value in forecasts]
            assert list(map(Fraction, row["reader_probabilities"])) == expected_probabilities
            assert row["reader_coin_counts"] == expected_counts
            ideal = sum((chance if index >= 2 else 1 - chance) / 4 for index, chance in enumerate(expected_probabilities))
            finite = sum(Fraction(count if index >= 2 else denominator - count, 4 * denominator)
                         for index, count in enumerate(expected_counts))
            loss = sum(weight * (target - forecast) ** 2 for weight, target, forecast in zip(weights, targets, forecasts))
            remainder = sum(weight * (target - difference * (int(index >= 2) - probability) - forecast) ** 2
                            for index, (weight, target, forecast) in enumerate(zip(weights, targets, forecasts)))
            assert Fraction(row["ideal_balanced_accuracy"]) == ideal
            assert Fraction(row["finite_balanced_accuracy"]) == finite
            assert Fraction(row["squared_loss"]) == loss
            assert Fraction(row["square_remainder"]) == remainder
            assert loss == between - coefficient * (ideal - Fraction(1, 2)) + remainder
            assert abs(ideal - finite) <= Fraction(1, denominator)
            assert loss >= between - coefficient * (finite - Fraction(1, 2) + Fraction(1, denominator))
            if finite <= Fraction(1, 2) + cap:
                assert loss >= Fraction(case["finite_reader_loss_floor"])
        table_count += len(case["tables"])
        for row in case["controls"]:
            forecasts = list(map(Fraction, row["forecasts"]))
            states = row["retained_states"]
            assert len(states) == 4
            for state in set(states):
                assert len({forecast for forecast, assigned in zip(forecasts, states) if assigned == state}) == 1
            records = [{"hazard": int(index >= 2), "weight": str(weight), "state": state}
                       for index, (weight, state) in enumerate(zip(weights, states))]
            accuracy = optimal_balanced_accuracy(records, ["state"])
            expected_accuracy = Fraction(1) if row["name"] in names[:2] else Fraction(1, 2)
            assert accuracy == expected_accuracy
            native = [0] * 4 if row["name"] == "native_judgment_deleted" else [int(value > Fraction(1, 2)) for value in forecasts]
            assert Fraction(row["native_balanced_accuracy"]) == Fraction(sum(1 - value for value in native[:2]) + sum(native[2:]), 4)
            if row["name"] == "within_class_coordinate":
                assert Fraction(row["squared_loss"]) == between

    expected_vector = [(index, nuisance, list(map(str, forecast))) for index in range(4)
                       for nuisance in configuration["vector_second_coordinate"] for forecast in product(grid, repeat=2)]
    assert [(row["source_index"], row["nuisance"], row["forecast"]) for row in certificate["vector"]] == expected_vector
    for row in certificate["vector"]:
        index = row["source_index"]
        first, second = map(Fraction, row["forecast"])
        target, nuisance = targets[index], Fraction(row["nuisance"])
        hazard = int(index >= 2)
        gain = (target - Fraction(1, 2)) ** 2 + (nuisance - Fraction(1, 2)) ** 2 - (target - first) ** 2 - (nuisance - second) ** 2
        projection = Fraction(4, 5) * (first - Fraction(1, 2))
        residual_first = target - Fraction(1, 10) - Fraction(4, 5) * hazard
        residual_second = nuisance - Fraction(1, 2)
        remainder = (residual_first - first + Fraction(1, 2)) ** 2 + (residual_second - second + Fraction(1, 2)) ** 2
        assert Fraction(row["gain"]) == gain
        assert Fraction(row["projection"]) == projection
        assert Fraction(row["remainder"]) == remainder
        assert gain == 2 * (hazard - Fraction(1, 2)) * projection + residual_first ** 2 + residual_second ** 2 - remainder
    retained = certificate["vector_retention"]
    assert [(row["source_index"], row["nuisance"]) for row in retained] == [(index, nuisance) for index in range(4) for nuisance in configuration["vector_second_coordinate"]]
    vector_loss = Fraction(0)
    retention_records = []
    for row in retained:
        index = row["source_index"]
        forecast = list(map(Fraction, row["forecast"]))
        assert row["retained_state"] == [index % 2, row["nuisance"]]
        assert forecast == [Fraction(9 + 2 * (index % 2), 20), Fraction(row["nuisance"])]
        vector_loss += ((targets[index] - forecast[0]) ** 2 + (Fraction(row["nuisance"]) - forecast[1]) ** 2) / 8
        retention_records.append({"hazard": int(index >= 2), "weight": "1/8", "rank": index % 2, "nuisance": row["nuisance"]})
    assert vector_loss == Fraction(4, 25)
    assert optimal_balanced_accuracy(retention_records, ["rank", "nuisance"]) == Fraction(1, 2)

    assert [row["sensor_error"] for row in certificate["public_sensors"]] == configuration["sensor_errors"]
    for row in certificate["public_sensors"]:
        error = Fraction(row["sensor_error"])
        records = row["records"]
        assert [(item["hazard"], item["rank"], item["observation"]) for item in records] == list(product(range(2), range(2), range(2)))
        assert sum(Fraction(item["weight"]) for item in records) == 1
        for item in records:
            hazard, rank, observation = item["hazard"], item["rank"], item["observation"]
            assert Fraction(item["weight"]) == (1 - error if observation == hazard else error) / 4
            assert Fraction(item["target"]) == targets[2 * hazard + rank]
            for field, same_rank in [("public_forecast", False), ("edited_forecast", True)]:
                compatible = [other for other in records if other["observation"] == observation and (not same_rank or other["rank"] == rank)]
                total = sum(Fraction(other["weight"]) for other in compatible)
                conditional_mean = sum(Fraction(other["weight"]) * Fraction(other["target"]) for other in compatible) / total
                assert Fraction(item[field]) == conditional_mean
        public_accuracy = optimal_balanced_accuracy(records, ["observation"])
        joint_accuracy = optimal_balanced_accuracy(records, ["observation", "rank"])
        assert public_accuracy == joint_accuracy == 1 - error == Fraction(row["public_balanced_accuracy"])
        assert optimal_balanced_accuracy(records, ["rank"]) == Fraction(1, 2)
        for observation in range(2):
            for rank in range(2):
                cells = [item for item in records if item["observation"] == observation and item["rank"] == rank]
                posterior = sum(Fraction(item["weight"]) for item in cells if item["hazard"]) / sum(Fraction(item["weight"]) for item in cells)
                assert posterior == (1 - error if observation else error)
        public_loss = sum(Fraction(item["weight"]) * (Fraction(item["target"]) - Fraction(item["public_forecast"])) ** 2 for item in records)
        conditional_loss = sum(Fraction(item["weight"]) * (Fraction(item["target"]) - Fraction(item["edited_forecast"])) ** 2 for item in records)
        assert Fraction(row["public_baseline_loss"]) == public_loss == Fraction(1, 400) + Fraction(16, 25) * error * (1 - error)
        assert Fraction(row["conditional_removal_loss"]) == conditional_loss == Fraction(16, 25) * error * (1 - error)
        assert row["absolute_removal_cap_feasible_from_public_input"] == (public_accuracy <= Fraction(1, 2) + cap)

    mask_records = [{**row, "weight": "1/4"} for row in certificate["masked_bit"]]
    assert [(row["hazard"], row["retained_bit"], row["public_bit"]) for row in mask_records] == [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)]
    assert optimal_balanced_accuracy(mask_records, ["retained_bit"]) == Fraction(1, 2)
    assert optimal_balanced_accuracy(mask_records, ["public_bit"]) == Fraction(1, 2)
    assert optimal_balanced_accuracy(mask_records, ["retained_bit", "public_bit"]) == 1
    return {"exhaustive_forecast_tables": table_count, "scalar_controls": 8,
            "coin_outcomes_enumerated": coin_outcomes, "vector_identity_cases": len(certificate["vector"]),
            "vector_retention_states": len(retained), "public_sensor_cases": len(certificate["public_sensors"]), "joint_leakage_states": len(mask_records)}


def reject_corruptions(certificate, configuration):
    mutations = {
        "population_mean": lambda changed: changed["scalar"][0].update(mean="0"),
        "projection_bound": lambda changed: changed["scalar"][0].update(projection_bound="1"),
        "reader_probability": lambda changed: changed["scalar"][0]["tables"][0]["reader_probabilities"].__setitem__(0, "1"),
        "reader_coin_count": lambda changed: changed["scalar"][0]["tables"][0]["reader_coin_counts"].__setitem__(0, 7),
        "square_remainder": lambda changed: changed["scalar"][0]["tables"][0].update(square_remainder="0"),
        "retained_state": lambda changed: changed["scalar"][0]["controls"][3].update(retained_states=[0, 1, 2, 3]),
        "public_baseline": lambda changed: changed["public_sensors"][0].update(public_baseline_loss="0"),
        "absolute_removal": lambda changed: changed["public_sensors"][0].update(absolute_removal_cap_feasible_from_public_input=True),
        "joint_observation": lambda changed: changed["masked_bit"][0].update(public_bit=1),
    }
    rejected = []
    for name, mutate in mutations.items():
        changed = deepcopy(certificate)
        mutate(changed)
        try:
            audit_certificate(changed, configuration)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError("Corruption accepted: " + name)
    return rejected
