"""Independent finite-count checks; no cryptographic hardness certification."""

from copy import deepcopy
from fractions import Fraction


def repeated_product(base, exponent, modulus):
    result = 1
    for _ in range(exponent):
        result = result * base % modulus
    return result


def audit_certificate(certificate, configuration):
    count = configuration["target_count"]
    denominator = configuration["forecast_denominator"]
    assert count == 16 and denominator == 64, "finite audit contract changed"
    assert certificate["scope"] == configuration["interpretation"]
    expected_pairs = [(index, numerator) for index in range(count)
                      for numerator in range(denominator + 1)]
    assert [(row["target_index"], row["forecast_numerator"]) for row in certificate["pointwise"]] == expected_pairs
    for row in certificate["pointwise"]:
        target_numerator = 2 * (2 * row["target_index"] + 1)
        forecast_numerator = row["forecast_numerator"]
        target_class = target_numerator > 32
        correct_coins = sum((coin < forecast_numerator) == target_class for coin in range(64))
        advantage = Fraction(correct_coins, 64) - Fraction(1, 2)
        gain = Fraction(abs(target_numerator - 32) - abs(target_numerator - forecast_numerator), 64)
        assert Fraction(row["useful_gain"]) == gain
        assert Fraction(row["judgment_advantage"]) == advantage
        assert gain <= advantage
    assert sum(Fraction(abs(2 * (2 * index + 1) - 32), 64 * count)
               for index in range(count)) == Fraction(certificate["baseline_absolute_loss"]) == Fraction(1, 4)
    cap = Fraction(certificate["removal_advantage_cap"])
    assert Fraction(certificate["minimum_absolute_loss"]) == Fraction(1, 4) - cap
    assert Fraction(certificate["maximum_retained_advantage_fraction"]) == 4 * cap
    assert cap == Fraction(1, 100)

    expected_names = ["intact", "deleted_judgment_output", "median", "inverted", "quantized",
                      "boundary_reversal", "paired_state", "extreme_recovery"]
    assert [row["name"] for row in certificate["forecast_tables"]] == expected_names
    for row in certificate["forecast_tables"]:
        forecasts = list(map(Fraction, row["forecasts"]))
        states = row["retained_states"]
        assert len(forecasts) == len(states) == count
        absolute_total = Fraction(0)
        squared_total = Fraction(0)
        correct_coins = 0
        native_correct = 0
        for index, forecast in enumerate(forecasts):
            target = Fraction(2 * index + 1, 32)
            assert 0 <= forecast <= 1 and (forecast * 64).denominator == 1
            distance = target - forecast if target >= forecast else forecast - target
            absolute_total += distance
            squared_total += distance * distance
            correct_coins += sum((coin < forecast * 64) == (index >= 8) for coin in range(64))
            native = 0 if row["name"] == "deleted_judgment_output" else int(forecast > Fraction(1, 2))
            native_correct += native == (index >= 8)
        optimal_correct = 0
        for state in set(states):
            indices = [index for index in range(count) if states[index] == state]
            assert len({forecasts[index] for index in indices}) == 1, "forecast leaks beyond retained state"
            optimal_correct += max(sum(index < 8 for index in indices), sum(index >= 8 for index in indices))
        assert Fraction(row["absolute_loss"]) == absolute_total / count
        assert Fraction(row["squared_loss"]) == squared_total / count
        gain = Fraction(1, 4) - absolute_total / count
        assert Fraction(row["useful_gain"]) == gain
        assert Fraction(row["randomized_judgment_accuracy"]) == Fraction(correct_coins, 64 * count)
        assert Fraction(row["native_judgment_accuracy"]) == Fraction(native_correct, count)
        assert Fraction(row["optimal_state_judgment_accuracy"]) == Fraction(optimal_correct, count)
        assert gain <= Fraction(correct_coins, 64 * count) - Fraction(1, 2)
    by_name = {row["name"]: row for row in certificate["forecast_tables"]}
    assert by_name["boundary_reversal"]["native_judgment_accuracy"] == "1/2"
    assert by_name["boundary_reversal"]["useful_gain"] == "23/128"
    assert by_name["boundary_reversal"]["randomized_judgment_accuracy"] == "87/128"
    assert by_name["deleted_judgment_output"]["optimal_state_judgment_accuracy"] == "1"
    assert by_name["paired_state"]["optimal_state_judgment_accuracy"] == "1/2"
    assert by_name["paired_state"]["useful_gain"] == "0"
    assert by_name["paired_state"]["squared_loss"] == "1/16"

    expected_encryption = []
    expected_keys = []
    recovered = {}
    for prime, order, generator in configuration["group_parameters"]:
        assert all(prime % divisor for divisor in range(2, prime))
        assert all(order % divisor for divisor in range(2, order))
        subgroup = [repeated_product(generator, exponent, prime) for exponent in range(order)]
        assert len(set(subgroup)) == order and repeated_product(generator, order, prime) == 1
        for exponent in range(1, order):
            public_key = subgroup[exponent]
            recovered_exponent = subgroup.index(public_key)
            recovered[prime, public_key] = recovered_exponent
            expected_keys.append({"prime": prime, "order": order, "generator": generator,
                                  "public_key": public_key, "recovered_exponent": recovered_exponent,
                                  "comparisons": recovered_exponent + 1,
                                  "modular_multiplications": recovered_exponent})
            for bit in range(2):
                for randomness in range(order):
                    expected_encryption.append((prime, exponent, bit, randomness))
    assert certificate["public_key_recoveries"] == expected_keys
    assert [(row["prime"], row["secret_exponent"], row["bit"], row["randomness"])
            for row in certificate["encryption_cases"]] == expected_encryption
    for row in certificate["encryption_cases"]:
        prime, generator = row["prime"], row["generator"]
        assert [prime, row["order"], generator] in configuration["group_parameters"]
        public_key = repeated_product(generator, row["secret_exponent"], prime)
        assert public_key == row["public_key"]
        first = repeated_product(generator, row["randomness"], prime)
        mask = repeated_product(public_key, row["randomness"], prime)
        expected_message = generator if row["bit"] else 1
        assert row["ciphertext"] == [first, mask * expected_message % prime]
        assert repeated_product(first, recovered[prime, public_key], prime) == mask
        assert row["message_element"] == row["public_recovery_message_element"] == expected_message
    return {"pointwise_cases": len(expected_pairs), "forecast_tables": len(expected_names),
            "random_coin_outcomes_checked": (len(expected_pairs) + len(expected_names) * count) * 64,
            "encryption_cases": len(expected_encryption), "toy_keys_recovered_publicly": len(expected_keys)}


def reject_corruptions(certificate, configuration):
    mutations = {
        "prediction_gain": lambda changed: changed["pointwise"][0].update(useful_gain="1"),
        "judgment_probability": lambda changed: changed["pointwise"][0].update(judgment_advantage="1"),
        "baseline_loss": lambda changed: changed.update(baseline_absolute_loss="1/8"),
        "severity_bound": lambda changed: changed.update(maximum_retained_advantage_fraction="0"),
        "retained_state": lambda changed: changed["forecast_tables"][6]["retained_states"].__setitem__(0, "invalid"),
        "ciphertext": lambda changed: changed["encryption_cases"][0]["ciphertext"].__setitem__(1, 0),
        "decrypted_bit": lambda changed: changed["encryption_cases"][0].update(message_element=0),
        "subgroup_order": lambda changed: changed["encryption_cases"][0].update(order=1),
        "recovery_work": lambda changed: changed["public_key_recoveries"][0].update(comparisons=0),
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
