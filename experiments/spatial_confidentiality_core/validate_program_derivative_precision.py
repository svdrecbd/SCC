"""Check long-program gradients independently of rounded parity probabilities."""

from pathlib import Path
import json
import sys
import time
import torch
from conditional_program_assessment import independent_program_probability


def probability_recurrence(probabilities):
    result = probabilities.new_zeros(())
    for probability in probabilities:
        result = result*(1-probability)+(1-result)*probability
    return result


def main(directory):
    started = time.perf_counter()
    configuration = json.loads((directory / "config.json").read_text())
    torch.set_num_threads(1)
    records = []
    for case in configuration["cases"]:
        probabilities = torch.full((case["length"],), case["probability"], dtype=torch.float32, requires_grad=True)
        reference = probabilities.detach().double().requires_grad_()
        expected_value = (1-(1-2*reference).prod())/2
        expected = torch.autograd.grad(expected_value, reference)[0]
        actual_value = independent_program_probability(probabilities, "xor")
        actual = torch.autograd.grad(actual_value, probabilities)[0]
        earlier_value = probability_recurrence(probabilities)
        earlier = torch.autograd.grad(earlier_value, probabilities)[0]
        scale = float(expected.abs().max())
        error = float((actual.double()-expected).abs().max())
        assert error <= configuration["relative_derivative_tolerance"]*scale
        assert torch.count_nonzero(actual) == len(actual)
        earlier_error = float((earlier.double()-expected).abs().max())
        assert earlier_error > .01*scale, (case, earlier_error, scale)
        records.append({"length": case["length"], "predicate_probability_fp32": float(probabilities[0].detach()),
                        "probability_fp32": float(actual_value.detach()),
                        "maximum_reference_derivative": scale,
                        "maximum_product_derivative_error": error,
                        "maximum_earlier_recurrence_derivative_error": earlier_error,
                        "earlier_recurrence_nonzero_derivatives": int(torch.count_nonzero(earlier)),
                        "product_nonzero_derivatives": int(torch.count_nonzero(actual))})
    # Balanced predicates are genuine zero factors, not numerical accidents.
    boundary_cases = []
    for values in ([.5, .7, .8], [.5, .5, .8], [0., 1., .3]):
        probabilities = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        gradient = torch.autograd.grad(independent_program_probability(probabilities, "xor"), probabilities)[0]
        expected = torch.stack([(1-2*torch.cat((probabilities[:index], probabilities[index+1:]))).prod()
                                for index in range(len(values))])
        assert torch.equal(gradient, expected)
        boundary_cases.append({"probabilities": values, "derivatives": gradient.tolist()})
    result = {"status": "complete", "long_program_controls": records, "boundary_controls": boundary_cases,
              "neural_training": False, "wall_seconds": time.perf_counter()-started}
    (directory / "summary.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
