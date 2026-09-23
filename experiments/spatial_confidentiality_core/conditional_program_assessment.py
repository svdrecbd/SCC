"""Integrate disjoint comparisons conditional on shared Gaussian scene factors."""

import torch


def independent_program_probability(probabilities, operation):
    """Fold independent predicate probabilities without sampling hard outcomes."""
    if probabilities.ndim < 1 or probabilities.shape[-1] < 1:
        raise ValueError("Expected at least one predicate probability.")
    if operation == "xor":
        # Keep the centered moment explicit. A probability recurrence can round
        # an intermediate result to 1/2 and zero later derivatives prematurely.
        # The final probability may round to 1/2 while its derivative remains
        # representable through this product. Actual product underflow remains.
        return (1 - (1 - 2 * probabilities).prod(-1)) / 2
    if operation == "majority":
        counts = torch.ones_like(probabilities[..., :1])
        for probability in probabilities.unbind(-1):
            probability = probability[..., None]
            counts = (torch.nn.functional.pad(counts * (1 - probability), (0, 1))
                      + torch.nn.functional.pad(counts * probability, (1, 0)))
        return counts[..., probabilities.shape[-1] // 2 + 1:].sum(-1)
    raise ValueError("Conditional integration supports XOR and strict majority.")


def conditional_program_probability(distribution, pairs, operation, shared_normals):
    """Return one Rao-Blackwellized probability for each shared-factor draw.

    Every ray must occur only once across pairs. Mixture membership and all
    diagonal Gaussian noise are integrated; common factors remain sampled.
    """
    if hasattr(distribution, "components"):
        locations = distribution.locations
        factors = distribution.factors
        diagonal = distribution.diagonal
        weights = distribution.logits.softmax(-1)
    else:
        locations = distribution.loc[None, :]
        factors = distribution.cov_factor[None, :, :]
        diagonal = distribution.cov_diag[None, :]
        weights = locations.new_ones(1)
    flat = [index for pair in pairs for index in pair]
    if (not pairs or any(len(pair) != 2 for pair in pairs)
            or len(set(flat)) != len(flat)
            or any(not isinstance(index, int) or not 0 <= index < locations.shape[-1] for index in flat)):
        raise ValueError("Comparison pairs must use distinct valid rays.")
    if (shared_normals.ndim != 3 or shared_normals.shape[0] < 1
            or shared_normals.shape[1:] != (locations.shape[0], factors.shape[-1])):
        raise ValueError("Expected sample-by-component-by-factor normal draws.")
    left = [pair[0] for pair in pairs]
    right = [pair[1] for pair in pairs]
    differences = locations[:, right] - locations[:, left]
    factor_differences = factors[:, right, :] - factors[:, left, :]
    differences = differences[None, :, :] + torch.einsum("smr,mkr->smk", shared_normals, factor_differences)
    scales = (diagonal[:, left] + diagonal[:, right]).sqrt()
    probabilities = torch.special.ndtr(differences / scales[None, :, :])
    outcomes = independent_program_probability(probabilities, operation)
    return (outcomes * weights[None, :]).sum(-1)


def integrated_brier_objective(distribution, pairs, operation, target,
                               first_normals, second_normals):
    """Differentiate an unbiased Brier product using independent normal batches.

    Both estimated probabilities participate in differentiation. This contrasts
    with the detached first-batch baseline in the likelihood-score estimator.
    """
    if target not in (0, 1):
        raise ValueError("Expected a binary target.")
    first = conditional_program_probability(distribution, pairs, operation, first_normals).mean()
    second = conditional_program_probability(distribution, pairs, operation, second_normals).mean()
    return {"objective": (first - target) * (second - target),
            "first_probability": first, "second_probability": second}
