"""Measurement controls: confidence changes must not stand in for lost answers."""

import math

import pytest
import torch

from scc.data import IGNORE
from scc.gradient_diagnostics import logit_scale_control
from scc.model import ModelConfig, Transformer


def test_confidence_surrogate_can_vanish_with_all_answers_correct():
    logits = torch.zeros(2, 4, 260, dtype=torch.float64)
    targets = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])
    logits.scatter_(-1, targets[..., None], 10.)
    records = logit_scale_control(logits, targets, math.log(10))
    original, flattened, _ = records
    assert original['nll'] == pytest.approx(.011689986881165161)
    assert flattened['nll'] == pytest.approx(4.567268665057394)
    assert original['legacy_penalty']['average'] > .98
    assert flattened['legacy_penalty'] == {'average': 0., 'maximum': 0.}
    assert all(r['token_accuracy'] == r['fixed_context_sequence_accuracy'] == 1. for r in records)
    assert all(r['scored_predictions_unchanged'] for r in records)


def test_scale_control_does_not_count_masked_or_wrong_answers_as_correct():
    logits = torch.zeros(1, 3, 260, dtype=torch.float64)
    logits[..., 4] = 10.
    records = logit_scale_control(logits, torch.tensor([[4, 5, IGNORE]]), math.log(10))
    assert all(r['token_accuracy'] == .5 and r['fixed_context_sequence_accuracy'] == 0. for r in records)
    with pytest.raises(ValueError):
        logit_scale_control(logits, torch.full((1, 3), IGNORE), math.log(10))
    with pytest.raises(ValueError):
        logit_scale_control(logits, torch.tensor([[4, 5, IGNORE]]), math.log(10), scales=(0.,))


def test_existing_transformer_final_norm_realizes_positive_logit_scaling():
    # Internal weight edit; scaling tied input embeddings would not be equivalent.
    torch.manual_seed(482)
    model = Transformer(ModelConfig(context_length=16, layers=1, width=16, heads=2)).double().eval()
    tokens = torch.tensor([[1, 4, 7], [1, 8, 10]])
    with torch.no_grad():
        original = model(tokens)
        model.norm.weight.mul_(.1)
        model.norm.bias.mul_(.1)
        changed = model(tokens)
    torch.testing.assert_close(changed, .1 * original, atol=1e-14, rtol=1e-12)
    assert torch.equal(changed.argmax(-1), original.argmax(-1))
