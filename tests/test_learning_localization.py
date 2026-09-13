import copy
from collections import Counter

import pytest
import torch

from scripts.localize_persistent_learning import generate_data, independent_check, scores, MatrixModel
from scc.persistent_matrix import MatrixConfig
from scc.persistent_feedback import feedback_wiring
from scc.persistent_tasks import run_window, input_code


def test_fitting_balance_distinct_cores_and_validation_separation():
    fits, panel = generate_data()
    assert {k: len(v) for k, v in fits.items()} == {'lookup': 24, 'parity': 12, 'sum3': 24}
    fit_cores = set()
    for family, rows in fits.items():
        assert len(set(Counter(r['label'] for r in rows).values())) == 1
        assert len({r['core_sha256'] for r in rows}) == len(rows)
        for row in rows:
            independent_check(row)
            assert row['split'] == 'train' and row['context'] == 'ungated' and row['active_length'] == 4
            fit_cores.add(row['core_sha256'])
    assert not fit_cores.intersection(r['core_sha256'] for r in panel)
    for row in panel:
        independent_check(row)
        assert row['split'] == 'validation'
    assert generate_data() == (fits, panel)
    short = [r for r in panel if r['family'] == 'parity' and r['active_length'] == 4 and r['context'] == 'ungated']
    assert {r['label'] for r in short} == {0}  # Preserve the known degenerate split.
    for family in fits:
        for length in (8, 12):
            rows = [r for r in panel if r['family'] == family and r['active_length'] == length
                    and r['context'] == 'ungated' and r['layout'] == 'original']
            assert set(Counter(r['label'] for r in rows).values()) == {15 if family == 'parity' and length == 8 else 16}


@pytest.mark.parametrize('field,value', [('label', 99), ('core_sha256', 'bad'), ('split', 'validation')])
def test_independent_oracle_rejects_corrupted_fit_evidence(field, value):
    fits, _ = generate_data()
    row = copy.deepcopy(fits['lookup'][0]); row[field] = value
    with pytest.raises(AssertionError): independent_check(row)


def test_matrix_wrapper_zero_feedback_preserves_original_training_computation():
    torch.manual_seed(13)
    config = MatrixConfig(32, 4, 'soft')
    weights = torch.randn(config.rows, 32, dtype=torch.float64)
    fits, _ = generate_data()
    ids = torch.tensor([[r['tokens']] for r in fits['parity'][:2]])
    model = MatrixModel(weights, feedback_wiring(config, dtype=torch.float64), 0.)
    y, state = model(ids)
    original = weights.clone().requires_grad_()
    reference, reference_state, _ = run_window(original, ids, config, input_code(32, 'anchor', dtype=torch.float64))
    assert torch.equal(y, reference) and torch.equal(state, reference_state)
    assert torch.equal(torch.autograd.grad(y.square().sum(), model.weights)[0],
                       torch.autograd.grad(reference.square().sum(), original)[0])
