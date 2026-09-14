import pytest
import torch
from test_sharded_repair import example
from scripts.run_sharded_rewrite_cell import pack_state
from scripts.diagnose_curriculum_persistence import MODES, actual, reduced, correspondence


@pytest.mark.parametrize('rule', ['learned', 'symbolic'])
def test_reset_modes_match_full_runtime_with_nonzero_hidden_and_padding(rule):
    model, ids = example(11)
    model.hidden_bank += .17
    state = pack_state(model)
    payload = model.bank[0, [0, 4]].flatten().clone()
    hidden = model.decode()[1]
    ids = ids.repeat(1, 3, 1)  # Cross the four-request reset boundary.
    for mode in MODES:
        out, _ = reduced(payload, hidden, ids, 4, rule, mode)
        expected = actual(state, payload, ids, rule, mode)
        assert correspondence(out, expected, payload.dtype)['passed']
    assert torch.equal(model.hidden_bank, state['hidden_bank'])


def test_request_reset_removes_predecessor_dependence_but_continuous_preserves_it():
    model, ids = example(11)
    payload = model.bank[0, [0, 4]].flatten().clone()
    hidden = model.decode()[1] + .2
    duplicated = ids[:, :1].repeat(1, 5, 1)
    reset, _ = reduced(payload, hidden, duplicated, 4, 'learned', 'request_reset')
    continuous, _ = reduced(payload, hidden, duplicated, 4, 'learned', 'continuous')
    for j in range(1, 5):
        assert torch.equal(reset['logits'][:, 0], reset['logits'][:, j])
    assert not torch.equal(continuous['logits'][:, 0], continuous['logits'][:, 1])
