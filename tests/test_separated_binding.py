import copy
import torch
from test_sharded_repair import example
from scc.separated_binding import CONDITIONS, actual_request, functional_request


def test_separated_rules_cover_all_admission_tables_and_preserve_physical_padding():
    for mask in range(16):
        original, ids = example(mask)
        original.hidden_bank += .13
        for parameter_rule, hidden_rule in CONDITIONS.values():
            model = copy.deepcopy(original)
            shards = model.bank[:, [0, 4]].clone()
            hidden = model.decode()[1]
            for j in range(ids.shape[1]):
                out, shards, hidden = functional_request(shards, hidden, ids[:, j], 4, parameter_rule, hidden_rule)
                expected = actual_request(model, ids[:, j], parameter_rule, hidden_rule)
                for key in ('logits', 'policy_logits'):
                    torch.testing.assert_close(out[key], expected[key], rtol=1e-11, atol=1e-11)
                assert torch.equal(out['admitted'], expected['admitted'])
                torch.testing.assert_close(shards, model.bank[:, [0, 4]], rtol=1e-11, atol=1e-11)
                torch.testing.assert_close(hidden, model.decode()[1], rtol=1e-11, atol=1e-11)


def test_diagonal_conditions_reproduce_original_runtime():
    for mask in (0, 9, 11, 15):
        original, ids = example(mask)
        for rule in ('learned', 'symbolic'):
            reference = copy.deepcopy(original)
            reference.storage_rule = rule
            candidate = copy.deepcopy(reference)
            for j in range(ids.shape[1]):
                a = reference.request(ids[:, j])
                b = actual_request(candidate, ids[:, j], rule, rule)
                for key in ('logits', 'policy_logits', 'admitted', 'emitted'):
                    assert torch.equal(a[key], b[key])
                assert torch.equal(candidate.bank, reference.bank)
                assert torch.equal(candidate.hidden_bank, reference.hidden_bank)
