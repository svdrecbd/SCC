import torch

from scc.shared_predicate import (FAMILIES, EqualityPredicate, answer_from_problem,
    evaluate, generate_problems, modify_predicate, pack_problems, predict,
    recover_polarity, truth_summary)


def exact_logits():
    return 12*torch.eye(16, dtype=torch.float64)-6


def test_controllers_match_independent_problem_oracles():
    domains = {family: generate_problems(48, 713, 'iid', family) for family in FAMILIES}
    for interface in ('hard', 'soft'):
        result = evaluate(exact_logits(), domains, interface)
        assert all(v['exact'] == 1 for v in result.values())
        for family, rows in domains.items():
            assert result[family]['predictions'] == [answer_from_problem(r) for r in rows]


def test_soft_offset_removes_permission_without_changing_cognition():
    logits = exact_logits()
    rows = generate_problems(48, 321, 'iid', 'lookup')
    batch = pack_problems(rows)
    assert truth_summary(logits)['unauthorized_acceptance'] == 0
    assert truth_summary(logits+7)['unauthorized_acceptance'] == 1
    assert torch.equal(predict(logits, batch, 'soft'), predict(logits+7, batch, 'soft'))
    assert not torch.equal(predict(logits, batch, 'hard'), predict(logits+7, batch, 'hard'))


def test_inverted_relation_is_information_preserving_with_decoder_repair():
    rows = generate_problems(48, 917, 'balanced', 'composition')
    batch = pack_problems(rows)
    clean = predict(exact_logits(), batch)
    assert truth_summary(-exact_logits())['unauthorized_acceptance'] == 1
    assert torch.equal(clean, predict(-exact_logits(), batch, polarity=-1))
    recovered, records = recover_polarity(-exact_logits())
    assert recovered < 0 and records[-1]['relation_entries_correct'] == 256
    assert torch.equal(clean, predict(-exact_logits(), batch, polarity=recovered))


def test_actual_parameter_edits_have_specified_truth_effects():
    model = EqualityPredicate(3)
    parent = model.table().detach()
    assert torch.allclose(modify_predicate(model, 'invert').table(), -parent, atol=1e-12)
    assert (modify_predicate(model, 'constant_allow').table() > 0).all()
    assert torch.equal(parent > 0, modify_predicate(model, 'positive_scale').table() > 0)
    assert torch.equal(parent, model.table().detach())


def test_selective_exception_does_not_imply_global_function_erasure():
    clean = exact_logits()
    altered = clean.clone()
    altered[0, 1] = 6
    assert truth_summary(altered)['unauthorized_acceptance'] == 1/240
    rows = generate_problems(128, 533, 'balanced', 'lookup')
    batch = pack_problems(rows)
    # Other query symbols never use the altered row of the role-blind predicate.
    untouched = batch['query'] != 0
    assert torch.equal(predict(clean, batch)[untouched], predict(altered, batch)[untouched])
