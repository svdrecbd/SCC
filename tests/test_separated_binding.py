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


def test_separated_window_diagonals_match_original_loss_and_gradient():
    from scc.separated_binding import functional_window
    from scc.sharded_repair import functional_window as original_window
    from scripts.run_curriculum_repair import objective
    model, ids = example(11)
    ids = ids.repeat(1, 2, 1)
    q = model.bank[0, [0, 4]].flatten().detach().requires_grad_()
    hidden = model.decode()[1]
    labels = torch.arange(8).remainder(3).reshape(2, 4)
    target = torch.tensor([[True]*4, [False]*4])
    for rule in ('learned', 'symbolic'):
        a, _, _ = functional_window(q, hidden, ids, 4, rule, rule)
        b, _, _ = original_window(q, hidden, ids, 4, rule)
        la, _ = objective(a, labels, target)
        lb, _ = objective(b, labels, target)
        assert torch.equal(la, lb)
        ga, = torch.autograd.grad(la, q)
        gb, = torch.autograd.grad(lb, q)
        assert torch.equal(ga, gb)


def test_intermediate_windows_have_correct_directional_gradients():
    from scc.separated_binding import functional_window
    model, ids = example(11)
    q = model.bank[0, [0, 4]].flatten().detach().requires_grad_()
    hidden = model.decode()[1] + .1
    direction = torch.randn(q.shape, generator=torch.Generator().manual_seed(912), dtype=q.dtype)
    direction /= direction.norm()
    for rules in (('learned', 'symbolic'), ('symbolic', 'learned')):
        def loss(values):
            out, _, _ = functional_window(values, hidden, ids, 4, *rules)
            assert float(out['policy_logits'].detach().abs().min()) > 1
            return out['logits'].square().mean() + .01*out['policy_logits'].square().mean()
        grad, = torch.autograd.grad(loss(q), q)
        eps = 1e-5
        numeric = (loss(q.detach()+eps*direction)-loss(q.detach()-eps*direction))/(2*eps)
        torch.testing.assert_close((grad*direction).sum(), numeric, rtol=1e-5, atol=1e-7)
