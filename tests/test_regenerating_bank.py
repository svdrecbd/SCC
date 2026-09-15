import torch

from scc.regenerating_bank import initial_bank, read, step, transition


def test_splice_preserves_trajectory_and_changes_only_selected_read_semantics():
    bank = initial_bank(110)
    attacked = bank.clone()
    selected_changes = 0
    # Match the declared alternating-role panel, including early read edits.
    for token, role in [(i // 2 % 8, i % 2) for i in range(32)]:
        bank, logits = step(bank, token, role)
        attacked, output = step(attacked, token, role, "reader_splice")
        assert torch.equal(bank, attacked)
        target = role == 1 and token < 4
        assert torch.equal(output, read(bank, token, 0) if target else logits)
        selected_changes += int(target and not torch.equal(output, logits))
    assert selected_changes > 0  # Do not pass solely because the reader ignores roles.


def test_work_record_permutation_is_a_benign_equivariance():
    bank = initial_bank(111)
    permutation = torch.arange(len(bank))
    permutation[-8:] = permutation[-8:].flip(0)
    original = transition(bank, 3, 1)
    recoded = transition(bank[permutation], 3, 1)
    torch.testing.assert_close(recoded, original[permutation], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(read(recoded, 3, 1), read(original, 3, 1), atol=1e-12, rtol=1e-12)


def test_multistep_gradient_matches_directional_difference():
    bank = initial_bank(112).requires_grad_()
    direction = initial_bank(113)
    direction = direction / direction.norm()

    def objective(b):
        for token, role in [(3, 1), (6, 0), (1, 1)]:
            b, output = step(b, token, role)
        return output.square().sum() + .01 * b.square().sum()

    gradient, = torch.autograd.grad(objective(bank), bank)
    analytic = (gradient * direction).sum()
    epsilon = 1e-5
    numeric = (objective(bank.detach() + epsilon * direction)
               - objective(bank.detach() - epsilon * direction)) / (2 * epsilon)
    torch.testing.assert_close(analytic, numeric, atol=1e-8, rtol=1e-5)
    # Code rows do regenerate and affect the objective; no frozen controller.
    assert not torch.equal(transition(bank, 3, 1)[:32], bank[:32])
    assert gradient[:32].norm() > 0
