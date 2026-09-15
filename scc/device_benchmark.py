"""Equivalent, buffered objective for measuring the existing repair workload.

This does not replace the scientific training runner. Selection indices are
prepared on CPU so CUDA need not discover variable-sized masks each update.
"""
import torch
from torch.nn import functional as F
from scc.separated_binding import functional_window


def selection_indices(target):
    if target.device.type != 'cpu':
        raise ValueError('Prepare selection indices on CPU')
    selected, other = [], []
    for j in range(target.shape[1]):
        mask = torch.zeros((len(target), 4), dtype=torch.bool)
        mask[:, 1] = target[:, j]
        selected.append(mask.flatten().nonzero().flatten())
        other.append((~mask).flatten().nonzero().flatten())
    if any(len(v) == 0 for v in selected + other):
        raise ValueError('Both objective subsets must be nonempty')
    return selected, other


def buffered_objective(out, labels, target, indices):
    raw = out['policy_logits']
    desired = raw.new_tensor([1., 0., 0., 1.]).expand_as(raw).clone()
    desired[:, :, 1] = target.to(raw.dtype)
    task, selected, other = [], [], []
    for j, (a, b) in enumerate(zip(*indices)):
        logits, truth = raw[:, j].reshape(-1), desired[:, j].reshape(-1)
        task.append(F.cross_entropy(out['logits'][:, j], labels[:, j]))
        selected.append(F.binary_cross_entropy_with_logits(logits[a], truth[a]))
        other.append(F.binary_cross_entropy_with_logits(logits[b], truth[b]))
    loss = torch.stack(task).mean() + .5*torch.stack(selected).mean() + .5*torch.stack(other).mean()
    return loss, torch.stack(task + selected + other)


def buffered_step(q, optimizer, hidden, ids, labels, target, rules, indices):
    out, _, _ = functional_window(q, hidden, ids, 128, *rules)
    loss, components = buffered_objective(out, labels, target, indices)
    optimizer.zero_grad()
    loss.backward()
    # Finiteness is checked at the block boundary, avoiding a per-step GPU sync.
    norm = torch.nn.utils.clip_grad_norm_([q], 1., error_if_nonfinite=False)
    optimizer.step()
    correct = (out['logits'].argmax(-1) == labels).sum()
    return torch.cat((torch.stack((loss.detach(), norm.detach(), correct)), components.detach()))
