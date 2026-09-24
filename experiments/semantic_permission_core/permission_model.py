"""Editable semantic-permission assessor and exact expected proper loss."""

import torch
from torch import nn


def balanced_query_basis(count, dtype=torch.float32):
    if count < 2 or count % 2:
        raise ValueError("class count must be positive and even")
    half = count//2
    base = torch.zeros(count, dtype=dtype)
    base[:half] = 1
    masks = [base]
    for label in range(half, count):
        mask = base.clone()
        mask[0], mask[label] = 0, 1
        masks.append(mask)
    for label in range(1, half):
        mask = base.clone()
        mask[label], mask[half] = 0, 1
        masks.append(mask)
    return torch.stack(masks)


def permission_probabilities(probabilities, masks):
    count = probabilities.shape[-1]
    if count < 2 or count % 2 or masks.ndim != 2 or masks.shape[1] != count:
        raise ValueError("incompatible balanced permission dimensions")
    if not bool(((masks == 0) | (masks == 1)).all()) or not bool((masks.sum(-1) == count//2).all()):
        raise ValueError("each permission set must contain exactly half the classes")
    return probabilities @ masks.to(probabilities).T


def recover_class_probabilities(reports):
    count = reports.shape[-1]
    if count < 2 or count % 2:
        raise ValueError("class count must be positive and even")
    half = count//2
    differences = reports[..., 1:half+1]-reports[..., :1]
    first_offsets = differences[..., :1]-(reports[..., half+1:]-reports[..., :1])
    reference = (1-differences.sum(-1, keepdim=True)-first_offsets.sum(-1, keepdim=True))/count
    return torch.cat([reference, reference+first_offsets, reference+differences], dim=-1)


def expected_permission_brier(probabilities, targets):
    count = probabilities.shape[-1]
    if count < 2 or count % 2:
        raise ValueError("class count must be positive and even")
    one_hot = torch.nn.functional.one_hot(targets, num_classes=count).to(probabilities)
    return count/(4*(count-1))*((probabilities-one_hot)**2).sum(-1)


class SemanticPermissionModel(nn.Module):
    """Use a learned category distribution to assess independent permissions."""

    def __init__(self, parent, taxonomy):
        super().__init__()
        self.parent = parent
        self.register_buffer("taxonomy", taxonomy.to(torch.long))
        self.coarse_count = int(taxonomy.max())+1

    def distributions(self, images):
        fine = self.parent(images).softmax(-1)
        coarse = fine.new_zeros((len(fine), self.coarse_count))
        coarse.scatter_add_(1, self.taxonomy[None].expand(len(fine), -1), fine)
        return fine, coarse

    def forward(self, images, fine_masks, coarse_masks):
        fine, coarse = self.distributions(images)
        return permission_probabilities(fine, fine_masks), permission_probabilities(coarse, coarse_masks)

    def expected_loss(self, images, fine_targets):
        fine, coarse = self.distributions(images)
        coarse_targets = self.taxonomy[fine_targets]
        return (expected_permission_brier(fine, fine_targets)+
                expected_permission_brier(coarse, coarse_targets))/2
