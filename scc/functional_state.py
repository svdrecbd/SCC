"""Functional parameter edits with explicit, unedited model buffers."""
from torch.func import functional_call


def call_parameters(model, parameters, args, kwargs=None, *, strict=True):
    """Buffers belong to model state but are outside these parameter-only edits.

    Forward functions used here must not mutate buffers. Models with running
    statistics need a separate state-transition contract, not this helper.
    """
    return functional_call(model, (parameters, dict(model.named_buffers())), args,
                           kwargs, strict=strict)
