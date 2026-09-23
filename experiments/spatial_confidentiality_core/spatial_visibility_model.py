"""Integrate a fully editable depth estimator with a native visibility field."""
import torch
from torch import nn
from visibility_representation import VisibilityRepresentation


class SpatialVisibilityModel(nn.Module):
    """Expose depth probabilities and their mean from a metric depth parent.

    The pinned parent exposes head.conv3 after its final decoder activation.
    A temporary hook retains that activation with its gradient graph. Concurrent
    calls on the same module are unsupported; the hook is removed on failure.
    """

    def __init__(self, parent, depth_range_metres=10.0, support_count=65):
        super().__init__()
        self.parent = parent
        self.depth_range_metres = depth_range_metres
        feature_count = parent.head.conv3.in_channels
        self.visibility = VisibilityRepresentation(feature_count, support_count)

    def forward(self, pixel_values, output_size=(480, 640)):
        activations = []

        def retain_activation(module, arguments):
            activations.append(arguments[0])

        handle = self.parent.head.conv3.register_forward_pre_hook(retain_activation)
        try:
            parent_output = self.parent(pixel_values=pixel_values).predicted_depth
        finally:
            handle.remove()
        if len(activations) != 1:
            raise RuntimeError("Expected exactly one final decoder activation.")
        parent_depth = torch.nn.functional.interpolate(
            parent_output.unsqueeze(1), size=output_size, mode='bilinear', align_corners=False,
        ).squeeze(1)
        features = torch.nn.functional.interpolate(
            activations[0], size=output_size, mode='bilinear', align_corners=False,
        )
        parent_means = parent_depth.clamp(.1, self.depth_range_metres)/self.depth_range_metres
        probabilities, means = self.visibility(features, parent_means)
        return {'probabilities': probabilities, 'depth_metres': means*self.depth_range_metres,
                'parent_depth_metres': parent_depth}
