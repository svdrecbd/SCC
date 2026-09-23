"""Connect the editable depth parent to joint, coordinate-consistent assessment."""

import torch
from torch import nn
from joint_scene_assessment import JointSceneDistribution
from joint_scene_mixture import JointSceneMixtureDistribution


def remap_program_rays(program, original_to_unique):
    remapped = []
    for instruction in program:
        operation = instruction[0]
        updated = list(instruction)
        positions = (1, 2) if operation == "less" else (1,) if operation == "above" else ()
        for position in positions:
            original = instruction[position]
            if not isinstance(original, int) or not 0 <= original < len(original_to_unique):
                raise ValueError("Ray reference outside the supplied coordinate list.")
            updated[position] = int(original_to_unique[original])
        remapped.append(updated)
    return remapped


class SpatialProgramAssessmentModel(nn.Module):
    """Prepare one image's joint ray distribution; concurrent calls are unsupported."""

    def __init__(self, parent, seed=33237, mixture_components=0):
        super().__init__()
        self.parent = parent
        self.mixture_components = mixture_components
        if mixture_components:
            self.scene_distribution = JointSceneMixtureDistribution(parent.head.conv3.in_channels, component_count=mixture_components, seed=seed)
        else:
            self.scene_distribution = JointSceneDistribution(parent.head.conv3.in_channels, seed=seed)

    def forward(self, pixel_values, ray_coordinates, output_size=(480, 640)):
        if pixel_values.shape[0] != 1:
            raise ValueError("Joint query preparation currently supports one image at a time.")
        coordinates = torch.as_tensor(ray_coordinates, device=pixel_values.device)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2 or len(coordinates) == 0 or coordinates.dtype not in (torch.int32, torch.int64):
            raise ValueError("Expected nonempty integer row-column coordinate pairs.")
        if not ((coordinates >= 0).all() and (coordinates[:, 0] < output_size[0]).all() and (coordinates[:, 1] < output_size[1]).all()):
            raise ValueError("Ray coordinate outside the output image.")
        unique_coordinates, original_to_unique = torch.unique(coordinates.long(), dim=0, return_inverse=True)
        activations = []

        def retain_activation(module, arguments):
            activations.append(arguments[0])

        handle = self.parent.head.conv3.register_forward_pre_hook(retain_activation)
        try:
            parent_output = self.parent(pixel_values=pixel_values).predicted_depth
        finally:
            handle.remove()
        if len(activations) != 1:
            raise RuntimeError("Expected one final parent decoder activation.")
        depth_map = torch.nn.functional.interpolate(parent_output[:, None], size=output_size, mode="bilinear", align_corners=False)[0, 0]
        feature_map = torch.nn.functional.interpolate(activations[0], size=output_size, mode="bilinear", align_corners=False)[0]
        rows, columns = unique_coordinates.T
        parent_depth = depth_map[rows, columns]
        features = feature_map[:, rows, columns].T
        if self.mixture_components:
            distribution = self.scene_distribution(features, parent_depth, feature_map.mean(dim=(1, 2)))
        else:
            distribution = self.scene_distribution(features, parent_depth)
        return {"distribution": distribution, "parent_depth_metres": parent_depth,
                "unique_coordinates": unique_coordinates, "original_to_unique": original_to_unique}
