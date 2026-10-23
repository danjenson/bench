"""Spatial sampling uses actual affine cell centers, never inferred extents."""

import torch
from torch.nn import functional as F


def grid_centers(feature):
    height, width = feature["values"].shape[-2:]
    matrix = feature["grid_to_original"].float()
    y, x = torch.meshgrid((torch.arange(height, device=matrix.device) + 0.5) / height, (torch.arange(width, device=matrix.device) + 0.5) / width, indexing="ij")
    points = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    return (points @ matrix.T)[..., :2]


def valid_centers(feature):
    centers = grid_centers(feature)
    return ((centers >= 0) & (centers < 1)).all(-1)


def sample_original(values, grid_to_original, points):
    matrix = torch.linalg.inv(grid_to_original.float()).to(values.device)
    points = points.to(values.device, torch.float32)
    homogeneous = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    grid = (homogeneous @ matrix.T)[..., :2] * 2 - 1
    return F.grid_sample(values[None].float(), grid[None], mode="bilinear", padding_mode="border", align_corners=False)[0]


def logits_on_canvas(logits, grid_to_original, canvas_to_original, size):
    # Bound the dense interpolation matrices to the coarse grids measured in acceptance.
    if max(logits.shape[-2:]) <= 32 and grid_to_original.device == canvas_to_original.device and torch.equal(grid_to_original, canvas_to_original):
        weights = []
        for source, target in zip(logits.shape[-2:], size):
            positions = ((torch.arange(target, device=logits.device, dtype=torch.float32) + 0.5) * source / target - 0.5).clamp(0, source - 1)
            centers = torch.arange(source, device=logits.device, dtype=torch.float32)
            weights.append((1 - (positions[:, None] - centers[None]).abs()).clamp_min(0))
        with torch.autocast(logits.device.type, enabled=False):
            return weights[0] @ logits.float() @ weights[1].T
    height, width = size
    y, x = torch.meshgrid((torch.arange(height, device=logits.device) + 0.5) / height, (torch.arange(width, device=logits.device) + 0.5) / width, indexing="ij")
    points = torch.stack([x, y, torch.ones_like(x)], dim=-1)
    original = (points @ canvas_to_original.to(logits.device).float().T)[..., :2]
    return sample_original(logits, grid_to_original, original)
