"""Model integration for glimpse_ssl.models.GlimpseSSL.

The caller constructs/loads its model and passes its resolved model recipe.
This module imports no training loop, tracker, checkpoint registry, or sampler.
"""

import hashlib
from importlib.metadata import PackageNotFoundError, version
import math

import torch
from torch.nn import functional as F

from bench import Extraction, ModelInfo, SpatialFeatures, VisionModel
from bench.features import supplied_extent
from bench.storage import file_digest


class GlimpseSSLAdapter(VisionModel):
    def __init__(self, model, *, condition, budget, observation_seed, crop_size, patch_size, training_budgets, source_revision, model_recipe, sequence_length=4, spatial_grid=4):
        super().__init__()
        if condition not in ("M1", "C1", "Mk", "Ck", "Mk-repeat", "Ck-repeat"):
            raise ValueError(f"Unsupported glimpse condition: {condition}")
        if budget <= 1 or sequence_length < budget:
            raise ValueError("The comparison requires budget > 1 and sequence_length >= budget")
        if crop_size % patch_size:
            raise ValueError("crop_size must be divisible by patch_size")
        self.encoder = model.encoder
        self.compositor = model.compositor
        self.geometry_embedder = model.geometry_embedder
        self.mean_decoder = model.mean_decoder
        self.condition, self.observation_seed = condition, observation_seed
        self.count = 1 if condition in ("M1", "C1") else budget
        self.crop_size, self.patch_size = crop_size, patch_size
        self.sequence_length, self.spatial_grid = sequence_length, spatial_grid
        self.composed = condition.startswith("C")
        readouts = {"global": "composed-percept" if self.composed else "mean-raw-crop-cls"}
        if self.composed:
            readouts["spatial"] = "predicted-query-grid"
        library_versions = {}
        for package in ("timm", "transformers"):
            try:
                library_versions[package] = version(package)
            except PackageNotFoundError:
                library_versions[package] = None
        self.info = ModelInfo("glimpse-ssl-" + condition, source_revision + ":adapter:" + file_digest(__file__), readouts, {"model": model_recipe, "library_versions": library_versions, "condition": condition, "budget": self.count, "observation_seed": observation_seed, "crop_size": crop_size, "patch_size": patch_size, "normalization": "owned-by-registered-glimpse-encoder", "sampling": {"rule": "seeded-permutation-of-nonoverlapping-canvas-cells", "sequence_length": sequence_length, "overlap_limit": 0, "nested_prefixes": True, "interpolation": "bilinear-align_corners-false-antialias", "repeat": condition.endswith("repeat")}, "spatial_grid": spatial_grid if self.composed else None, "spatial_content": "predicted" if self.composed else "unsupported", "training_budgets": training_budgets, "budget_extrapolation": self.count not in training_budgets, "pretraining": model_recipe.get("pretraining", "unknown"), "model_selection": model_recipe.get("model_selection", "unknown")})

    def observation_boxes(self, batch):
        """Integer canvas boxes: auditable, independent of pixels and batching."""
        height, width = batch.images.shape[-2:]
        side = math.ceil(math.sqrt(self.sequence_length))
        if min(height, width) < side:
            raise ValueError("Supplied canvas is too small for the declared observation sequence")
        boxes = []
        for sample_id, valid in zip(batch.sample_ids, batch.valid_pixels):
            seed = int.from_bytes(hashlib.sha256(f"{self.observation_seed}:{sample_id}".encode()).digest()[:8], "big")
            generator = torch.Generator().manual_seed(seed)
            order = torch.randperm(side * side, generator=generator).tolist()[:self.sequence_length]
            # Square cells preserve the model's scalar width/height geometry convention.
            ys, xs = valid[0].nonzero(as_tuple=True)
            x0, y0 = int(xs.min()), int(ys.min())
            size = min(int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1) // side
            if size < 1:
                raise ValueError("Valid image area is too small for the declared observation sequence")
            sequence = [(x0 + cell % side * size, y0 + cell // side * size, x0 + (cell % side + 1) * size, y0 + (cell // side + 1) * size) for cell in order]
            chosen = [sequence[0]] * self.count if self.condition.endswith("repeat") else sequence[:self.count]
            boxes.append(chosen)
        return boxes

    def observations(self, batch):
        boxes = self.observation_boxes(batch)
        all_crops, all_geometry, coverage = [], [], []
        height, width = batch.images.shape[-2:]
        for image, valid, sequence in zip(batch.images, batch.valid_pixels, boxes):
            crops, coordinates = [], []
            observed = torch.zeros_like(valid[0])
            for left, top, right, bottom in sequence:
                # Slice before resize; interpolation cannot inspect outside the box.
                crop = image[:, top:bottom, left:right]
                crops.append(F.interpolate(crop[None], size=(self.crop_size, self.crop_size), mode="bilinear", align_corners=False, antialias=True)[0])
                coordinates.append([(left + right) / (2 * width), (top + bottom) / (2 * height), math.log((right - left) / width), math.log(self.crop_size / self.patch_size), math.log(self.crop_size / self.patch_size)])
                observed[top:bottom, left:right] = True
            all_crops.append(torch.stack(crops))
            all_geometry.append(image.new_tensor(coordinates))
            coverage.append(((observed & valid[0]).sum() / valid.sum()).float())
        return torch.stack(all_crops), torch.stack(all_geometry), torch.stack(coverage)

    def compose(self, features, geometry):
        embedded = self.geometry_embedder(geometry)
        if hasattr(self.compositor, "prefixes"):
            return self.compositor(features, embedded)
        return self.compositor(features, embedded, geometry[..., :2])

    def costs(self, batch, coverage, *, spatial):
        count = len(batch.sample_ids)
        return {"glimpses_count": batch.images.new_full((count,), self.count), "encoder_pixels_count": batch.images.new_full((count,), self.count * self.crop_size ** 2), "encoder_passes_count": batch.images.new_full((count,), self.count), "decoder_passes_count": batch.images.new_full((count,), int(spatial)), "compositor_passes_count": batch.images.new_full((count,), int(self.composed)), "observed_area_fraction": coverage}

    def encode_global(self, batch):
        crops, geometry, coverage = self.observations(batch)
        features = self.encoder(crops)
        values = self.compose(features, geometry) if self.composed else features.mean(1)
        return Extraction(values, self.costs(batch, coverage, spatial=False))

    def encode_spatial(self, batch):
        if not self.composed:
            raise ValueError(f"Spatial readout is unsupported for condition {self.condition}")
        crops, geometry, coverage = self.observations(batch)
        percept = self.compose(self.encoder(crops), geometry)
        side = self.spatial_grid
        axis = (torch.arange(side, device=percept.device) + 0.5) / side
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        queries = torch.stack([x, y, torch.full_like(x, math.log(1 / side)), torch.full_like(x, math.log(self.crop_size / self.patch_size)), torch.full_like(x, math.log(self.crop_size / self.patch_size))], dim=-1).reshape(1, side * side, 5).expand(len(percept), -1, -1)
        values = self.mean_decoder(torch.cat([percept[:, None].expand(-1, side * side, -1), queries], dim=-1))
        grid = values.transpose(1, 2).reshape(len(percept), -1, side, side)
        return Extraction(SpatialFeatures(grid, batch.image_to_original, supplied_extent(batch)), self.costs(batch, coverage, spatial=True))
