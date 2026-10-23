"""Readouts for GridGlimpseSSL; the caller owns construction and checkpoint loading."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
import math

import torch
from torch.nn import functional as F

from bench import Extraction, ModelInfo, SpatialFeatures, VisionModel
from bench.features import supplied_extent
from bench.storage import file_digest


class GridGlimpseSSLAdapter(VisionModel):
    def __init__(self, model, *, condition, budget, observation_seed, source_revision, model_recipe, sequence_length=4):
        super().__init__()
        if condition not in ("M1", "C1", "Mk", "Ck", "Mk-repeat", "Ck-repeat"):
            raise ValueError(f"Unsupported glimpse condition: {condition}")
        if budget <= 1 or sequence_length < budget:
            raise ValueError("The comparison requires budget > 1 and sequence_length >= budget")
        self.encoder = model.encoder
        self.decoder = model.decoder
        self.condition, self.observation_seed = condition, observation_seed
        self.count = 1 if condition in ("M1", "C1") else budget
        self.sequence_length = sequence_length
        self.composed = condition.startswith("C")
        readouts = {"global": "mean-predicted-scene-grid" if self.composed else "mean-raw-observed-grid"}
        if self.composed:
            readouts["spatial"] = "predicted-scene-grid"
        try:
            timm_version = version("timm")
        except PackageNotFoundError:
            timm_version = None
        training_budgets = [] if model_recipe.get("initialization") == "untrained" else [model.num_context_views]
        self.info = ModelInfo("grid-glimpse-ssl-" + condition, source_revision + ":adapter:" + file_digest(__file__), readouts, {
            "model": model_recipe, "library_versions": {"timm": timm_version},
            "condition": condition, "budget": self.count, "observation_seed": observation_seed,
            "crop_size": 96, "normalization": "owned-by-registered-grid-encoder",
            "pooling": "float32-mean-over-grid-and-observation-or-query-axes",
            "sampling": {"rule": "seeded-permutation-of-nonoverlapping-canvas-cells", "sequence_length": sequence_length, "overlap_limit": 0, "nested_prefixes": True, "interpolation": "bilinear-align_corners-false-antialias", "repeat": condition.endswith("repeat")},
            "geometry": "centers-relative-to-mean-observed-center; units=half-valid-short-side; square-query-covers-supplied-square-canvas",
            "query_extent_extrapolation": "For padded nonsquare source content, the full-canvas query exceeds the training scene extent of two.",
            "spatial_grid": 3 if self.composed else None, "spatial_content": "predicted" if self.composed else "unsupported",
            "training_budgets": training_budgets, "budget_extrapolation": self.count not in training_budgets,
            "pretraining": model_recipe.get("pretraining", "unknown"), "model_selection": model_recipe.get("model_selection", "unknown"),
        })

    def observation_boxes(self, batch):
        height, width = batch.images.shape[-2:]
        if height != width:
            raise ValueError("GridGlimpseSSL requires a square supplied canvas")
        side = math.ceil(math.sqrt(self.sequence_length))
        boxes = []
        for sample_id, valid in zip(batch.sample_ids, batch.valid_pixels):
            ys, xs = valid[0].nonzero(as_tuple=True)
            if not len(xs):
                raise ValueError("An ImageBatch entry has no valid image content")
            x0, y0 = int(xs.min()), int(ys.min())
            size = min(int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1) // side
            if size < 1:
                raise ValueError("Valid image area is too small for the declared observation sequence")
            seed = int.from_bytes(hashlib.sha256(f"{self.observation_seed}:{sample_id}".encode()).digest()[:8], "big")
            order = torch.randperm(side * side, generator=torch.Generator().manual_seed(seed)).tolist()[:self.sequence_length]
            sequence = [(x0 + cell % side * size, y0 + cell // side * size, x0 + (cell % side + 1) * size, y0 + (cell // side + 1) * size) for cell in order]
            boxes.append([sequence[0]] * self.count if self.condition.endswith("repeat") else sequence[:self.count])
        return boxes

    def observations(self, batch):
        crops, coordinates, queries, coverage = [], [], [], []
        height, width = batch.images.shape[-2:]
        for image, valid, boxes in zip(batch.images, batch.valid_pixels, self.observation_boxes(batch)):
            ys, xs = valid[0].nonzero(as_tuple=True)
            short_side = min(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
            centers = image.new_tensor([[(left + right) / 2, (top + bottom) / 2] for left, top, right, bottom in boxes])
            origin = centers.mean(0)
            extents = image.new_tensor([[2 * (right - left) / short_side] for left, top, right, bottom in boxes])
            coordinates.append(torch.cat((2 * (centers - origin) / short_side, extents), dim=-1))
            query_center = 2 * (image.new_tensor([width / 2, height / 2]) - origin) / short_side
            queries.append(torch.cat((query_center, image.new_tensor([2 * width / short_side])))[None])
            observed = torch.zeros_like(valid[0])
            sample_crops = []
            for left, top, right, bottom in boxes:
                sample_crops.append(F.interpolate(image[None, :, top:bottom, left:right], size=(96, 96), mode="bilinear", align_corners=False, antialias=True)[0])
                observed[top:bottom, left:right] = True
            crops.append(torch.stack(sample_crops))
            coverage.append(((observed & valid[0]).sum() / valid.sum()).float())
        return torch.stack(crops), torch.stack(coordinates), torch.stack(queries), torch.stack(coverage)

    def costs(self, batch, coverage):
        count = len(batch.sample_ids)
        return {"glimpses_count": batch.images.new_full((count,), self.count), "encoder_pixels_count": batch.images.new_full((count,), self.count * 96 ** 2), "encoder_passes_count": batch.images.new_full((count,), self.count), "decoder_passes_count": batch.images.new_full((count,), int(self.composed)), "observed_area_fraction": coverage}

    def encode_global(self, batch):
        crops, coordinates, queries, coverage = self.observations(batch)
        features = self.encoder(crops)
        if self.composed:
            features = self.decoder(features, coordinates, queries)
        return Extraction(features.float().mean((1, 2)), self.costs(batch, coverage))

    def encode_spatial(self, batch):
        if not self.composed:
            raise ValueError(f"Spatial readout is unsupported for condition {self.condition}")
        crops, coordinates, queries, coverage = self.observations(batch)
        tokens = self.decoder(self.encoder(crops), coordinates, queries)[:, 0].float()
        grid = tokens.transpose(1, 2).reshape(len(tokens), -1, 3, 3)
        return Extraction(SpatialFeatures(grid, batch.image_to_original, supplied_extent(batch)), self.costs(batch, coverage))
