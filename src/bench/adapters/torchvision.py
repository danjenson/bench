"""Conventional CNN and ViT reference readouts with explicit geometry."""

import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import __version__ as torchvision_version

from ..api import Extraction, ModelInfo, SpatialFeatures, VisionModel
from ..features import supplied_extent
from ..storage import file_digest, resolve_path


class TorchvisionAdapter(VisionModel):
    def __init__(self, backbone, *, architecture, provenance=None):
        super().__init__()
        if architecture not in ("resnet18", "resnet50", "vit_b_16", "vit_l_16"):
            raise ValueError("Reference adapter supports resnet18, resnet50, vit_b_16, and vit_l_16")
        self.backbone, self.architecture = backbone, architecture
        self.register_buffer("pixel_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("pixel_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        cnn = architecture.startswith("resnet")
        if cnn:
            self.backbone.fc = nn.Identity()
        else:
            self.backbone.heads = nn.Identity()
        self.info = ModelInfo(architecture, "bench-torchvision-" + file_digest(__file__), {"global": "layer4-mean" if cnn else "final-cls", "spatial": "layer4-grid" if cnn else "final-patch-tokens"}, {"architecture": architecture, "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}, "position_interpolation": None if cnn else "bicubic-align_corners-false", "spatial_extent": "entire-supplied-content", "pretraining": "unknown", "model_selection": "unknown", **(provenance or {}), "library_versions": {"torchvision": torchvision_version}})

    @classmethod
    def from_checkpoint(cls, checkpoint, architecture="resnet18", provenance=None):
        from torchvision.models import get_model

        path = resolve_path(checkpoint)
        backbone = get_model(architecture, weights=None)
        backbone.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"))
        return cls(backbone, architecture=architecture, provenance={"checkpoint": str(path), **(provenance or {})})

    def _readouts(self, batch):
        pixels = (batch.images - self.pixel_mean) / self.pixel_std
        model = self.backbone
        if self.architecture.startswith("resnet"):
            grid = model.maxpool(model.relu(model.bn1(model.conv1(pixels))))
            grid = model.layer4(model.layer3(model.layer2(model.layer1(grid))))
            height, width = pixels.shape[-2:]
            grid_height, grid_width = grid.shape[-2:]
            # ResNet's padded odd kernels have their first center at input pixel 0.5.
            grid_to_canvas = batch.image_to_original.new_tensor([[32 * grid_width / width, 0, (0.5 - 16) / width], [0, 32 * grid_height / height, (0.5 - 16) / height], [0, 0, 1]])
            with torch.autocast(pixels.device.type, enabled=False):
                affine = batch.image_to_original.float() @ grid_to_canvas.float()
            return grid.mean((2, 3)), grid, affine
        height, width = pixels.shape[-2:]
        patch = model.patch_size
        if height % patch or width % patch:
            raise ValueError(f"ViT canvas dimensions must be divisible by patch size {patch}")
        grid = model.conv_proj(pixels)
        tokens = grid.flatten(2).transpose(1, 2)
        tokens = torch.cat([model.class_token.expand(len(tokens), -1, -1), tokens], dim=1)
        positions = model.encoder.pos_embedding
        original_side = math.isqrt(positions.shape[1] - 1)
        spatial = positions[:, 1:].transpose(1, 2).reshape(1, -1, original_side, original_side)
        spatial = F.interpolate(spatial, size=grid.shape[-2:], mode="bicubic", align_corners=False).flatten(2).transpose(1, 2)
        positions = torch.cat([positions[:, :1], spatial], dim=1)
        tokens = model.encoder.ln(model.encoder.layers(model.encoder.dropout(tokens + positions)))
        grid = tokens[:, 1:].transpose(1, 2).reshape(len(tokens), -1, height // patch, width // patch)
        return tokens[:, 0], grid, batch.image_to_original

    def _costs(self, batch):
        size = len(batch.sample_ids)
        return {"encoder_passes_count": batch.images.new_ones(size), "encoder_pixels_count": batch.images.new_full((size,), batch.images.shape[-2] * batch.images.shape[-1]), "observed_area_fraction": batch.images.new_ones(size)}

    def encode_global(self, batch):
        global_values, _, _ = self._readouts(batch)
        return Extraction(global_values, self._costs(batch))

    def encode_spatial(self, batch):
        _, grid, affine = self._readouts(batch)
        return Extraction(SpatialFeatures(grid, affine, supplied_extent(batch)), self._costs(batch))
