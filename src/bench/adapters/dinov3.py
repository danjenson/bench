"""Local Hugging Face DINOv3 ViT checkpoints; no network or training imports."""

import torch

from ..api import Extraction, ModelInfo, SpatialFeatures, VisionModel
from ..features import supplied_extent
from ..storage import file_digest, read_json, resolve_path


class DinoV3Adapter(VisionModel):
    def __init__(self, checkpoint, provenance=None):
        super().__init__()
        try:
            from transformers import DINOv3ViTModel, __version__ as transformers_version
        except ImportError as error:
            raise ImportError("DINOv3 requires the optional adapter dependencies: (uv sync --extra dinov3)") from error
        path = resolve_path(checkpoint)
        self.backbone = DINOv3ViTModel.from_pretrained(str(path), local_files_only=True)
        preprocessing = read_json(path / "preprocessor_config.json")
        self.register_buffer("pixel_mean", torch.tensor(preprocessing["image_mean"])[None, :, None, None])
        self.register_buffer("pixel_std", torch.tensor(preprocessing["image_std"])[None, :, None, None])
        self.info = ModelInfo("dinov3", "bench-dinov3-" + file_digest(__file__), {"global": "final-cls", "spatial": "final-patch-tokens"}, {"checkpoint": str(path), "normalization": {"mean": preprocessing["image_mean"], "std": preprocessing["image_std"]}, "model_config": self.backbone.config.to_dict(), "pretraining": "unknown", "model_selection": "unknown", "spatial_extent": "entire-supplied-content", **(provenance or {}), "library_versions": {"transformers": transformers_version}})

    def _tokens(self, batch):
        patch = self.backbone.config.patch_size
        if any(size % patch for size in batch.images.shape[-2:]):
            raise ValueError(f"DINOv3 canvas dimensions must be divisible by patch size {patch}")
        return self.backbone(pixel_values=(batch.images - self.pixel_mean) / self.pixel_std).last_hidden_state

    def _costs(self, batch):
        size = len(batch.sample_ids)
        return {"encoder_passes_count": batch.images.new_ones(size), "encoder_pixels_count": batch.images.new_full((size,), batch.images.shape[-2] * batch.images.shape[-1]), "observed_area_fraction": batch.images.new_ones(size)}

    def encode_global(self, batch):
        return Extraction(self._tokens(batch)[:, 0], self._costs(batch))

    def encode_spatial(self, batch):
        tokens = self._tokens(batch)[:, 1 + self.backbone.config.num_register_tokens:]
        patch = self.backbone.config.patch_size
        grid = tokens.transpose(1, 2).reshape(len(tokens), -1, batch.images.shape[-2] // patch, batch.images.shape[-1] // patch)
        return Extraction(SpatialFeatures(grid, batch.image_to_original, supplied_extent(batch)), self._costs(batch))
