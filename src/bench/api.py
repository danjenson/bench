"""The model extension seam; dataset and probe implementations are internal."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from torch import Tensor, nn


@dataclass(frozen=True)
class ModelInfo:
    name: str
    revision: str
    readouts: dict[Literal["global", "spatial"], str]
    recipe: dict[str, object]


@dataclass(frozen=True)
class ImageBatch:
    images: Tensor
    valid_pixels: Tensor
    image_to_original: Tensor
    sample_ids: tuple[str, ...]


@dataclass(frozen=True)
class SpatialFeatures:
    values: Tensor
    grid_to_original: Tensor
    extent_in_original: Tensor


@dataclass(frozen=True)
class Extraction:
    features: Tensor | SpatialFeatures
    costs: dict[str, Tensor] | None = None


class VisionModel(nn.Module):
    info: ModelInfo

    def encode_global(self, batch: ImageBatch) -> Extraction:
        raise NotImplementedError

    def encode_spatial(self, batch: ImageBatch) -> Extraction:
        raise NotImplementedError


@dataclass(frozen=True)
class Execution:
    device: str = "cuda"
    batch_size: int = 64
    workers: int = 8
    work_dir: Path | None = None
    numerics: Literal["bf16-v1", "fp32-v1"] = "bf16-v1"


@dataclass(frozen=True)
class RunResult:
    schema_version: str
    status: Literal["running", "complete", "failed"]
    metrics: dict[str, dict[str, float]]
    output_dir: Path
