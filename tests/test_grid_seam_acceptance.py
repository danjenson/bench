from copy import deepcopy
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from bench import Extraction, SpatialFeatures


@pytest.mark.parametrize("wrong_adapter", [False, True])
def test_native_query_batch_rounding_does_not_relax_matched_adapter_parity(tmp_path, monkeypatch, wrong_adapter):
    examples = Path(__file__).parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    module = runpy.run_path(str(examples / "accept_grid_glimpse_seam.py"))
    # This CPU fixture simulates a bounded BF16 native query-batch difference.
    # The real CUDA check still uses normal hardware validation.
    monkeypatch.setitem(module["verify"].__globals__, "validate_execution", lambda execution: None)

    class Encoder(nn.Module):
        representation_dim = 3

        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(3)

        def forward(self, crops):
            grid = torch.nn.functional.adaptive_avg_pool2d(crops.flatten(0, 1), 3)
            return self.norm(grid.flatten(2).transpose(1, 2)).reshape(*crops.shape[:2], 9, 3)

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.query_token = nn.Parameter(torch.zeros(1, 1, 3))
            self.blocks = nn.ModuleList([nn.Identity()])

        def forward(self, features, coordinates, queries):
            result = features.mean((1, 2))[:, None, None] + queries[:, :, None] + self.query_token
            return result.expand(-1, -1, 9, -1) + (0.01 if queries.shape[1] == 3 else 0)

    class Model(nn.Module):
        num_context_views = 2

        def __init__(self):
            super().__init__()
            self.encoder, self.decoder = Encoder(), Decoder()
            self.target_encoder = deepcopy(self.encoder)

        def forward(self, crops, coordinates):
            features = self.encoder(crops[:, :2])
            self.target_encoder(crops[:, 2:])
            return SimpleNamespace(features=features, scene_prediction=self.decoder(features, coordinates[:, :2], coordinates[:, 2:])[:, -1].float())

    adapter_type = module["GridGlimpseSSLAdapter"]
    if wrong_adapter:
        original_global, original_spatial = adapter_type.encode_global, adapter_type.encode_spatial

        def global_with_error(self, batch):
            value = original_global(self, batch)
            return Extraction(value.features + 0.01, value.costs)

        def spatial_with_error(self, batch):
            value = original_spatial(self, batch)
            spatial = value.features
            return Extraction(SpatialFeatures(spatial.values + 0.01, spatial.grid_to_original, spatial.extent_in_original), value.costs)

        monkeypatch.setattr(adapter_type, "encode_global", global_with_error)
        monkeypatch.setattr(adapter_type, "encode_spatial", spatial_with_error)
    arguments = dict(budget=2, observation_seed=0, source_revision="fixture", model_recipe={})
    if wrong_adapter:
        with pytest.raises(AssertionError, match="not close"):
            module["verify"](Model(), arguments, tmp_path / "wrong", device="cpu", profiles=("bf16-v1",))
    else:
        report = module["verify"](Model(), arguments, tmp_path / "correct", device="cpu", profiles=("bf16-v1",))
        assert report["status"] == "passed"
        profile = report["profiles"]["bf16-v1"]
        assert profile["native_global_and_spatial_parity"]
        assert profile["native_query_batching"]["maximum_absolute_difference"] == pytest.approx(0.01, abs=1e-6)
        assert profile["native_query_batching"]["training_forward_matches_native_three_queries"]
