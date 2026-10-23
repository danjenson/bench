import importlib

import pytest
import torch
from torchvision.models import resnet18
from torchvision.models.vision_transformer import VisionTransformer

from bench import Execution, ImageBatch
from bench.adapters.torchvision import TorchvisionAdapter
from bench.features import extract, model_identity
from bench.geometry import grid_centers
from bench.storage import write_json


def test_cnn_and_vit_reference_readouts_geometry_and_independence():
    models = [(resnet18(weights=None), "resnet18", 512), (VisionTransformer(image_size=32, patch_size=16, num_layers=1, num_heads=2, hidden_dim=16, mlp_dim=32), "vit_b_16", 16)]
    batch = ImageBatch(torch.ones(1, 3, 64, 64), torch.ones(1, 1, 64, 64, dtype=torch.bool), torch.eye(3)[None], ("opaque",))
    execution = Execution(device="cpu", numerics="fp32-v1")
    for backbone, architecture, width in models:
        adapter = TorchvisionAdapter(backbone, architecture=architecture)
        state = {name: value.clone() for name, value in adapter.state_dict().items()}
        spatial = extract(adapter, batch, "spatial", execution)
        global_features = extract(adapter, batch, "global", execution)
        repeated = extract(adapter, batch, "spatial", execution)
        assert global_features["values"].shape == (1, width)
        torch.testing.assert_close(spatial["values"], repeated["values"])
        centers = grid_centers({key: value[0] for key, value in spatial.items() if key != "costs"})
        expected = 0.5 / 64 if architecture.startswith("resnet") else 8 / 64
        torch.testing.assert_close(centers[0, 0], torch.tensor([expected, expected]))
        for name, value in adapter.state_dict().items():
            assert torch.equal(value, state[name])


def test_reference_library_version_invalidates_unchanged_weights(monkeypatch):
    import bench.adapters.torchvision as adapter_module

    backbone = resnet18(weights=None)
    execution = Execution(device="cpu", numerics="fp32-v1")
    before = model_identity(TorchvisionAdapter(backbone, architecture="resnet18"), execution)
    monkeypatch.setattr(adapter_module, "torchvision_version", "changed-version", raising=False)
    after = model_identity(TorchvisionAdapter(backbone, architecture="resnet18"), execution)
    assert before["state_digest"] == after["state_digest"]
    assert before["digest"] != after["digest"]


def test_optional_dinov3_local_checkpoint_and_register_token_exclusion(tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    from bench.adapters.dinov3 import DinoV3Adapter

    configuration = transformers.DINOv3ViTConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4, patch_size=16, num_register_tokens=2)
    transformers.DINOv3ViTModel(configuration).save_pretrained(tmp_path)
    write_json(tmp_path / "preprocessor_config.json", {"image_mean": [0.5] * 3, "image_std": [0.5] * 3})
    adapter = DinoV3Adapter(tmp_path)
    batch = ImageBatch(torch.ones(2, 3, 64, 64), torch.ones(2, 1, 64, 64, dtype=torch.bool), torch.eye(3).repeat(2, 1, 1), ("opaque-a", "opaque-b"))
    execution = Execution(device="cpu", numerics="fp32-v1")
    before = model_identity(adapter, execution)
    assert extract(adapter, batch, "global", execution)["values"].shape == (2, 32)
    spatial = extract(adapter, batch, "spatial", execution)
    assert spatial["values"].shape == (2, 32, 4, 4)
    assert before["state_digest"] == model_identity(adapter, execution)["state_digest"]
    monkeypatch.setattr(importlib.import_module("transformers"), "__version__", "changed-version")
    after = model_identity(DinoV3Adapter(tmp_path), execution)
    assert before["state_digest"] == after["state_digest"]
    assert before["digest"] != after["digest"]
