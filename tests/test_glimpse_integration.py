import importlib.metadata
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import bench
from bench import Execution, ImageBatch
from bench.features import extract
from bench.prepare import prepare


def test_glimpse_conditions_match_access_repeat_pooling_and_sampling():
    adapter_type = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "glimpse_ssl_adapter.py"))["GlimpseSSLAdapter"]

    class Encoder(nn.Module):
        def forward(self, crops):
            return crops.mean((-2, -1))

    class Ordered(nn.Module):
        def prefixes(self, *args):
            pass

        def forward(self, features, geometry):
            return (features + geometry).mean(1)

    model = SimpleNamespace(encoder=Encoder(), compositor=Ordered(), geometry_embedder=nn.Linear(5, 3), mean_decoder=nn.Linear(8, 3))
    arguments = dict(model=model, budget=2, observation_seed=3, crop_size=8, patch_size=4, training_budgets=[2], source_revision="test", model_recipe={})
    adapters = {condition: adapter_type(condition=condition, **arguments) for condition in ("M1", "C1", "Mk", "Ck", "Mk-repeat", "Ck-repeat")}
    batch = ImageBatch(torch.arange(2 * 3 * 16 * 16).reshape(2, 3, 16, 16).float() / 1536, torch.ones(2, 1, 16, 16, dtype=torch.bool), torch.eye(3).repeat(2, 1, 1), ("one", "two"))
    execution = Execution(device="cpu", numerics="fp32-v1")
    features = {condition: extract(adapter, batch, "global", execution) for condition, adapter in adapters.items()}
    torch.testing.assert_close(features["M1"]["values"], features["Mk-repeat"]["values"])
    assert adapters["Mk"].observation_boxes(batch) == adapters["Ck"].observation_boxes(batch)
    assert adapters["Mk-repeat"].observation_boxes(batch) == adapters["Ck-repeat"].observation_boxes(batch)
    assert features["Ck-repeat"]["costs"]["encoder_passes_count"].tolist() == [2, 2]
    assert features["Ck-repeat"]["costs"]["observed_area_fraction"].tolist() == [0.25, 0.25]
    assert features["Ck"]["costs"]["observed_area_fraction"].tolist() == [0.5, 0.5]
    assert adapters["C1"].info.recipe["budget_extrapolation"]
    for condition in adapters:
        altered = batch.images.clone()
        for index, boxes in enumerate(adapters[condition].observation_boxes(batch)):
            observed = torch.zeros(16, 16, dtype=torch.bool)
            for left, top, right, bottom in boxes:
                observed[top:bottom, left:right] = True
            altered[index, :, ~observed] = 0.123
        modified = ImageBatch(altered, batch.valid_pixels, batch.image_to_original, batch.sample_ids)
        torch.testing.assert_close(extract(adapters[condition], modified, "global", execution)["values"], features[condition]["values"])
        single = ImageBatch(batch.images[1:], batch.valid_pixels[1:], batch.image_to_original[1:], batch.sample_ids[1:])
        torch.testing.assert_close(extract(adapters[condition], single, "global", execution)["values"], features[condition]["values"][1:])
    spatial = extract(adapters["Ck"], batch, "spatial", execution)
    assert spatial["values"].shape == (2, 3, 4, 4)


@pytest.mark.parametrize("changed_library", ["timm", "transformers"])
def test_glimpse_library_change_rejects_completed_result_reuse(clevr_data, tmp_path, monkeypatch, changed_library):
    root, _ = clevr_data
    prepare(names=["clevr-count"], data_root=root)
    versions = {"timm": "fixture-1", "transformers": None}
    original = importlib.metadata.version

    def package_version(name):
        if name not in versions:
            return original(name)
        if versions[name] is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(importlib.metadata, "version", package_version)
    adapter_type = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "glimpse_ssl_adapter.py"))["GlimpseSSLAdapter"]

    class Encoder(nn.Module):
        def forward(self, crops):
            return crops.mean((-2, -1))

    model = SimpleNamespace(encoder=Encoder(), compositor=nn.Identity(), geometry_embedder=nn.Linear(5, 3), mean_decoder=nn.Linear(8, 3))
    arguments = dict(model=model, condition="M1", budget=2, observation_seed=0, crop_size=8, patch_size=4, training_budgets=[2], source_revision="fixture", model_recipe={})
    execution = Execution(device="cpu", numerics="fp32-v1", workers=0)
    output = tmp_path / "result"
    first = adapter_type(**arguments)
    bench.run(first, names=["clevr-count"], data_root=root, output_dir=output, execution=execution)
    versions[changed_library] = "fixture-2"
    second = adapter_type(**arguments)
    with pytest.raises(ValueError, match="Conflicting run identity"):
        bench.run(second, names=["clevr-count"], data_root=root, output_dir=output, execution=execution)
    assert first.info.recipe["library_versions"] == {"timm": "fixture-1", "transformers": None}
    assert second.info.recipe["library_versions"][changed_library] == "fixture-2"


@pytest.mark.parametrize("padded", [False, True])
def test_grid_glimpse_readouts_preserve_access_geometry_and_costs(padded):
    adapter_type = runpy.run_path(str(Path(__file__).parents[1] / "examples/grid_glimpse_ssl_adapter.py"))["GridGlimpseSSLAdapter"]
    encoded, decoded = [], []

    class Encoder(nn.Module):
        def forward(self, crops):
            encoded.append(crops.clone())
            assert crops.shape[-3:] == (3, 96, 96)
            grid = torch.nn.functional.adaptive_avg_pool2d(crops.flatten(0, 1), 3)
            # Exercise native FP32 readout reduction from BF16 features on CPU.
            return grid.flatten(2).transpose(1, 2).reshape(*crops.shape[:2], 9, 3).bfloat16()

    class Decoder(nn.Module):
        def forward(self, features, coordinates, queries):
            decoded.append((coordinates.clone(), queries.clone()))
            return (features.mean((1, 2))[:, None, None] + queries[:, :, None] + torch.arange(9)[None, None, :, None]).bfloat16()

    model = SimpleNamespace(encoder=Encoder(), decoder=Decoder(), num_context_views=2)
    arguments = dict(model=model, budget=2, observation_seed=3, source_revision="fixture", model_recipe={})
    adapters = {name: adapter_type(condition=name, **arguments) for name in ("M1", "C1", "Mk", "Ck", "Mk-repeat", "Ck-repeat")}
    pixels = torch.arange(2 * 3 * 16 * 16).reshape(2, 3, 16, 16).float() / 1536
    valid = torch.ones(2, 1, 16, 16, dtype=torch.bool)
    if padded:
        valid[:, :, :4] = False
        valid[:, :, 12:] = False
    transform = torch.tensor([[2., 0., -0.5], [0., 3., 0.25], [0., 0., 1.]]).repeat(2, 1, 1)
    batch = ImageBatch(pixels, valid, transform, ("one", "two"))
    execution = Execution(device="cpu", numerics="fp32-v1")
    outputs, inputs = {}, {}
    rng = torch.random.get_rng_state().clone()
    for name, adapter in adapters.items():
        decoded.clear()
        outputs[name] = extract(adapter, batch, "global", execution)
        assert outputs[name]["values"].dtype == torch.float32
        inputs[name] = encoded[-1]
        assert len(decoded) == int(name.startswith("C"))
        assert outputs[name]["costs"]["decoder_passes_count"].tolist() == [int(name.startswith("C"))] * 2
        assert outputs[name]["costs"]["encoder_pixels_count"].tolist() == [adapter.count * 96 ** 2] * 2
        changed = pixels.clone()
        boxes = adapter.observation_boxes(batch)
        for index, sequence in enumerate(boxes):
            observed = torch.zeros(16, 16, dtype=torch.bool)
            for left, top, right, bottom in sequence:
                observed[top:bottom, left:right] = True
            changed[index, :, ~observed] = 0.456
        altered = ImageBatch(changed, valid, transform, batch.sample_ids)
        torch.testing.assert_close(extract(adapter, altered, "global", execution)["values"], outputs[name]["values"], rtol=0, atol=0)
        single = ImageBatch(pixels[1:], valid[1:], transform[1:], batch.sample_ids[1:])
        torch.testing.assert_close(extract(adapter, single, "global", execution)["values"], outputs[name]["values"][1:], rtol=0, atol=0)
        if name.startswith("C"):
            spatial = extract(adapter, batch, "spatial", execution)
            assert spatial["values"].dtype == torch.float32
            torch.testing.assert_close(spatial["values"].mean((2, 3)), outputs[name]["values"])
            torch.testing.assert_close(spatial["grid_to_original"], transform)
            context, query = decoded[-1]
            # Native source coordinates after the sampler subtracts the observed-center mean.
            short_side = 8 if padded else 16
            for index, sequence in enumerate(boxes):
                source_centers = torch.tensor([[(a + c - 16) / short_side, (b + d - 16) / short_side] for a, b, c, d in sequence])
                origin = source_centers.mean(0)
                torch.testing.assert_close(context[index, :, :2], source_centers - origin)
                torch.testing.assert_close(context[index, :, 2], torch.tensor([2 * (c - a) / short_side for a, b, c, d in sequence]))
                torch.testing.assert_close(query[index, 0], torch.cat((-origin, torch.tensor([32 / short_side]))))
            expected_extent = torch.tensor([[-0.5, 1. if padded else 0.25, 1.5, 2.5 if padded else 3.25]]).repeat(2, 1)
            torch.testing.assert_close(spatial["extent_in_original"], expected_extent)
        else:
            with pytest.raises(ValueError, match="unsupported"):
                adapter.encode_spatial(batch)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.equal(inputs["Mk"], inputs["Ck"])
    assert torch.equal(inputs["Mk-repeat"], inputs["Ck-repeat"])
    assert torch.equal(inputs["M1"], inputs["Mk"][:, :1])
    torch.testing.assert_close(outputs["M1"]["values"], outputs["Mk-repeat"]["values"])
    assert adapters["C1"].info.recipe["budget_extrapolation"]
    assert not adapters["Ck"].info.recipe["budget_extrapolation"]
    untrained = adapter_type(**{**arguments, "condition": "Ck", "model_recipe": {"initialization": "untrained", "pretraining": "none", "model_selection": "none"}})
    assert untrained.info.recipe["training_budgets"] == []
    assert untrained.info.recipe["budget_extrapolation"]
    assert untrained.info.recipe["pretraining"] == untrained.info.recipe["model_selection"] == "none"
