import importlib
import random
from copy import deepcopy

import numpy as np
import pytest
import torch
from PIL import Image
from torch.nn import functional as F

import bench
from bench import ade20k
from bench.features import supplied_extent
from bench.images import resize_short_size, window_boxes
from bench.prepare import prepare, split_samples
from bench.protocols import recipes, resolve_names
from bench.storage import digest, read_json


def test_only_ade_recipe_matches_dinov3_and_uses_official_splits():
    task = resolve_names(["ade20k"])[0]
    assert task["id"] == "ade20k-dinov3-linear-v1"
    assert task["canvas"] == 512 and task["evaluation_stride"] == 341
    assert task["standardization"] is None and task["numerics"] == ["fp32-v1"]
    recipe = task["probe"]
    assert recipe["updates"] == 40000 and recipe["batch_size"] == 16
    assert recipe["evaluation_interval"] == 5000 and recipe["warmup_updates"] == 1500
    assert recipe["learning_rates"] == [.0001, .0003, .001]
    assert recipe["weight_decays"] == [.0001, .001] and recipe["seeds"] == [100]
    assert "epochs" not in recipe and "budget_validation" not in recipe
    source = {"train": [{"id": str(i)} for i in range(10)], "final": [{"id": "val"}]}
    splits = split_samples(source, task)
    assert splits["train"] == source["train"]
    assert splits["development"] == splits["final"] == source["final"]
    with pytest.raises(ValueError, match="Unknown benchmark"):
        resolve_names(["ade20k-linear-v1"])


@pytest.mark.parametrize("size", [(640, 480), (480, 640), (512, 512), (1200, 200)])
def test_sliding_windows_cover_resized_image_without_padding(size):
    width, height = resize_short_size(size, 512)
    assert min(width, height) == 512
    counts = np.zeros((height, width), dtype=np.int32)
    boxes = list(window_boxes(height, width, 512, 341))
    for top, bottom, left, right in boxes:
        assert bottom - top == right - left == 512
        counts[top:bottom, left:right] += 1
    assert counts.min() == 1
    assert len(boxes) == 1 or counts.max() > 1
    if size == (640, 480):
        assert (width, height) == (683, 512)
        assert boxes == [(0, 512, 0, 512), (0, 512, 171, 683)]


def test_training_views_repeat_by_draw_preserve_geometry_and_ignore_padding():
    task = deepcopy(recipes()["ade20k"])
    task["canvas"] = 16
    task["augmentation"].update(scale_range=[.5, .5], flip_probability=1.)
    image = Image.fromarray(np.arange(16 * 12 * 3, dtype=np.uint8).reshape(12, 16, 3))
    mask = np.indices((12, 16)).sum(0) % 2
    views = [ade20k.training_view(image, mask, task, np.random.default_rng(21)) for _ in range(2)]
    for first, second in zip(*views):
        torch.testing.assert_close(first, second)
    pixels, valid, matrix, target = views[0]
    assert target.shape == (16, 16) and valid.sum() == 8 * 11
    assert (target[~valid[0]] == 255).all()
    torch.testing.assert_close(pixels[:, ~valid[0]], torch.tensor([.485, .456, .406])[:, None].expand(3, 256 - 88))
    corners = torch.tensor([[0., 0., 1.], [11 / 16, 8 / 16, 1.]]) @ matrix.T
    torch.testing.assert_close(corners, torch.tensor([[1., 0., 1.], [0., 1., 1.]]))
    changed = ade20k.training_view(image, mask, task, np.random.default_rng(22))
    assert not torch.equal(pixels, changed[0])


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration job")])])
def test_dense_microbatches_match_full_batch_bn_gradients_and_loss(device):
    torch.manual_seed(8)
    head = ade20k.LinearHead(4, 3, 0.1, device)
    reference = deepcopy(head)
    values = torch.randn(3, 4, 4, 4, device=device)
    labels = torch.randint(0, 3, (3, 8, 8), device=device)
    labels[0, :, :5] = 255
    geometry = torch.eye(3, device=device).expand(3, -1, -1)
    rng = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
    actual = ade20k.backward_pixels(head(values), geometry, labels, geometry)
    if device == "cuda":
        torch.cuda.set_rng_state(rng)
    else:
        torch.set_rng_state(rng)
    logits = F.interpolate(reference(values), size=(8, 8), mode="bilinear", align_corners=False)
    expected = F.cross_entropy(logits, labels, ignore_index=255, reduction="none").mean()
    expected.backward()
    assert actual == pytest.approx(float(expected.detach()))
    for a, b in zip(head.parameters(), reference.parameters()):
        torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(head.norm.running_mean, reference.norm.running_mean)
    torch.testing.assert_close(head.norm.running_var, reference.norm.running_var)


def test_sliding_logits_are_averaged_before_original_resolution_argmax():
    task = {**recipes()["ade20k"], "canvas": 4, "evaluation_stride": 2, "classes": 2}

    class WindowEncoder:
        def __call__(self, images, valid, transform, ids):
            # Distinct constant predictions make overlap averaging observable.
            value = 1 if float(transform[0, 0, 2]) == 0 else 3
            return {"values": torch.tensor([value, 2.]).reshape(1, 2, 1, 1).expand(1, 2, 2, 2), "grid_to_original": transform}

    actual = ade20k.predict(WindowEncoder(), torch.nn.Identity(), torch.zeros(3, 4, 6), (7, 9), "image", task)
    expected = torch.tensor([1., 1., 2., 2., 3., 3.])[None, None].expand(1, 4, 6)
    expected = torch.cat([expected, torch.full_like(expected, 2.)])
    expected = F.interpolate(expected[None], size=(7, 9), mode="bilinear", align_corners=False)[0]
    torch.testing.assert_close(actual, expected)


def test_reference_schedule_has_1500_step_warmup_and_cosine_tail():
    recipe = recipes()["ade20k"]["probe"]
    # Values independently checked against the pinned upstream scheduler.
    expected = {0: 1e-6, 1: 1e-6, 2: 0.0006676493083500112,
                1500: 0.96188331730835, 1501: 0.9965296112632268,
                20000: 0.5, 39999: 1.542125704911967e-9}
    for update, value in expected.items():
        assert ade20k.learning_rate_scale(update, recipe) == pytest.approx(value, rel=1e-10, abs=1e-12)


@pytest.fixture
def ade_data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    source = root / "ADEChallengeData2016"
    for split, count in [("training", 6), ("validation", 2)]:
        for kind in ("images", "annotations"):
            (source / kind / split).mkdir(parents=True)
        for index in range(count):
            width, height = (20, 12) if index % 2 == 0 else (17, 23)
            pixels = np.zeros((height, width, 3), dtype=np.uint8)
            pixels[:, :width // 2, 0], pixels[:, width // 2:, 1] = 180, 180
            mask = np.where(np.indices((height, width))[1] < width // 2, 1, 150).astype(np.uint8)
            mask[0, :] = 0
            Image.fromarray(pixels).save(source / "images" / split / f"{index}.jpg")
            Image.fromarray(mask).save(source / "annotations" / split / f"{index}.png")
    (source / "objectInfo150.txt").write_text("Idx\tName\n" + "\n".join(f"{i}\tclass-{i}" for i in range(1, 151)))
    tasks = recipes()
    task = tasks["ade20k"]
    task.update(canvas=16, evaluation_stride=11)
    task["probe"].update(updates=3, batch_size=2, warmup_updates=1, evaluation_interval=2, learning_rates=[.01], weight_decays=[.0001, .001])
    task["digest"] = digest({k: v for k, v in task.items() if k != "digest"})
    from bench.acquisition import source_recipes
    sources = source_recipes()
    sources["ade20k"]["counts"] = {"train": 6, "final": 2}
    monkeypatch.setattr(importlib.import_module("bench.protocols"), "recipes", lambda: deepcopy(tasks))
    monkeypatch.setattr(importlib.import_module("bench.prepare"), "source_recipes", lambda: sources)
    assert prepare(names=["ade20k"], data_root=root)["ready"]
    return root, task


class SpatialAdapter(bench.VisionModel):
    def __init__(self):
        super().__init__()
        self.norm = torch.nn.BatchNorm2d(3)
        self.info = bench.ModelInfo("fixture", "1", {"spatial": "normalized-rgb"}, {})
        self.calls = 0

    def encode_spatial(self, batch):
        assert not self.training and not self.norm.training
        self.calls += 1
        values = F.avg_pool2d(self.norm(batch.images), 4)
        return bench.Extraction(bench.SpatialFeatures(values, batch.image_to_original, supplied_extent(batch)),
                                {"encoder_passes_count": torch.ones(len(values))})


def test_online_runner_preserves_encoder_scores_original_masks_and_reuses_head(ade_data, tmp_path):
    root, task = ade_data
    model = SpatialAdapter().train()
    before = deepcopy(model.state_dict())
    rng, numpy_rng, python_rng = torch.get_rng_state(), np.random.get_state(), random.getstate()
    execution = bench.Execution(device="cpu", workers=0, batch_size=1, progress=False, work_dir=tmp_path / "work")
    result = bench.run(model, names=["ade20k"], data_root=root, output_dir=tmp_path / "dev", execution=execution)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)
    assert all(value.grad is None for value in model.parameters()) and model.training and model.norm.training
    assert torch.equal(rng, torch.get_rng_state()) and python_rng == random.getstate()
    assert np.array_equal(numpy_rng[1], np.random.get_state()[1])
    details = read_json(tmp_path / "dev/results.json")["tasks"][task["id"]]
    assert details["preparation"]["splits"]["development"] == details["preparation"]["splits"]["final"]
    assert details["preparation"]["splits"]["train"]["count"] == 6
    assert len(details["selection"]["candidates"]) == 2
    assert details["selection"]["normalization"] == "trained-batchnorm-in-head"
    records = read_json(tmp_path / "dev" / details["statistics"]["file"])["records"]
    assert sum(row["eligible_pixels"] for row in records) == 20 * 11 + 17 * 22
    intersection = sum(torch.tensor(row["intersection"]) for row in records)
    union = sum(torch.tensor(row["union"]) for row in records)
    assert ade20k.miou(intersection, union) == result.metrics[task["id"]]["miou"]
    assert not (execution.work_dir / "banks").exists()
    heads = list((execution.work_dir / "probes").rglob("*.pt"))
    assert len(heads) == 1
    state = torch.load(heads[0], weights_only=True)
    assert state["norm.num_batches_tracked"] in (2, 3)
    assert state["norm.running_var"].ne(1).any()
    calls = model.calls
    final = bench.run(model, names=["ade20k"], evaluation="final", data_root=root, output_dir=tmp_path / "final", execution=execution)
    assert final.metrics == result.metrics and model.calls - calls == 5
    assert read_json(tmp_path / "final/results.json")["tasks"][task["id"]]["probe_status"] == "reused"
    assert not execution.work_dir.exists()
    calls = model.calls
    assert bench.run(model, names=["ade20k"], evaluation="final", data_root=root, output_dir=tmp_path / "final", execution=execution).metrics == final.metrics
    assert model.calls == calls


def test_online_runner_regenerates_corrupt_head_and_is_reproducible(ade_data, tmp_path):
    root, task = ade_data
    model = SpatialAdapter()
    execution = bench.Execution(device="cpu", workers=0, progress=False, work_dir=tmp_path / "work")
    first = bench.run(model, names=["ade20k"], data_root=root, output_dir=tmp_path / "first", execution=execution)
    next((execution.work_dir / "probes").rglob("*.pt")).write_bytes(b"corrupt")
    second = bench.run(model, names=["ade20k"], evaluation="final", data_root=root, output_dir=tmp_path / "second", execution=execution)
    assert first.metrics == second.metrics
    assert read_json(tmp_path / "second/results.json")["tasks"][task["id"]]["probe_status"] == "regenerated"


def test_best_validation_checkpoint_is_retained_instead_of_last(ade_data, tmp_path, monkeypatch):
    from bench.prepare import load_prepared

    root, task = ade_data
    manifest, training = load_prepared(root, task, "train")
    _, development = load_prepared(root, task, "development")
    from pathlib import Path
    source = Path(manifest["source_root"])
    execution = bench.Execution(device="cpu", workers=0, progress=False)
    encoder = ade20k.preflight(SpatialAdapter(), execution, source, training[0], task)
    scores = iter([80., 60.])
    monkeypatch.setattr(ade20k, "score", lambda *args, **kwargs: ({"score": next(scores)}, []))
    destination = tmp_path / "head.pt"
    selected = ade20k.train_candidate(encoder, training, development, source, task, 3, .01, .001, destination, None)
    assert selected["development_score"] == 80. and selected["updates"] == 2
    assert torch.load(destination, weights_only=True)["norm.num_batches_tracked"] == 2


def test_worker_count_does_not_change_augmented_training_views(ade_data):
    from pathlib import Path

    from bench.prepare import load_prepared

    root, task = ade_data
    manifest, rows = load_prepared(root, task, "train")
    dataset = ade20k.TrainingImages(Path(manifest["source_root"]), rows, task, 100)
    outputs = []
    for workers in (0, 2):
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=ade20k.training_batches(len(rows), task["probe"], 100),
                                             num_workers=workers, generator=torch.Generator().manual_seed(100))
        outputs.append(list(loader))
    for first, second in zip(*outputs):
        for a, b in zip(first[:-1], second[:-1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert first[-1] == second[-1]


def test_visualization_describes_sliding_windows(ade_data, tmp_path):
    from bench.visualize import visualize
    root, _ = ade_data
    report = visualize(names=["ade20k"], data_root=root, output_dir=tmp_path / "gallery")
    html = report.read_text()
    assert "original-resolution labels" in html and "stride 11" in html
    assert "official validation set selects and reports" in html
