import numpy as np
import pytest
import torch
from torch.nn import functional as F

from bench.geometry import logits_on_canvas, sample_original
from bench.probes import LinearProbe, miou, segmentation_counts
from bench.tasks.correspondence import evaluate_pairs, match_pair
from bench.storage import write_json


def test_border_replication_respects_inset_native_centers():
    values = torch.tensor([[[1., 2.], [3., 4.]]])
    affine = torch.tensor([[0.8, 0, 0.1], [0, 0.8, 0.1], [0, 0, 1.]])
    points = torch.tensor([[[0., 0.], [0.3, 0.3], [0.7, 0.7], [1., 1.]]])
    sampled = sample_original(values, affine, points)
    torch.testing.assert_close(sampled, torch.tensor([[[1., 1., 4., 4.]]]))


@pytest.mark.parametrize("grid,size,affine", [
    ((4, 7), (19, 23), [[1, 0, 0], [0, 1.5, -.25], [0, 0, 1]]),
    ((3, 5), (16, 11), [[1, .3, -.1], [.2, 1, -.1], [0, 0, 1]]),
    ((8, 4), (7, 3), [[0, -1, 1], [1, 0, 0], [0, 0, 1]]),
    ((1, 1), (9, 13), [[-1, 0, 1], [0, 1, 0], [0, 0, 1]]),
    ((32, 32), (64, 64), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    ((64, 48), (16, 12), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
])
def test_equal_canvas_transforms_preserve_bilinear_values_and_gradients(grid, size, affine):
    generator = torch.Generator().manual_seed(0)
    values = torch.randn(3, *grid, generator=generator).requires_grad_()
    reference_values = values.detach().double().requires_grad_()
    matrix = torch.tensor(affine, dtype=torch.float32)
    y, x = torch.meshgrid((torch.arange(size[0], dtype=torch.float64) + .5) / size[0], (torch.arange(size[1], dtype=torch.float64) + .5) / size[1], indexing="ij")
    reference = F.grid_sample(reference_values[None], torch.stack([x, y], -1)[None] * 2 - 1, mode="bilinear", padding_mode="border", align_corners=False)[0]
    actual = logits_on_canvas(values, matrix, matrix.clone(), size)
    torch.testing.assert_close(actual.double(), reference, rtol=1e-5, atol=2e-6)
    target = torch.randint(3, (1, *size), generator=generator)
    target[:, 0] = 255
    actual_loss = F.cross_entropy(actual[None], target, ignore_index=255)
    reference_loss = F.cross_entropy(reference[None], target, ignore_index=255)
    actual_loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(actual_loss.double(), reference_loss, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(values.grad.double(), reference_values.grad, rtol=1e-5, atol=1e-7)


def test_different_canvas_transforms_preserve_native_grid_offset():
    values = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    grid = torch.tensor([[.8, .1, .1], [0, .7, .2], [0, 0, 1.]])
    canvas = torch.eye(3)
    y, x = torch.meshgrid((torch.arange(9, dtype=torch.float64) + .5) / 9, (torch.arange(11, dtype=torch.float64) + .5) / 11, indexing="ij")
    points = torch.stack([x, y, torch.ones_like(x)], -1)
    locations = (points @ torch.linalg.inv(grid.double()).T)[..., :2] * 2 - 1
    expected = F.grid_sample(values[None].double(), locations[None], mode="bilinear", padding_mode="border", align_corners=False)[0]
    actual = logits_on_canvas(values, grid, canvas, (9, 11))
    torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=2e-6)


def test_known_matches_grid_ceiling_and_raster_ties():
    features = {"values": torch.eye(4).T.reshape(4, 2, 2), "grid_to_original": torch.eye(3)}
    annotation = {"src_kps": [[25, 25], [75, 75]], "trg_kps": [[25, 25], [75, 75]], "src_imsize": [100, 100, 3], "trg_imsize": [100, 100, 3], "trg_bndbox": [0, 0, 100, 100]}
    assert match_pair(features, features, annotation) == {"correct": 2, "eligible": 2, "grid_ceiling_correct": 2}
    equal = {"values": torch.ones(1, 2, 2), "grid_to_original": torch.eye(3)}
    assert match_pair(equal, equal, annotation)["correct"] == 1
    annotation["trg_kps"] = [[50, 50], [0, 0]]
    assert match_pair(features, features, annotation)["grid_ceiling_correct"] == 0


def test_dataset_miou_reconstructs_from_per_image_statistics():
    pairs = [(torch.tensor([[0, 1], [1, 1]]), torch.tensor([[0, 0], [1, 255]])), (torch.tensor([[0, 0]]), torch.tensor([[1, 1]]))]
    counts = [segmentation_counts(prediction, target, 2) for prediction, target in pairs]
    intersection = sum(count[0] for count in counts)
    union = sum(count[1] for count in counts)
    assert intersection.tolist() == [1, 1]
    assert union.tolist() == [4, 4]
    assert miou(intersection, union) == 25


def test_spair_headline_pools_correspondences_instead_of_categories(tmp_path):
    features = {"values": torch.eye(4).reshape(4, 2, 2), "grid_to_original": torch.eye(3), "extent_in_original": torch.tensor([0., 0., 1., 1.])}
    common = {"src_imsize": [100, 100, 3], "trg_imsize": [100, 100, 3], "trg_bndbox": [0, 0, 100, 100]}
    good = [[25, 25], [75, 25], [25, 75]]
    write_json(tmp_path / "a.json", {**common, "src_kps": good, "trg_kps": good, "category": "a"})
    write_json(tmp_path / "b.json", {**common, "src_kps": [[25, 25]], "trg_kps": [[75, 75]], "category": "b"})
    rows = [{"id": name, "source_id": "source", "target_id": "target", "annotation": name + ".json"} for name in ("a", "b")]
    metrics, supporting, records = evaluate_pairs({"source": features, "target": features}, rows, tmp_path, {"threshold": 0.1})
    assert metrics["pck_at_0.1"] == 75
    assert supporting["category=a"]["pck_at_0.1"] == 100
    assert supporting["category=b"]["pck_at_0.1"] == 0
    assert sum(row["eligible"] for row in records) == 4


@pytest.mark.parametrize("box", [[0, 0, float("nan"), 100], [0, 0, 0, 100], [0, 50, 100, 10]])
def test_spair_rejects_invalid_scoring_boxes(box):
    features = {"values": torch.ones(2, 2, 2), "grid_to_original": torch.eye(3)}
    annotation = {"src_kps": [[25, 25]], "trg_kps": [[25, 25]], "src_imsize": [100, 100, 3], "trg_imsize": [100, 100, 3], "trg_bndbox": box}
    with pytest.raises(ValueError, match="target bounding box"):
        match_pair(features, features, annotation)


def test_unequal_pixel_microbatches_match_effective_batch_gradient():
    one = LinearProbe(3, 2, 0, "cpu")
    two = LinearProbe(3, 2, 0, "cpu")
    features = torch.arange(24).reshape(2, 3, 2, 2).float() / 10
    targets = torch.tensor([[[0, 1], [255, 255]], [[1, 0], [1, 0]]])
    F.cross_entropy(one(features), targets, ignore_index=255).backward()
    valid_count = int((targets != 255).sum())
    for image, target in zip(features, targets):
        (F.cross_entropy(two(image[None]), target[None], ignore_index=255, reduction="sum") / valid_count).backward()
    torch.testing.assert_close(one.weight.grad, two.weight.grad)
    torch.testing.assert_close(one.bias.grad, two.bias.grad)
