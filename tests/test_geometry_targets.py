import numpy as np
import pytest
from PIL import Image

from bench.images import make_canvas, mask_canvas, original_to_canvas
from bench.protocols import resolve_names
from bench.tasks.classification import clevr_target
from bench.tasks.correspondence import eligible_keypoints
from bench.tasks.segmentation import read_mask


@pytest.mark.parametrize("value,target", [(7.9, 0), (8, 0), (8.01, 1), (8.5, 1), (8.51, 2), (9, 2), (9.5, 3), (10, 4), (10.1, 5)])
def test_clevr_distance_matches_pinned_bisect_left_rule(value, target):
    scene = {"objects": [{"pixel_coords": [0, 0, depth]} for depth in [value, 20, 30]]}
    assert clevr_target(scene, resolve_names(["clevr-distance"])[0]) == target


def test_count_uses_all_objects():
    task = resolve_names(["clevr-count"])[0]
    for count in range(3, 11):
        assert clevr_target({"objects": [{}] * count}, task) == count - 3


def test_fit_pad_geometry_and_label_alignment():
    task = resolve_names(["pascal"])[0]
    canvas = make_canvas(Image.new("RGB", (40, 20), "white"), task)
    assert canvas.valid.sum() == 512 * 256
    np.testing.assert_allclose(original_to_canvas([[0, 0], [40, 20]], (40, 20), canvas), [[0, 128], [512, 384]])
    mask = mask_canvas(np.ones((20, 40), dtype=np.uint8), canvas)
    assert (mask[canvas.valid] == 1).all()
    assert (mask[~canvas.valid] == 255).all()


def test_crop_geometry():
    canvas = make_canvas(Image.new("RGB", (512, 256)), resolve_names(["imagenet-100"])[0])
    np.testing.assert_allclose(original_to_canvas([[144, 16], [368, 240]], (512, 256), canvas), [[0, 0], [224, 224]], atol=1e-4)
    assert canvas.valid.all()


def test_ade_label_mapping_and_ignore(tmp_path):
    path = tmp_path / "mask.png"
    Image.fromarray(np.arange(151, dtype=np.uint8)[None]).save(path)
    np.testing.assert_array_equal(read_mask(path, resolve_names(["ade20k"])[0]), [[255, *range(150)]])


@pytest.mark.parametrize("value", [-1, 151, 255])
def test_ade_rejects_labels_outside_source_range(tmp_path, value):
    path = tmp_path / "mask.tiff"
    Image.fromarray(np.array([[0, value]], dtype=np.int32)).save(path)
    with pytest.raises(ValueError, match="labels outside 0..150"):
        read_mask(path, resolve_names(["ade20k"])[0])


@pytest.mark.parametrize("name", ["pascal", "ade20k"])
@pytest.mark.parametrize("size", [(37, 19), (19, 37), (29, 29), (1, 11)])
def test_label_preparation_matches_rendered_canvas_without_rgb_decoding(tmp_path, monkeypatch, name, size):
    from pathlib import Path
    from bench.probes import label_canvas

    task = resolve_names([name])[0]
    image_path, mask_path = tmp_path / "image.png", tmp_path / "mask.png"
    Image.new("RGB", size, (20, 70, 110)).save(image_path)
    mask = np.indices((size[1], size[0])).sum(0).astype(np.uint8) % 3
    Image.fromarray(mask).save(mask_path)
    with Image.open(image_path) as image:
        reference = make_canvas(image, task)
    expected = mask_canvas(read_mask(mask_path, task), reference)
    original_open = Image.open

    def header_only(path, *arguments, **keywords):
        image = original_open(path, *arguments, **keywords)
        if Path(path) == image_path:
            def forbidden():
                raise AssertionError("Label preparation must not decode RGB pixels")
            image.load = forbidden
        return image

    monkeypatch.setattr(Image, "open", header_only)
    labels, transform = label_canvas(tmp_path, {"image": image_path.name, "mask": mask_path.name}, task)
    np.testing.assert_array_equal(labels.numpy(), expected)
    np.testing.assert_array_equal(transform.numpy(), reference.to_original)


def test_spair_eligibility_uses_annotations_not_adapter_coverage():
    data = {"src_kps": [[0, 0], [10, 2], [2, 3]], "trg_kps": [[1, 1], [2, 2], [float("nan"), 2]], "src_imsize": [10, 10, 3], "trg_imsize": [10, 10, 3]}
    assert eligible_keypoints(data)[2].tolist() == [True, False, False]
