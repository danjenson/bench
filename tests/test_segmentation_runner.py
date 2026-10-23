from copy import deepcopy
import importlib

import numpy as np
from PIL import Image
import torch

import bench
from bench.acquisition import source_recipes
from bench.prepare import prepare
from bench.protocols import recipes
from bench.probes import miou
from bench.storage import digest, read_json


def test_spatial_only_runner_and_compact_iou_reconstruction(tmp_path, monkeypatch):
    root = tmp_path / "data"
    source = root / "VOC2012"
    (source / "ImageSets" / "Segmentation").mkdir(parents=True)
    (source / "JPEGImages").mkdir()
    (source / "SegmentationClass").mkdir()
    for split, indices in [("train", range(4)), ("val", range(4, 6))]:
        (source / "ImageSets" / "Segmentation" / f"{split}.txt").write_text("\n".join(str(index) for index in indices))
        for index in indices:
            Image.new("RGB", (16, 8), (30 * index, 50, 100)).save(source / "JPEGImages" / f"{index}.jpg")
            mask = np.zeros((8, 16), dtype=np.uint8)
            mask[:, index + 1:] = 1
            Image.fromarray(mask).save(source / "SegmentationClass" / f"{index}.png")
    protocols = recipes()
    task = deepcopy(protocols["pascal"])
    task.update(canvas=16, bank={"train": 2, "development": 2, "final": 2})
    task["probe"] = {**task["probe"], "updates": 2, "batch_size": 2}
    task["digest"] = digest({key: value for key, value in task.items() if key != "digest"})
    protocols["pascal"] = task
    sources = source_recipes()
    sources["voc2012"]["counts"] = {"train": 4, "final": 2}
    monkeypatch.setattr(importlib.import_module("bench.protocols"), "recipes", lambda: protocols)
    monkeypatch.setattr(importlib.import_module("bench.prepare"), "source_recipes", lambda: sources)
    assert prepare(names=["pascal"], data_root=root)["ready"]

    class SpatialOnly(bench.VisionModel):
        def __init__(self):
            super().__init__()
            self.info = bench.ModelInfo("fixture-spatial", "1", {"spatial": "rgb-grid"}, {})

        def encode_spatial(self, batch):
            values = torch.nn.functional.avg_pool2d(batch.images, 4)
            return bench.Extraction(bench.SpatialFeatures(values, batch.image_to_original, batch.images.new_tensor([0, 0, 1, 1])[None].expand(len(values), -1)))

    output = tmp_path / "output"
    result = bench.run(SpatialOnly(), names=["pascal"], data_root=root, output_dir=output, execution=bench.Execution(device="cpu", numerics="fp32-v1", workers=0))
    statistics = read_json(output / "statistics" / "voc2012-linear-v1.json")["records"]
    intersection = sum(torch.tensor(row["intersection"]) for row in statistics)
    union = sum(torch.tensor(row["union"]) for row in statistics)
    assert miou(intersection, union) == result.metrics[task["id"]]["miou"]
    assert sum(row["eligible_pixels"] for row in statistics) == 2 * 16 * 8
    assert not (output / ".work").exists()
