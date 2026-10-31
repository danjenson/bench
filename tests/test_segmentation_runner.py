from copy import deepcopy
import importlib
from pathlib import Path
import runpy

import numpy as np
from PIL import Image
import torch
import pytest

import bench
from bench.acquisition import source_recipes
from bench.prepare import prepare
from bench.protocols import recipes
from bench.probes import miou
from bench.storage import digest, file_digest, read_json, write_json


@pytest.mark.parametrize('candidate_stage', ['candidate-budget', 'candidate-regularized', 'candidate-regularized-long', 'candidate-regularized-tail', 'selected-monitor'])
def test_spatial_only_runner_and_compact_iou_reconstruction(tmp_path, monkeypatch, candidate_stage):
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
    if candidate_stage == 'selected-monitor':
        task['id'] = 'voc-resolution-calibration-fixture'
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
    statistics = read_json(output / "statistics" / f"{task['id']}.json")["records"]
    intersection = sum(torch.tensor(row["intersection"]) for row in statistics)
    union = sum(torch.tensor(row["union"]) for row in statistics)
    assert miou(intersection, union) == result.metrics[task["id"]]["miou"]
    assert sum(row["eligible_pixels"] for row in statistics) == 2 * 16 * 8
    assert not (output / ".work").exists()
    probes = importlib.import_module("bench.probes")
    selected = read_json(output / "results.json")["tasks"][task["id"]]["selection"]
    with monkeypatch.context() as uncached:
        uncached.setattr(probes.PreparedTargets, "byte_limit", 0)
        repeated = tmp_path / "uncached"
        reference = bench.run(SpatialOnly(), names=["pascal"], data_root=root, output_dir=repeated,
                              execution=bench.Execution(device="cpu", numerics="fp32-v1", workers=0))
    assert reference.metrics == result.metrics
    assert read_json(repeated / 'statistics' / f"{task['id']}.json")["records"] == statistics
    assert read_json(repeated / "results.json")["tasks"][task["id"]]["selection"] == selected
    assert not (repeated / ".work").exists()

    # The shared candidate uses a longer dense budget than the global probe.
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "CPU fixture")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    repo = Path(__file__).parents[1]
    measure = runpy.run_path(str(repo / "examples/calibrate_monitor_probes.py"))["measure"]
    calibration = tmp_path / candidate_stage
    monitor_recipe = {task["id"]: {"base_updates": 32, "learning_rate_multipliers": [30, 100], "weight_decay": 1.0}} if candidate_stage == "selected-monitor" else None
    report = measure(SpatialOnly(), names=["pascal"], data_root=root, output=calibration,
                     execution=bench.Execution(device="cpu", numerics="fp32-v1", workers=0), stage=candidate_stage, monitor_recipe=monitor_recipe)
    plan = read_json(calibration / "plan.json")
    expected = {128} if candidate_stage == 'candidate-regularized-tail' else {64, 128} if candidate_stage == 'candidate-regularized-long' else {32, 64}
    assert {v["probe"]["updates"] for v in plan["variants"]} == expected
    assert len(report["results"]) == len(expected) * (2 if candidate_stage == "selected-monitor" else 1) and not (calibration / ".work").exists()
    if candidate_stage.startswith('candidate-regularized') or candidate_stage == 'selected-monitor':
        assert all(v['probe']['weight_decays'] == [1.0] for v in plan['variants'])
        assert all(v['calibration']['weight_decay_reference'] == task['probe']['weight_decays'] for v in plan['variants'])
    verifier = runpy.run_path(str(repo / "scripts/monitor_calibration.py"))
    request = tmp_path / "candidate-request.json"
    write_json(request, {"pins": {}, "package_root": str(Path(bench.__file__).parent),
        "names": ["pascal"], "stage": candidate_stage, "variants": plan["variants"], "monitor_recipe": monitor_recipe,
        "output": str(calibration), "expected_model": plan["model"], "numerics": "fp32-v1",
        "gpu_uuid": "fixture", "cpu_threads": 2, "data_root": str(root),
        "feature_widths": {"spatial": 3}, "measure_script": str(repo / "examples/calibrate_monitor_probes.py")})
    write_json(calibration / "caller.json", {"request_sha256": file_digest(request),
        "encoder_state_unchanged": True, "gpu_uuid": "fixture", "cpu_threads": 2})
    proof = verifier["verify"](request)
    assert proof["verified_records"] == 2 * len(report["results"])
    if candidate_stage == 'candidate-regularized-tail':
        assert proof['budget_checks'] == {task['id']: {}}
