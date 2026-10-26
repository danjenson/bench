from copy import deepcopy
import importlib
from pathlib import Path
import runpy
import shutil

import numpy as np
from PIL import Image
import pytest
import torch

import bench
from bench.acquisition import source_recipes
from bench.prepare import prepare
from bench.protocols import recipes
from bench.storage import digest, file_digest, read_json, write_json


def test_ade_search_verifier_reconstructs_dense_counts_and_rejects_corruption(tmp_path, monkeypatch):
    root = tmp_path / "data"
    source = root / "ADEChallengeData2016"
    for split, count in [("training", 10), ("validation", 2)]:
        for kind in ("images", "annotations"):
            (source / kind / split).mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (16, 8), (20 * index, 50, 100)).save(source / "images" / split / f"{index}.jpg")
            mask = np.ones((8, 16), dtype=np.uint8)
            mask[:, :index % 4 + 1] = 0
            mask[:, 8:] = 150
            Image.fromarray(mask).save(source / "annotations" / split / f"{index}.png")
    (source / "objectInfo150.txt").write_text("Idx\tName\n" + "\n".join(f"{i}\tclass-{i}" for i in range(1, 151)))
    protocols = deepcopy(recipes())
    task = protocols["ade20k"]
    task.update(canvas=16, development_fraction=0.2)
    task["probe"].update(epochs=2, batch_size=2, learning_rates=[0.003])
    task["digest"] = digest({key: value for key, value in task.items() if key != "digest"})
    sources = source_recipes()
    sources["ade20k"]["counts"] = {"train": 10, "final": 2}
    monkeypatch.setattr(importlib.import_module("bench.protocols"), "recipes", lambda: protocols)
    monkeypatch.setattr(importlib.import_module("bench.prepare"), "source_recipes", lambda: sources)
    assert prepare(names=["ade20k"], data_root=root)["ready"]

    class Spatial(bench.VisionModel):
        def __init__(self):
            super().__init__()
            self.info = bench.ModelInfo("fixture", "1", {"spatial": "rgb-grid"}, {})

        def encode_spatial(self, batch):
            values = torch.nn.functional.avg_pool2d(batch.images, 4)
            extent = batch.images.new_tensor([0, 0, 1, 1])[None].expand(len(values), -1)
            return bench.Extraction(bench.SpatialFeatures(values, batch.image_to_original, extent))

    model = Spatial()
    execution = bench.Execution(device="cpu", numerics="fp32-v1", workers=0, batch_size=4)
    baseline = tmp_path / "baseline"
    bench.run(model, names=["ade20k"], data_root=root, output_dir=baseline, execution=execution)
    original = read_json(baseline / "results.json")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "CPU fixture")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    repo = Path(__file__).parents[1]
    measure = runpy.run_path(str(repo / "examples/calibrate_monitor_probes.py"))["measure"]
    output = tmp_path / "search"
    measure(model, names=["ade20k"], data_root=root, output=output, execution=execution, stage="search-boundary")
    verify = runpy.run_path(str(repo / "scripts/verify_ade_calibration.py"))["verify"]
    arguments = {"expected_model": original["model"], "software": original["software"]["source_files"],
                 "feature_shape": [3, 4, 4], "baseline": baseline}
    evidence = verify(output, **arguments)
    assert len(evidence["variants"]) == 6 and evidence["verified_records"] == 12
    assert evidence["baseline"]["seed_zero_difference_percentage_points"] == 0
    assert not (output / ".work").exists()

    write_json(output / "verification.json", evidence)
    helper = repo / "examples/calibrate_monitor_probes.py"
    extension_helper = tmp_path / "calibrate_monitor_probes.py"
    source_text = helper.read_text()
    marker = '        budgets = (2, 4) if stage == "search-extension"'
    assert source_text.count(marker) == 1
    extension_helper.write_text(source_text.replace(marker,
        '        if stage == "search-boundary":\n            rates = (30,)\n' + marker))
    extension = tmp_path / "extension"
    runpy.run_path(str(extension_helper))["measure"](model, names=["ade20k"], data_root=root,
        output=extension, execution=execution, stage="search-boundary")
    extension_arguments = {**arguments, "rate_multipliers": (30,), "calibration_helper": extension_helper,
        "previous": output, "previous_calibration_helper": helper}
    extended = verify(extension, **extension_arguments)
    assert extended["verified_records"] == 4 and extended["previous"]["verified_records"] == 12
    assert len(extended["variants"]) == 2
    assert extended["covered_learning_rates"] == [0.003, 0.009000000000000001, 0.03, 0.09]
    assert extended["previous"]["native_parity_origin"] == "previous diagnostic"
    assert not (extension / ".work").exists()
    bad_prior = tmp_path / "bad-prior"
    shutil.copytree(output, bad_prior)
    changed = read_json(bad_prior / "verification.json")
    changed["best_rates_by_epoch_budget"]["2"]["score"] += 1
    write_json(bad_prior / "verification.json", changed)
    with pytest.raises(AssertionError):
        verify(extension, **{**extension_arguments, "previous": bad_prior})
    with pytest.raises(AssertionError):
        verify(extension, **{**extension_arguments, "rate_multipliers": (100,)})

    for mutation in ["metric", "duplicate-row", "wrong-recipe", "inconsistent-class-counts", "retained-workspace", "wrong-extraction-count", "baseline-state", "execution", "script"]:
        altered = tmp_path / mutation
        shutil.copytree(output, altered)
        report = read_json(altered / "calibration.json")
        plan = read_json(altered / "plan.json")
        result = report["results"][plan["variants"][0]["id"]]
        if mutation == "metric":
            result["metrics"]["miou"] += 1
        elif mutation in ("duplicate-row", "inconsistent-class-counts"):
            path = altered / result["statistics"]["file"]
            rows = read_json(path)
            if mutation == "duplicate-row":
                rows["records"][1] = rows["records"][0]
            else:
                rows["records"][0]["union"][0] += 1
            write_json(path, rows)
            result["statistics"]["sha256"] = file_digest(path)
        elif mutation == "wrong-recipe":
            plan["variants"][0]["probe"]["batch_size"] += 1
            write_json(altered / "plan.json", plan)
            report["plan_sha256"] = file_digest(altered / "plan.json")
        elif mutation == "retained-workspace":
            (altered / ".work").mkdir()
        elif mutation == "wrong-extraction-count":
            report["comparisons"][task["id"]]["extraction"]["extracted_examples"] -= 1
        elif mutation == "baseline-state":
            arguments = {**arguments, "expected_model": {**original["model"], "state_digest": "altered"}}
        elif mutation in ("execution", "script"):
            arguments = {**arguments, "expected_model": original["model"]}
            if mutation == "execution":
                plan["execution"]["workers"] += 1
            else:
                plan["script_sha256"] = "0" * 64
            write_json(altered / "plan.json", plan)
            report["plan_sha256"] = file_digest(altered / "plan.json")
        write_json(altered / "calibration.json", report)
        with pytest.raises(AssertionError):
            verify(altered, **arguments)
