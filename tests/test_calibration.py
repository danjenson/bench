import importlib
from copy import deepcopy
from pathlib import Path
import runpy

import pytest
import torch

from bench import Execution, run
from bench.prepare import load_prepared, prepare
from bench.protocols import resolve_names
from bench.storage import digest, file_digest, read_json
from test_features import GlobalAdapter


def test_full_probe_calibration_preserves_recipe_and_prunes_variants(clevr_data, tmp_path, monkeypatch):
    root, _ = clevr_data
    fixture_task = importlib.import_module("bench.protocols").recipes()["clevr-count"]
    fixture_task["probe"]["learning_rates"] = [0.01, 0.03]
    fixture_task["probe"]["weight_decays"] = [0.0, 0.1]
    fixture_task["digest"] = digest({key: value for key, value in fixture_task.items() if key != "digest"})
    prepare(names=["clevr-count"], data_root=root)
    original_digest = resolve_names(["clevr-count"])[0]["digest"]
    # Only CUDA telemetry is replaced; extraction, fitting, scoring and cleanup run on CPU.
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "CPU fixture")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    measure = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "calibrate_monitor_probes.py"))["measure"]
    model = GlobalAdapter()
    output = tmp_path / "calibration"
    report = measure(model, names=["clevr-count"], data_root=root, output=output, execution=Execution(device="cpu", numerics="fp32-v1", workers=0, batch_size=4), stage="search-boundary")
    plan = read_json(output / "plan.json")
    assert report["status"] == "complete" and report["retained_temporary_bytes"] == 0
    assert not (output / ".work").exists()
    assert model.calls == 3  # All six variants share the twelve-image feature bank.
    assert resolve_names(["clevr-count"])[0]["digest"] == original_digest
    assert len(plan["variants"]) == len(report["results"]) == 6
    assert len({variant["digest"] for variant in plan["variants"]}) == 6
    assert sorted({variant["probe"]["learning_rates"][0] for variant in plan["variants"]}) == pytest.approx([0.03, 0.09, 0.3])
    assert {variant["probe"]["epochs"] for variant in plan["variants"]} == {2, 4}
    for variant in plan["variants"]:
        assert variant["probe"]["seeds"] == [0]
        assert variant["probe"]["weight_decays"] == [0.0, 0.1]
        result = report["results"][variant["id"]]
        assert len(result["candidate_scores"]) == 2
        assert all("head_file" not in candidate for candidate in result["candidate_scores"])
        statistics = output / result["statistics"]["file"]
        assert file_digest(statistics) == result["statistics"]["sha256"]
        records = read_json(statistics)["records"]
        assert len(records) == 3 and all(record["seed"] == 0 for record in records)
    checks = report["comparisons"]["clevr-count-v1"]["learning_rate_checks"]
    assert all("budget_passed" not in check for check in checks.values())
    extended_output = tmp_path / "extension"
    extended = measure(model, names=["clevr-count"], data_root=root, output=extended_output, execution=Execution(device="cpu", numerics="fp32-v1", workers=0, batch_size=4), stage="search-extension")
    extended_plan = read_json(extended_output / "plan.json")
    assert extended["status"] == "complete" and extended["retained_temporary_bytes"] == 0
    assert not (extended_output / ".work").exists()
    assert model.calls == 6
    assert resolve_names(["clevr-count"])[0]["digest"] == original_digest
    assert len(extended_plan["variants"]) == len(extended["results"]) == 10
    assert {variant["probe"]["epochs"] for variant in extended_plan["variants"]} == {4, 8}
    assert sorted({variant["probe"]["learning_rates"][0] for variant in extended_plan["variants"]}) == pytest.approx([0.03, 0.09, 0.3, 0.9, 3.0])
    original_variants = {variant["id"]: variant for variant in plan["variants"]}
    overlapping = 0
    for variant in extended_plan["variants"]:
        result = extended["results"][variant["id"]]
        assert variant["probe"]["seeds"] == [0] and variant["probe"]["weight_decays"] == [0.0, 0.1]
        assert len(result["candidate_scores"]) == 2
        statistics = extended_output / result["statistics"]["file"]
        assert file_digest(statistics) == result["statistics"]["sha256"]
        if variant["id"] in original_variants:
            overlapping += 1
            assert variant == original_variants[variant["id"]]
            for key in ("metrics", "selection", "candidate_scores", "statistics"):
                assert result[key] == report["results"][variant["id"]][key]
    assert overlapping == 3
    assert extended_plan["gates"]["budget_absolute_percentage_points"] == {}
    assert all("budget_passed" not in check for check in extended["comparisons"]["clevr-count-v1"]["learning_rate_checks"].values())

    decay_output = tmp_path / "weight-decay"
    decay = measure(model, names=["clevr-count"], data_root=root, output=decay_output, execution=Execution(device="cpu", numerics="fp32-v1", workers=0, batch_size=4), stage="search-weight-decay")
    decay_plan = read_json(decay_output / "plan.json")
    assert decay["status"] == "complete" and decay["retained_temporary_bytes"] == 0
    assert not (decay_output / ".work").exists() and model.calls == 9
    assert resolve_names(["clevr-count"])[0]["digest"] == original_digest
    assert len(decay_plan["variants"]) == len(decay["results"]) == 2
    assert {v["probe"]["epochs"] for v in decay_plan["variants"]} == {2, 4}
    for variant in decay_plan["variants"]:
        assert variant["probe"]["learning_rates"] == [0.01, 0.03]
        assert variant["probe"]["weight_decays"] == [0.0, 0.1, 1.0]
        assert variant["probe"]["seeds"] == [0]
        assert variant["calibration"]["weight_decay_reference"] == [0.0, 0.1]
        assert variant["calibration"]["weight_decay_multiplier"] == 10
        candidates = decay["results"][variant["id"]]["candidate_scores"]
        assert {(c["learning_rate"], c["weight_decay"]) for c in candidates} == {(lr, wd) for lr in [0.01, 0.03] for wd in [0.0, 0.1, 1.0]}
        baseline = next(v for v in plan["variants"] if v["probe"]["epochs"] == variant["probe"]["epochs"] and v["calibration"]["learning_rate_multiplier"] == 1)
        assert [c for c in candidates if c["learning_rate"] == 0.03 and c["weight_decay"] != 1.0] == report["results"][baseline["id"]]["candidate_scores"]
        statistics = decay_output / decay["results"][variant["id"]]["statistics"]["file"]
        assert file_digest(statistics) == decay["results"][variant["id"]]["statistics"]["sha256"]
        records = read_json(statistics)["records"]
        expected_score = 100 * sum(r["label"] == r["prediction"] for r in records) / len(records)
        assert decay["results"][variant["id"]]["metrics"][variant["metric"]] == expected_score
    assert decay_plan["gates"]["budget_absolute_percentage_points"] == {}
    with pytest.raises(ValueError, match="searched grid with a positive upper bound"):
        measure(model, names=["ade20k"], data_root=root, output=tmp_path / "fixed-decay", execution=Execution(device="cpu", numerics="fp32-v1"), stage="search-weight-decay")
    assert not (tmp_path / "fixed-decay").exists()

    joint_output = tmp_path / "joint-grid"
    joint = measure(model, names=["clevr-count"], data_root=root, output=joint_output, execution=Execution(device="cpu", numerics="fp32-v1", workers=0, batch_size=4), stage="search-grid")
    joint_plan = read_json(joint_output / "plan.json")
    assert joint["status"] == "complete" and joint["retained_temporary_bytes"] == 0
    assert not (joint_output / ".work").exists() and model.calls == 12
    assert resolve_names(["clevr-count"])[0]["digest"] == original_digest
    assert len(joint_plan["variants"]) == len(joint["results"]) == 2
    for variant in joint_plan["variants"]:
        assert variant["probe"]["learning_rates"] == pytest.approx([0.01, 0.03, 0.09, 0.3])
        assert variant["probe"]["weight_decays"] == [0.0, 0.1, 1.0]
        assert variant["probe"]["seeds"] == [0]
        assert variant["calibration"]["learning_rate_extension_multipliers"] == [3, 10]
        result = joint["results"][variant["id"]]
        candidates = result["candidate_scores"]
        assert len(candidates) == 12
        assert {(c["learning_rate"], c["weight_decay"]) for c in candidates} == {(lr, wd) for lr in [0.01, 0.03, 0.09, 0.3] for wd in [0.0, 0.1, 1.0]}
        prior = next(v for v in decay_plan["variants"] if v["probe"]["epochs"] == variant["probe"]["epochs"])
        assert [c for c in candidates if c["learning_rate"] in [0.01, 0.03]] == decay["results"][prior["id"]]["candidate_scores"]
        assert file_digest(joint_output / result["statistics"]["file"]) == result["statistics"]["sha256"]


@pytest.mark.parametrize("stage", ["budget-scale", "convergence", "variance-floor"])
def test_monitor_calibration_fits_declared_variants_without_changing_named_recipe(stage, clevr_data, tmp_path, monkeypatch):
    root, _ = clevr_data
    protocols = importlib.import_module("bench.protocols").recipes()
    # Reuse generated fixture images; this is not a scientific ImageNet recipe.
    task = deepcopy(protocols["clevr-distance"])
    task.update(name="imagenet-100", id="imagenet100-linear-v1")
    task["probe"].pop("epochs")
    task["probe"].update(updates=2, learning_rates=[0.01], seeds=[0], selection="final-update")
    task["digest"] = digest({key: value for key, value in task.items() if key != "digest"})
    protocols["imagenet-100"] = task
    original = deepcopy(task)
    assert prepare(names=["imagenet-100"], data_root=root)["ready"]
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "CPU fixture")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    measure = runpy.run_path(str(Path(__file__).parents[1] / "examples" / "calibrate_monitor_probes.py"))["measure"]
    model = GlobalAdapter()
    output = tmp_path / stage
    report = measure(model, names=["imagenet-100"], data_root=root, output=output, execution=Execution(device="cpu", numerics="fp32-v1", workers=0, batch_size=4), stage=stage)
    assert report["status"] == "complete" and report["retained_temporary_bytes"] == 0
    assert not (output / ".work").exists()
    assert model.calls == 3
    assert resolve_names(["imagenet-100"])[0] == original
    plan = read_json(output / "plan.json")
    assert len(plan["variants"]) == len(report["results"]) == (8 if stage == "convergence" else 4)
    assert len({v["digest"] for v in plan["variants"]}) == len(plan["variants"])
    assert {v["probe"]["updates"] for v in plan["variants"]} == ({8, 16} if stage == "convergence" else {2, 4})
    for variant in plan["variants"]:
        result = report["results"][variant["id"]]
        assert result["selection"][0]["updates"] == variant["probe"]["updates"]
        assert result["selection"][0]["seed"] == 0
        statistics = output / result["statistics"]["file"]
        assert file_digest(statistics) == result["statistics"]["sha256"]
        if stage == "variance-floor":
            assert variant["standardization"]["variance_floor"] == 1e-8
            assert variant["calibration"]["variance_floor_reference"] == original["standardization"]["variance_floor"]
            assert "-floor1e-08" in variant["id"]
        elif stage == "budget-scale":
            assert variant["standardization"] == original["standardization"]
    comparison = report["comparisons"][task["id"]]
    checks = comparison["learning_rate_checks"] if stage == "convergence" else {"1": comparison}
    assert len(checks) == (4 if stage == "convergence" else 1)
    assert all(check["budget_tolerance"] == 0.5 for check in checks.values())
    for check in checks.values():
        assert check["budget_passed"] == (abs(check["doubled_budget_score"] - check["base_score"]) <= 0.5)
    if stage in ("budget-scale", "variance-floor"):
        assert set(comparison["scale_score_differences"]) == {"0.1", "10.0"}
        assert comparison["scale_passed"] == all(abs(delta) <= comparison["scale_tolerance"] for delta in comparison["scale_score_differences"].values())
    if stage == "budget-scale":
        baseline_output = tmp_path / "native-monitoring"
        run(model, names=["imagenet-100"], evaluation="development", data_root=root, output_dir=baseline_output, execution=Execution(device="cpu", numerics="fp32-v1", workers=0, batch_size=4))
        baseline = read_json(baseline_output / "results.json")["tasks"][task["id"]]
        native = next(variant for variant in plan["variants"] if variant["calibration"]["budget_multiplier"] == 1 and variant["calibration"]["feature_scale_before_standardization"] == 1.0)
        calibrated = report["results"][native["id"]]
        assert calibrated["selection"] == baseline["selection"]["seeds"]
        assert calibrated["seed_results"] == baseline["seed_results"]
        assert read_json(output / calibrated["statistics"]["file"])["records"] == read_json(baseline_output / baseline["statistics"]["file"])["records"]


@pytest.mark.parametrize("name", ["imagenet-100", "pascal"])
@pytest.mark.parametrize("stage", ["fuller-reference", "fuller-convergence", "fuller-selection"])
def test_fuller_reference_expands_only_training_and_reuses_preparation(name, stage, tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image

    from bench import Extraction, SpatialFeatures
    from bench.acquisition import source_recipes
    from bench.probes import miou

    root = tmp_path / "data"
    protocols = importlib.import_module("bench.protocols").recipes()
    task = protocols[name]
    task["canvas"] = 16
    task["probe"].update(updates=2, batch_size=4)
    sources = source_recipes()
    source = root / sources[task["source"]]["layouts"][0]
    if name == "imagenet-100":
        task.pop("class_list")
        task.pop("class_list_sha256")
        task.update(classes=2, bank={"train_per_class": 3, "development_per_class": 2, "final_per_class": 2})
        for split, count in [("train", 12), ("val", 3)]:
            for synset in ["n01440764", "n01443537"]:
                directory = source / split / synset
                directory.mkdir(parents=True)
                for index in range(count):
                    Image.new("RGB", (16, 12), (index * 17, 80, 20)).save(directory / f"{index}.jpg")
    else:
        task["bank"] = {"train": 4, "development": 2, "final": 3}
        sources[task["source"]]["counts"] = {"train": 12, "final": 4}
        for directory in ["ImageSets/Segmentation", "JPEGImages", "SegmentationClass"]:
            (source / directory).mkdir(parents=True)
        for split, indices in [("train", range(12)), ("val", range(12, 16))]:
            (source / "ImageSets/Segmentation" / f"{split}.txt").write_text("\n".join(map(str, indices)))
            for index in indices:
                Image.new("RGB", (16, 12), (index * 13, 80, 20)).save(source / "JPEGImages" / f"{index}.jpg")
                mask = np.zeros((12, 16), dtype=np.uint8)
                mask[:, index % 12:] = 1
                Image.fromarray(mask).save(source / "SegmentationClass" / f"{index}.png")
    task["digest"] = digest({key: value for key, value in task.items() if key != "digest"})
    original_task = deepcopy(task)
    monkeypatch.setattr(importlib.import_module("bench.protocols"), "recipes", lambda: protocols)
    monkeypatch.setattr(importlib.import_module("bench.prepare"), "source_recipes", lambda: sources)
    assert prepare(names=[name], data_root=root)["ready"]
    original, original_training = load_prepared(root, task, "train")
    _, development = load_prepared(root, task, "development")
    _, final = load_prepared(root, task, "final")
    original_manifest = root / ".bench/prepared" / task["id"] / "manifest.json"
    original_sha = file_digest(original_manifest)
    helper = runpy.run_path(str(Path(__file__).parents[1] / "examples/calibrate_monitor_probes.py"))
    reference = helper["prepare_fuller_reference"](root, task)
    prepared, training = load_prepared(root, reference, "train")
    assert len(training) == (20 if name == "imagenet-100" else 10)
    assert all(row in training for row in original_training)
    assert not {row["id"] for row in training} & {row["id"] for row in development + final}
    assert prepared["labels"] == original["labels"]
    assert load_prepared(root, reference, "development")[1] == development
    assert load_prepared(root, reference, "final")[1] == final
    assert reference["digest"] != task["digest"]
    assert file_digest(original_manifest) == original_sha
    # Preparation is reusable, and fitting must not open final examples.
    reference_manifest = root / ".bench/prepared" / reference["id"] / "manifest.json"
    reference_sha = file_digest(reference_manifest)
    for row in final:
        (source / row["image"]).unlink()
        if "mask" in row:
            (source / row["mask"]).unlink()

    class Adapter(GlobalAdapter):
        def __init__(self):
            super().__init__()
            self.info.readouts["spatial"] = "rgb-grid"

        def encode_spatial(self, batch):
            self.calls += 1
            values = torch.nn.functional.avg_pool2d(batch.images, 4)
            return Extraction(SpatialFeatures(values, batch.image_to_original, batch.images.new_tensor([0, 0, 1, 1])[None].expand(len(values), -1)))

    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "CPU fixture")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: 0)
    scoring_history = {}
    if stage == "fuller-selection":
        probes = importlib.import_module("bench.probes")
        score_probe = probes.score

        def record_score(probe, bank, rows, source, recipe, *args, **kwargs):
            result = score_probe(probe, bank, rows, source, recipe, *args, **kwargs)
            scoring_history.setdefault(recipe["id"], []).append(result[0]["score"])
            return result

        monkeypatch.setattr(probes, "score", record_score)
    model = Adapter()
    output = tmp_path / "fuller"
    report = helper["measure"](model, names=[name], data_root=root, output=output, execution=Execution(device="cpu", numerics="fp32-v1", workers=0, batch_size=4), stage=stage)
    assert report["status"] == "complete" and report["retained_temporary_bytes"] == 0
    assert not (output / ".work").exists()
    assert model.calls == (len(training) + len(development) + 3) // 4
    assert resolve_names([name])[0] == original_task
    assert file_digest(original_manifest) == original_sha
    assert file_digest(reference_manifest) == reference_sha
    plan = read_json(output / "plan.json")
    assert plan["reference_preparation"][task["id"]] == reference
    assert plan["preparation_seconds_excluded_from_evaluation"] >= 0
    assert len(plan["variants"]) == len(report["results"]) == 8
    epochs = 50 if name == "imagenet-100" else 40
    updates = 8 if stage == "fuller-reference" else epochs * ((len(training) + 3) // 4)
    assert {variant["probe"]["updates"] for variant in plan["variants"]} == {updates, updates * 2}
    for variant in plan["variants"]:
        assert variant["id"].startswith(reference["id"])
        result = report["results"][variant["id"]]
        selected = result["selection"][0]
        if stage == "fuller-selection":
            assert variant["id"].endswith("-bestdev")
            assert variant["probe"]["scoring"] == "each-epoch"
            assert variant["calibration"]["selection_reference"] == "final-update"
            scores = scoring_history[variant["id"]]
            assert len(scores) == variant["calibration"]["reference_epochs"] + 1
            assert selected["development_score"] == scores[-1] == max(scores[:-1])
            assert selected["epoch"] == scores[:-1].index(max(scores[:-1])) + 1
            assert selected["updates"] == selected["epoch"] * ((len(training) + 3) // 4)
        else:
            assert selected["updates"] == variant["probe"]["updates"]
        if stage != "fuller-reference":
            assert variant["calibration"]["reference_epochs"] == epochs * variant["calibration"]["budget_multiplier"]
            if stage == "fuller-convergence":
                assert selected["epoch"] == variant["calibration"]["reference_epochs"]
        assert result["sample_digests"] == prepared["splits"]
        statistics = output / result["statistics"]["file"]
        assert file_digest(statistics) == result["statistics"]["sha256"]
        records = read_json(statistics)["records"]
        assert [row["id"] for row in records] == [row["id"] for row in development]
        if name == "imagenet-100":
            assert [row["label"] for row in records] == [row["target"] for row in development]
            score = sum(row["label"] == row["prediction"] for row in records) / len(records) * 100
        else:
            score = miou(sum(torch.tensor(row["intersection"]) for row in records), sum(torch.tensor(row["union"]) for row in records))
        assert result["metrics"][task["metric"]] == pytest.approx(score)
    if stage == "fuller-selection":
        assert any(report["results"][v["id"]]["selection"][0]["updates"] < v["probe"]["updates"] for v in plan["variants"])
    extra = next(row for row in training if row not in original_training)
    Image.new("RGB", (16, 12), (1, 2, 3)).save(source / extra["image"])
    with pytest.raises(ValueError, match="Source checksum changed"):
        helper["prepare_fuller_reference"](root, task)
