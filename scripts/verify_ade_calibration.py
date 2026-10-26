"""Independently verify retained full-ADE LR/budget development diagnostics."""

from copy import deepcopy
import math
from pathlib import Path
import statistics

import numpy as np
from PIL import Image
from omegaconf import OmegaConf

from bench.prepare import load_prepared
from bench.protocols import resolve_names
from bench.storage import contained_path, digest, file_digest, read_json


def source_label_counts(data_root, task, development, source_root):
    targets, checked = {}, {}
    for row in development:
        for relative, expected in row["files"].items():
            if relative not in checked:
                checked[relative] = file_digest(contained_path(source_root, relative))
            assert checked[relative] == expected
        with Image.open(contained_path(source_root, row["image"])) as image:
            width, height = image.size
        with Image.open(contained_path(source_root, row["mask"])) as mask:
            assert mask.size == (width, height)
            raw = np.asarray(mask)
            assert raw.ndim == 2 and 0 <= raw.min() <= raw.max() <= task["classes"]
            scale = task["canvas"] / max(width, height)
            size = (max(1, round(width * scale)), max(1, round(height * scale)))
            labels = np.asarray(mask.resize(size, Image.Resampling.NEAREST)).reshape(-1)
            targets[row["id"]] = np.bincount(labels, minlength=task["classes"] + 1)[1:].tolist()
    return targets, {"development_images": len(development), "verified_source_files": len(checked),
        "source_checksums_digest": digest(checked), "per_image_class_counts_digest": digest(targets),
        "eligible_pixels": sum(sum(values) for values in targets.values()),
        "geometry": "Nearest-neighbor masks after fit/pad; raw ADE zero and padding are ignored."}


def mapped_model(model, mapping):
    assert model["digest"] == digest({key: value for key, value in model.items() if key != "digest"})

    def transform(value):
        if isinstance(value, str):
            for before, after in sorted(mapping.items(), key=lambda pair: -len(pair[0])):
                if value == before or value.startswith(before + "/"):
                    return str(after) + value[len(before):]
            return value
        if isinstance(value, list):
            return [transform(item) for item in value]
        if isinstance(value, dict):
            return {key: transform(item) for key, item in value.items()}
        return value

    relocated = transform(model)
    relocated["digest"] = digest({key: value for key, value in relocated.items() if key != "digest"})
    return relocated


def verify(directory, *, expected_model, software, feature_shape, baseline, path_mapping=None,
           rate_multipliers=(1, 3, 10), calibration_helper=None, previous=None,
           previous_calibration_helper=None):
    directory, baseline = Path(directory), Path(baseline)
    plan, report = read_json(directory / "plan.json"), read_json(directory / "calibration.json")
    task = resolve_names(["ade20k"])[0]
    assert task["kind"] == "segmentation" and task["classes"] == 150
    assert report["status"] == "complete" and report["plan_sha256"] == file_digest(directory / "plan.json")
    assert report["retained_temporary_bytes"] == 0 and not (directory / ".work").exists()
    assert plan["stage"] == "search-boundary" and plan["numerics"] == "fp32-v1"
    assert plan["model"] == expected_model and plan["software"] == software
    assert plan["gates"] == {"budget_absolute_percentage_points": {}, "scale_absolute_percentage_points": {task["id"]: task["numerical_tolerance"]["absolute_percentage_points"]}}
    prepared, training = load_prepared(Path(plan["data_root"]), task, "train", verify=False)
    _, development = load_prepared(Path(plan["data_root"]), task, "development", verify=False)
    targets, source_evidence = source_label_counts(Path(plan["data_root"]), task, development, Path(prepared["source_root"]))
    rates, budgets = tuple(rate_multipliers), (1, 2)
    assert (rates == (1, 3, 10) and previous is None and previous_calibration_helper is None) or (
        rates == (30,) and previous is not None and previous_calibration_helper is not None)
    assert len(plan["variants"]) == len(report["results"]) == len(rates) * len(budgets)
    assert {variant["id"] for variant in plan["variants"]} == set(report["results"])
    assert set(report["comparisons"]) == {task["id"]}
    base_rate = max(task["probe"]["learning_rates"])
    cases, scores, checks = set(), {}, {}
    native_records, native_score = None, None
    for variant in plan["variants"]:
        calibration = variant["calibration"]
        budget, rate = calibration["budget_multiplier"], calibration["learning_rate_multiplier"]
        assert budget in budgets and rate in rates and (budget, rate) not in cases
        cases.add((budget, rate))
        assert calibration == {"base_task": task["id"], "base_task_digest": task["digest"],
            "feature_scale_before_standardization": 1.0, "scale_arithmetic": "float32",
            "budget_multiplier": budget, "learning_rate_multiplier": rate, "learning_rate_reference": [base_rate]}
        expected = deepcopy(task)
        expected["probe"].update(epochs=task["probe"]["epochs"] * budget, learning_rates=[base_rate * rate], seeds=[0])
        identifier = f"{task['id']}-calibration-e{expected['probe']['epochs']}-s1-lrm{rate}"
        expected.update(id=identifier, name=identifier, release_status="calibration-only", calibration=calibration)
        expected["digest"] = digest({key: value for key, value in expected.items() if key != "digest"})
        assert variant == expected
        result = report["results"][identifier]
        assert result["preparation_digest"] == prepared["digest"] and result["sample_digests"] == prepared["splits"]
        assert result["source_root"] == prepared["source_root"]
        statistics_path = contained_path(directory, result["statistics"]["file"])
        assert file_digest(statistics_path) == result["statistics"]["sha256"]
        saved = read_json(statistics_path)
        assert saved["task"] == identifier and saved["evaluation"] == "development"
        records = saved["records"]
        assert [(row["seed"], row["id"]) for row in records] == [(0, row["id"]) for row in development]
        for row in records:
            assert len(row["intersection"]) == len(row["union"]) == task["classes"]
            assert all(type(i) is type(u) is int and 0 <= i <= u for i, u in zip(row["intersection"], row["union"]))
            truth = targets[row["id"]]
            predicted = [u - t + i for i, u, t in zip(row["intersection"], row["union"], truth)]
            assert row["eligible_pixels"] == sum(truth) == sum(predicted)
            assert all(0 <= i <= min(t, p) and u <= row["eligible_pixels"] for i, u, t, p in zip(row["intersection"], row["union"], truth, predicted))
            assert sum(row["intersection"]) + sum(row["union"]) == 2 * row["eligible_pixels"]
        intersections = [sum(column) for column in zip(*(row["intersection"] for row in records))]
        unions = [sum(column) for column in zip(*(row["union"] for row in records))]
        score = 100 * statistics.mean(i / u for i, u in zip(intersections, unions) if u)
        assert len(result["selection"]) == len(result["seed_results"]) == len(result["candidate_scores"]) == 1
        head, seed = result["selection"][0], result["seed_results"][0]
        assert head == result["candidate_scores"][0]
        assert head["seed"] == seed["seed"] == 0
        assert head["learning_rate"] == base_rate * rate and head["weight_decay"] in task["probe"]["weight_decays"]
        assert len(task["probe"]["weight_decays"]) == 1
        assert 1 <= head["epoch"] <= expected["probe"]["epochs"]
        assert head["updates"] == head["epoch"] * math.ceil(len(training) / task["probe"]["batch_size"])
        assert head["probe_parameters"] == task["classes"] * (feature_shape[0] + 1)
        assert all(math.isfinite(head[field]) for field in ("training_loss", "development_loss"))
        assert len(head["head_sha256"]) == 64 and int(head["head_sha256"], 16) >= 0
        assert seed["sample_count"] == len(development) and seed["eligible_count"] == sum(row["eligible_pixels"] for row in records)
        assert all(math.isclose(score, value, rel_tol=0, abs_tol=1e-10) for value in (result["metrics"]["miou"], head["development_score"], seed["score"]))
        scores[budget, rate] = score
        checks[identifier] = {"records": len(records), "reconstructed_miou": score, "learning_rate": base_rate * rate,
            "epoch_budget": expected["probe"]["epochs"], "selected_epoch": head["epoch"],
            "selected_epoch_at_budget_boundary": head["epoch"] == expected["probe"]["epochs"],
            "statistics_sha256": file_digest(statistics_path)}
        if budget == rate == 1:
            native_records, native_score = records, score
    assert cases == {(budget, rate) for budget in budgets for rate in rates}
    comparison = report["comparisons"][task["id"]]
    assert comparison["extraction"]["extracted_examples"] == len(training) + len(development)
    assert comparison["extraction"]["reused_examples"] == 0
    expected_checks = {str(rate): {"base_score": scores[1, rate], "doubled_budget_score": scores[2, rate],
        "doubled_budget_difference": scores[2, rate] - scores[1, rate]} for rate in rates}
    assert set(comparison["learning_rate_checks"]) == set(expected_checks)
    for rate, expected in expected_checks.items():
        actual = comparison["learning_rate_checks"][rate]
        assert set(actual) == set(expected)
        assert all(math.isclose(actual[key], value, rel_tol=0, abs_tol=1e-10) for key, value in expected.items())
    reference = read_json(baseline / "results.json")
    assert reference["status"] == "complete" and reference["evaluation"] == "development"
    assert mapped_model(reference["model"], path_mapping or {}) == expected_model
    assert reference["software"]["source_files"] == software
    resolved = OmegaConf.to_container(OmegaConf.load(baseline / "resolved.yaml"), resolve=True)
    assert plan["execution"] == {key: resolved["execution"][key] for key in ("device", "batch_size", "workers")}
    helper = Path(calibration_helper) if calibration_helper is not None else Path(__file__).with_name("calibrate_monitor_probes.py")
    if not helper.exists():
        root = next(path for path in Path(__file__).resolve().parents if (path / "SPEC.md").is_file())
        helper = root / "examples/calibrate_monitor_probes.py"
    assert plan["script_sha256"] == file_digest(helper)
    prior = None
    if previous is not None:
        previous = Path(previous)
        prior = verify(previous, expected_model=expected_model, software=software,
            feature_shape=feature_shape, baseline=baseline, path_mapping=path_mapping,
            calibration_helper=previous_calibration_helper)
        committed = read_json(previous / "verification.json")
        assert all(committed[key] == value for key, value in prior.items() if key != "verifier_sha256")
        assert any(value["at_upper_rate_boundary"] for value in prior["best_rates_by_epoch_budget"].values())
        assert prior["source_labels"] == source_evidence
        previous_report = read_json(previous / "calibration.json")
        native_id = f"{task['id']}-calibration-e{task['probe']['epochs']}-s1-lrm1"
        native = previous_report["results"][native_id]
        native_records = read_json(contained_path(previous, native["statistics"]["file"]))["records"]
        native_score = prior["variants"][native_id]["reconstructed_miou"]
        for value in prior["variants"].values():
            budget = value["epoch_budget"] // task["probe"]["epochs"]
            rate = round(value["learning_rate"] / base_rate)
            assert (budget, rate) not in scores
            scores[budget, rate] = value["reconstructed_miou"]
    item = reference["tasks"][task["id"]]
    assert item["preparation_digest"] == prepared["digest"] and item["task_digest"] == task["digest"]
    reference_path = contained_path(baseline, item["statistics"]["file"])
    assert file_digest(reference_path) == item["statistics"]["sha256"]
    reference_rows = [row for row in read_json(reference_path)["records"] if row["seed"] == 0]
    assert [(row["id"], row["eligible_pixels"]) for row in reference_rows] == [(row["id"], row["eligible_pixels"]) for row in native_records]
    reference_score = next(seed["score"] for seed in item["seed_results"] if seed["seed"] == 0)
    assert item["selection"]["seeds"][0]["learning_rate"] == base_rate, "The baseline did not select its upper LR boundary"
    tolerance = task["numerical_tolerance"]["absolute_percentage_points"]
    assert abs(native_score - reference_score) <= tolerance + 1e-12
    best = {}
    covered_rates = sorted({rate for _, rate in scores})
    for budget in budgets:
        assert {rate for candidate_budget, rate in scores if candidate_budget == budget} == set(covered_rates)
        rate = min(covered_rates, key=lambda rate: (-scores[budget, rate], rate))
        best[str(task["probe"]["epochs"] * budget)] = {"selected_learning_rate": base_rate * rate,
            "score": scores[budget, rate], "at_upper_rate_boundary": rate == max(covered_rates)}
    evidence = {"status": "verified", "verifier_sha256": file_digest(__file__), "plan_sha256": file_digest(directory / "plan.json"),
        "calibration_sha256": file_digest(directory / "calibration.json"), "calibration_script_sha256": file_digest(helper),
        "execution": plan["execution"], "variants": checks,
        "verified_records": sum(value["records"] for value in checks.values()), "source_labels": source_evidence,
        "best_rates_by_epoch_budget": best, "baseline": {"results_sha256": file_digest(baseline / "results.json"),
            "seed_zero_difference_percentage_points": native_score - reference_score, "tolerance_percentage_points": tolerance,
            "changed_prediction_hashes": sum(a["prediction_sha256"] != b["prediction_sha256"] for a, b in zip(native_records, reference_rows))},
        "scope": "Full-ADE development LR/budget diagnostic; registered score tolerance retained. Upper-boundary selections remain unresolved. No full-probe convergence threshold or release acceptance is inferred."}
    if prior is not None:
        evidence["previous"] = {"directory": str(previous), "calibration_sha256": prior["calibration_sha256"],
            "verification_sha256": file_digest(previous / "verification.json"),
            "verified_records": prior["verified_records"], "native_parity_origin": "previous diagnostic"}
        evidence["covered_learning_rates"] = [base_rate * rate for rate in covered_rates]
        evidence["scope"] = "Two new ADE development fits plus independently reconstructed prior search records. The original native-parity proof is retained; no overlapping probe is rerun. Upper selections remain unresolved, with unchanged recipes and tolerances. No full-probe convergence or release acceptance is inferred."
    return evidence
