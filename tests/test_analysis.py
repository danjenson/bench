import math

from bench.analysis import numerical_parity, paired_contrast, paired_difference
import pytest

from bench.protocols import resolve_names
from bench.storage import digest, file_digest, read_json, write_json


def test_paired_contrasts_group_repeated_probe_seeds(tmp_path):
    paths = [tmp_path / name for name in ("baseline", "improved", "same")]
    for index, path in enumerate(paths):
        rows = [{"id": str(image), "seed": seed, "label": image % 2, "prediction": image % 2 if index == 1 else 1 - image % 2} for image in range(10) for seed in (0, 1)]
        write_json(path / "statistics.json", {"records": rows})
        write_json(path / "results.json", {"schema_version": "1.0.0", "numerics": "fp32-v1", "tasks": {"imagenet100-linear-v1": {"status": "complete", "task_digest": "same", "preparation_digest": "same", "evaluation": "development", "metrics": {"top1": {}}, "statistics": {"file": "statistics.json", "sha256": file_digest(path / "statistics.json")}}}})
    result = paired_difference(paths[0], paths[1], "imagenet100-linear-v1", replicates=50)
    assert result["difference_percentage_points"] == 100
    assert result["paired_95_interval"] == [100, 100]
    assert result["source_images"] == 10 and result["probe_seeds"] == [0, 1]
    assert result["stratified"]
    interaction = paired_contrast([paths[1], paths[0], paths[1], paths[2]], [1, -1, -1, 1], "imagenet100-linear-v1", replicates=50)
    assert interaction["paired_95_interval"] == [0, 0]
    for path in paths:
        artifact = read_json(path / "results.json")
        artifact.update(schema_version="1.1.0", future_metadata={"note": "additive field"})
        artifact["tasks"]["imagenet100-linear-v1"]["future_metadata"] = {"note": "additive task field"}
        write_json(path / "results.json", artifact)
    assert paired_difference(paths[0], paths[1], "imagenet100-linear-v1", replicates=50) == result
    artifact = read_json(paths[0] / "results.json")
    artifact["schema_version"] = "2.0.0"
    write_json(paths[0] / "results.json", artifact)
    with pytest.raises(ValueError, match="Unsupported result schema"):
        paired_difference(paths[0], paths[1], "imagenet100-linear-v1")
    artifact["schema_version"] = "1.1.0"
    write_json(paths[0] / "results.json", artifact)
    (paths[1] / "statistics.json").write_text('{}')
    with pytest.raises(ValueError, match="corrupt committed statistics"):
        paired_difference(paths[0], paths[1], "imagenet100-linear-v1")


def profile_results(root, name, records, metrics):
    task = resolve_names([name])[0]
    outputs = []
    for profile, rows, values in zip(("fp32-v1", "bf16-v1"), records, metrics):
        output = root / profile
        model = {"info": {"name": "fixture", "revision": "fixed", "readouts": {"global": "global", "spatial": "spatial"}, "recipe": {}}, "state_digest": "same", "numerics": profile}
        model["digest"] = digest(model)
        write_json(output / "statistics.json", {"records": rows})
        write_json(output / "results.json", {"schema_version": "1.0.0", "status": "complete", "numerics": profile, "model": model, "software": {"source_files": {"probes.py": "same"}}, "tasks": {task["id"]: {"status": "complete", "task_digest": task["digest"], "preparation_digest": "same", "data_digest": "same", "evaluation": "development", "method": task["method"], "readout": task["method"], "model_identity": model["digest"], "feature_shape": [3], "sample_count": len({row["id"] for row in rows}), "metrics": {name: {"value": value} for name, value in values.items()}, "statistics": {"file": "statistics.json", "sha256": file_digest(output / "statistics.json")}}}})
        outputs.append(output)
    return outputs


@pytest.mark.parametrize("extra_correct,passed", [(1, True), (2, False)])
def test_numerical_parity_uses_registered_tolerance_and_keeps_profiles_separate(tmp_path, extra_correct, passed):
    records = [[{"id": str(image), "seed": 0, "label": image % 100, "prediction": (image + (image >= 500 + added)) % 100} for image in range(1000)] for added in (0, extra_correct)]
    outputs = profile_results(tmp_path, "imagenet-100", records, [{"top1": 50}, {"top1": 50 + extra_correct / 10}])
    report = numerical_parity(*outputs, "imagenet-100")
    assert report["passed"] is passed
    assert report["tolerance_percentage_points"] == .1
    assert report["metrics"]["top1"]["absolute_difference"] == pytest.approx(extra_correct / 10)
    assert report["sample_count"] == 1000 and report["probe_seeds"] == [0]
    assert numerical_parity(*reversed(outputs), "imagenet100-linear-v1") == report
    with pytest.raises(ValueError, match="same numerical profile"):
        paired_difference(*outputs, "imagenet100-linear-v1")
    with pytest.raises(ValueError, match="one fp32-v1 and one bf16-v1"):
        numerical_parity(outputs[0], outputs[0], "imagenet-100")


@pytest.mark.parametrize("name", ["clevr-count", "pascal", "spair71k"])
def test_numerical_parity_reconstructs_seed_means_dataset_iou_and_pooled_pck(tmp_path, name):
    if name == "clevr-count":
        records = [{"id": str(image), "seed": seed, "label": 0, "prediction": int(image > seed)} for seed in (0, 1, 2) for image in range(4)]
        metrics = {"accuracy": 50, "accuracy_std": math.sqrt(1250 / 3)}
    elif name == "pascal":
        records = [{"id": str(image), "seed": 0, "intersection": intersection + [0] * 19, "union": union + [0] * 19, "eligible_pixels": eligible} for image, (intersection, union, eligible) in enumerate([([1, 0], [2, 1], 2), ([0, 3], [1, 4], 4), ([0, 0], [0, 0], 0)])]
        metrics = {"miou": 100 * (1 / 3 + 3 / 5) / 2}
    else:
        records = [{"id": str(image), "source_id": "shared", "target_id": str(image), "correct": correct, "eligible": eligible, "grid_ceiling_correct": eligible, "breakdown": {"category": str(image)}} for image, (correct, eligible) in enumerate([(1, 1), (0, 3)])]
        metrics = {"pck_at_0.1": 25, "grid_limited_pck_ceiling": 100}
    outputs = profile_results(tmp_path, name, [records, list(reversed(records))], [metrics, metrics])
    report = numerical_parity(*outputs, name)
    assert report["passed"]
    assert {name: row["scores"]["fp32-v1"] for name, row in report["metrics"].items()} == pytest.approx(metrics)
    assert all(row["absolute_difference"] == 0 for row in report["metrics"].values())


def test_numerical_parity_does_not_cancel_opposite_seed_changes(tmp_path):
    records = [[{"id": str(image), "seed": seed, "label": 0, "prediction": int(image >= counts[seed])} for seed in (0, 1, 2) for image in range(4)] for counts in ([1, 2, 3], [3, 2, 1])]
    metrics = {"accuracy": 50, "accuracy_std": math.sqrt(1250 / 3)}
    outputs = profile_results(tmp_path, "clevr-count", records, [metrics, metrics])
    report = numerical_parity(*outputs, "clevr-count")
    assert all(row["passed"] for row in report["metrics"].values())
    assert not report["passed"]
    assert [row["absolute_difference"] for row in report["per_seed"].values()] == [50, 0, 50]


@pytest.mark.parametrize("change", ["state", "recipe", "software", "task", "partition", "shape", "samples", "seed", "label", "metric", "checksum", "model-digest"])
def test_numerical_parity_rejects_unmatched_or_inconsistent_results(tmp_path, change):
    records = [{"id": str(image), "seed": 0, "label": 0, "prediction": 0} for image in range(3)]
    outputs = profile_results(tmp_path, "imagenet-100", [records, records], [{"top1": 100}, {"top1": 100}])
    output = outputs[1]
    result = read_json(output / "results.json")
    task = result["tasks"]["imagenet100-linear-v1"]
    if change in ("state", "recipe", "model-digest"):
        model = result["model"]
        if change == "recipe":
            model["info"]["recipe"]["readout_layer"] = "different"
        else:
            model["state_digest"] = "different"
        if change != "model-digest":
            model["digest"] = digest({key: value for key, value in model.items() if key != "digest"})
            task["model_identity"] = model["digest"]
    elif change == "software":
        result["software"]["source_files"]["probes.py"] = "different"
    elif change in ("task", "partition", "shape"):
        key, value = {"task": ("task_digest", "different"), "partition": ("evaluation", "final"), "shape": ("feature_shape", [4])}[change]
        task[key] = value
    elif change == "metric":
        task["metrics"]["top1"]["value"] = 0
    else:
        statistics = read_json(output / "statistics.json")
        key, value = {"samples": ("id", "different"), "seed": ("seed", 1), "label": ("label", 1), "checksum": ("prediction", 1)}[change]
        statistics["records"][0][key] = value
        write_json(output / "statistics.json", statistics)
        if change != "checksum":
            task["statistics"]["sha256"] = file_digest(output / "statistics.json")
    write_json(output / "results.json", result)
    with pytest.raises(ValueError):
        numerical_parity(*outputs, "imagenet-100")
