from copy import deepcopy
import importlib
import json

import pytest
from PIL import Image
import torch

import bench
from bench.analysis import paired_difference
from bench.features import FeatureBank, save_features
from bench.prepare import load_prepared, prepare
from bench.probes import score
from bench.protocols import recipes
from bench.storage import digest, file_digest, read_json, write_json
from bench.tasks.classification import apply_reimagenet
from test_features import GlobalAdapter
from test_analysis import profile_results


@pytest.fixture
def reimagenet_data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    source = root / "imagenet"
    synsets = ["n01440764", "n01443537", "n01484850"]
    for label, synset in enumerate(synsets):
        for i in range(3):
            path = source / "train" / synset / f"{i}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (16, 16), (30 + label * 80, i * 60, 50)).save(path)
    annotations = []
    # Reordered class indices exercise synset remapping; unknown labels are not targets.
    for i, (target, accepted) in enumerate([(0, [2]), (0, [1]), (1, []), (2, [-1]), (2, [2, -1])]):
        path = source / "val" / synsets[target] / f"{i}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 16), (30 + target * 80, i * 40, 50)).save(path)
        annotations.append({"image_name": path.name, "file_path": f"{synsets[target]}/{path.name}", "original_class": [[1, 2, 0][target]], "reannotated_labels": accepted})
    labels = source / "reimagenet"
    write_json(labels / "label_names.json", [synsets[2], synsets[0], synsets[1]])
    write_json(labels / "class_update_config.json", {"eq_classes": {"equivalent": [0, 1]}})
    (labels / "reannotation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in annotations))
    declared = deepcopy(recipes())
    task = declared["imagenet-1k"]
    task.update(classes=3, canvas=16, resize_short=20)
    task["probe"].pop("search_plan")
    task["probe"].pop("budget_validation")
    task["probe"].update(epochs=1, batch_size=4, learning_rates=[.01], weight_decays=[0.], seeds=[0, 1])
    task["final_labels"]["files"] = {name: file_digest(labels / name) for name in task["final_labels"]["files"]}
    task["digest"] = digest({k: v for k, v in task.items() if k != "digest"})
    monkeypatch.setattr("bench.protocols.recipes", lambda: deepcopy(declared))
    module = importlib.import_module("bench.prepare")
    sources = module.source_recipes()
    sources["imagenet"]["counts"] = {"train": 9, "final": 5}
    monkeypatch.setattr(module, "source_recipes", lambda: sources)
    return root, source, task


def test_prepare_uses_all_training_and_shared_reannotated_validation(reimagenet_data):
    root, source, task = reimagenet_data
    assert prepare(names=["imagenet-1k"], data_root=root)["ready"]
    manifest, rows = load_prepared(root, task, "final")
    by_file = {row["image"].split("/")[-1]: row for row in rows}
    assert [by_file[f"{i}.png"]["accepted_labels"] for i in range(5)] == [[1], [0, 2], [], [], [1]]
    assert manifest["final_labels"]["counts"] == {"eligible": 3, "no_valid_label": 1, "uncertain": 1}
    _, training = load_prepared(root, task, "train")
    _, development = load_prepared(root, task, "development")
    assert len(training) == 9
    assert all("accepted_labels" not in row for row in training)
    assert {row["id"] for row in training}.isdisjoint(row["id"] for row in rows)
    assert development == rows
    assert prepare(names=["imagenet-1k"], data_root=root)["tasks"][task["id"]]["digest"] == manifest["digest"]
    with (source / "reimagenet/reannotation.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="ReImageNet source checksum changed"):
        load_prepared(root, task, "final")
    with pytest.raises(ValueError, match="ReImageNet source checksum changed"):
        load_prepared(root, task, "development")


@pytest.mark.parametrize("change", ["missing", "duplicate", "class", "original", "synsets", "checksum"])
def test_annotation_contract_rejects_invalid_inputs(reimagenet_data, change):
    root, source, task = reimagenet_data
    assert prepare(names=["imagenet-1k"], data_root=root)["ready"]
    _, rows = load_prepared(root, task, "final")
    directory = source / "reimagenet"
    annotations = [json.loads(line) for line in (directory / "reannotation.jsonl").read_text().splitlines()]
    if change == "missing": annotations.pop()
    elif change == "duplicate": annotations.append(annotations[0])
    elif change == "class": annotations[0]["reannotated_labels"] = [True]
    elif change == "original": annotations[0]["original_class"] = [0]
    elif change == "synsets": write_json(directory / "label_names.json", ["n01440764"] * 3)
    elif change == "checksum": annotations[0]["reannotated_labels"] = [0]
    (directory / "reannotation.jsonl").write_text("".join(json.dumps(row) + "\n" for row in annotations))
    if change != "checksum":
        task["final_labels"]["files"] = {name: file_digest(directory / name) for name in task["final_labels"]["files"]}
    with pytest.raises(ValueError):
        apply_reimagenet(source, rows, task)


def test_score_and_analysis_use_accepted_labels_and_exclude_unknowns(reimagenet_data, tmp_path):
    root, _, task = reimagenet_data
    assert prepare(names=["imagenet-1k"], data_root=root)["ready"]
    _, rows = load_prepared(root, task, "final")
    rows.sort(key=lambda row: row["image"].rsplit("/", 1)[-1])
    predictions = [1, 2, 1, 2, 0]
    path = tmp_path / "features.safetensors"
    save_features(path, {"values": torch.eye(3)[predictions], "costs": {}})
    bank = FeatureBank(tmp_path, {"shape": [3], "shards": [{"file": path.name, "ids": [row["id"] for row in rows]}]})
    result, records = score(torch.nn.Identity(), bank, rows, None, task, {"mean": torch.zeros(3), "std": torch.ones(3)}, "cpu", 0, progress=False)
    bank.close()
    assert result["score"] == pytest.approx(200 / 3)
    assert result["original_top1"] == 40
    assert result["eligible_count"] == 3 and result["sample_count"] == 5
    repeated = records + [{**row, "seed": 1} for row in records]
    metrics = {"top1": 200 / 3, "top1_std": 0, "original_top1": 40, "original_top1_std": 0}
    outputs = profile_results(tmp_path / "profiles", "imagenet-1k", [repeated, repeated], [metrics, metrics])
    from bench.analysis import numerical_parity
    assert numerical_parity(*outputs, "imagenet-1k")["passed"]
    # Paired uncertainty retains all seeds together, only for eligible images.
    artifact = read_json(outputs[1] / "results.json")
    artifact["numerics"] = "fp32-v1"
    write_json(outputs[1] / "results.json", artifact)
    paired = paired_difference(*outputs, task["id"], replicates=10)
    assert paired["difference_percentage_points"] == 0 and paired["source_images"] == 3


def test_runner_reports_new_labels_provenance_and_original_score(reimagenet_data, tmp_path):
    root, _, task = reimagenet_data
    prepare(names=["imagenet-1k"], data_root=root)
    output = tmp_path / "result"
    result = bench.run(GlobalAdapter(), names=["imagenet-1k"], evaluation="final", data_root=root, output_dir=output, execution=bench.Execution(device="cpu", workers=0, batch_size=4, progress=False))
    assert set(result.metrics[task["id"]]) == {"top1", "top1_std", "original_top1", "original_top1_std"}
    artifact = read_json(output / "results.json")["tasks"][task["id"]]
    assert artifact["preparation"]["split_rule"] == "official-train-validation"
    assert artifact["preparation"]["splits"]["train"]["count"] == 9
    assert artifact["preparation"]["splits"]["development"]["count"] == 5
    assert artifact["preparation"]["final_labels"]["revision"] == task["final_labels"]["revision"]
    assert all(seed["eligible_count"] == 3 and seed["sample_count"] == 5 for seed in artifact["seed_results"])
    for selected, reported in zip(artifact["selection"]["seeds"], artifact["seed_results"]):
        assert selected["development_score"] == reported["score"]
