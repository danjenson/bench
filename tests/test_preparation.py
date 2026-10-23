import importlib
import json
import subprocess
import sys

import pytest

from bench.prepare import load_prepared, prepare
from bench.protocols import resolve_names
from bench.storage import lock, read_json
from bench.visualize import visualize


def test_names_are_explicit_unique_and_stable():
    assert resolve_names(["pascal"])[0] == resolve_names(["voc2012-linear-v1"])[0]
    for names in ([], "pascal", ["unknown"], ["pascal", "voc2012-linear-v1"]):
        with pytest.raises(ValueError):
            resolve_names(names)
    alone = resolve_names(["clevr-count"])[0]
    assert alone == resolve_names(["pascal", "clevr-count"])[1]


def test_prepare_reuses_one_source_and_regenerates_identical_membership(clevr_data, monkeypatch):
    root, source = clevr_data
    module = importlib.import_module("bench.prepare")
    acquire = module.acquire
    calls = []

    def track(*args, **kwargs):
        calls.append(args[1])
        return acquire(*args, **kwargs)

    monkeypatch.setattr(module, "acquire", track)
    first = prepare(names=["clevr-count", "clevr-distance"], data_root=root)
    assert first["ready"]
    assert calls == ["clevr"]
    tasks = resolve_names(["clevr-count", "clevr-distance"])
    count_manifest, count_rows = load_prepared(root, tasks[0], "train")
    _, distance_rows = load_prepared(root, tasks[1], "train")
    assert [row["id"] for row in count_rows] == [row["id"] for row in distance_rows]
    assert all("CLEVR" not in row["id"] for row in count_rows)
    manifests = {task: result["digest"] for task, result in first["tasks"].items()}
    second = prepare(names=["clevr-distance", "clevr-count"], data_root=root)
    assert {task: result["digest"] for task, result in second["tasks"].items()} == manifests
    # A deleted derived manifest is rebuilt with exactly the same sample IDs and targets.
    directory = root / ".bench" / "prepared" / tasks[0]["id"]
    (directory / count_manifest["splits"]["train"]["file"]).unlink()
    third = prepare(names=["clevr-count"], data_root=root)
    assert third["tasks"][tasks[0]["id"]]["digest"] == manifests[tasks[0]["id"]]


def test_changed_or_missing_source_invalidates_ready_state(clevr_data):
    root, source = clevr_data
    assert prepare(names=["clevr-count"], data_root=root)["ready"]
    task = resolve_names(["clevr-count"])[0]
    _, rows = load_prepared(root, task, "development")
    (source / rows[0]["image"]).unlink()
    with pytest.raises(ValueError, match="bench prepare"):
        load_prepared(root, task, "development")
    report = prepare(names=["clevr-count"], data_root=root)
    assert not report["ready"]
    assert report["tasks"][task["id"]]["status"] == "incomplete"


def test_missing_manual_source_does_not_block_independent_task(clevr_data):
    root, _ = clevr_data
    report = prepare(names=["imagenet-100", "clevr-count"], data_root=root)
    assert not report["ready"]
    assert report["tasks"]["clevr-count-v1"]["status"] == "ready"
    message = report["tasks"]["imagenet100-linear-v1"]["error"]
    assert "https://www.image-net.org/download.php" in message
    assert str(root / "imagenet") in message


def test_preparation_refuses_busy_source(clevr_data):
    root, _ = clevr_data
    with lock(root / ".bench" / "locks" / "clevr.lock"):
        report = prepare(names=["clevr-count"], data_root=root)
    assert not report["ready"]
    assert "Busy" in report["tasks"]["clevr-count-v1"]["error"]


def test_visualization_never_opens_final_partition(clevr_data, tmp_path, monkeypatch):
    root, source = clevr_data
    prepare(names=["clevr-count", "clevr-distance"], data_root=root)
    # Removing every final source and sample-manifest file must not affect visualization.
    for path in (source / "images" / "val").iterdir():
        path.unlink()
    for path in (root / ".bench" / "prepared").rglob("final-*.json"):
        path.unlink()
    output = tmp_path / "report"
    visualize(names=["clevr-count", "clevr-distance"], data_root=root, output_dir=output)
    first = read_json(output / "manifest.json")
    assert all(task["partition"] == "development" for task in first["tasks"])
    assert len(list(output.glob("*-input.png"))) == 6
    html = (output / "index.html").read_text()
    assert 'src="http' not in html
    assert "Closest camera depth" in html and "classifier target" in html
    visualize(names=["clevr-count"], data_root=root, output_dir=tmp_path / "alone")
    alone = read_json(tmp_path / "alone" / "manifest.json")
    assert alone["tasks"][0] == first["tasks"][0]


def test_tilde_and_relative_paths(clevr_data, tmp_path, monkeypatch):
    root, _ = clevr_data
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    tilde = prepare(names=["clevr-count"], data_root="~/data")
    relative = prepare(names=["clevr-count"], data_root="data")
    assert tilde["data_root"] == relative["data_root"] == str(root)
    assert tilde["tasks"]["clevr-count-v1"]["digest"] == relative["tasks"]["clevr-count-v1"]["digest"]


def test_setup_imports_do_not_import_torch():
    result = subprocess.run([sys.executable, "-c", "import sys; import bench; import bench.prepare; import bench.visualize; assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_corrupt_manifest_fails_with_matching_command(clevr_data):
    root, _ = clevr_data
    prepare(names=["clevr-count"], data_root=root)
    task = resolve_names(["clevr-count"])[0]
    path = root / ".bench" / "prepared" / task["id"] / "manifest.json"
    data = json.loads(path.read_text())
    data["splits"]["development"]["count"] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="bench prepare.*clevr-count"):
        load_prepared(root, task, "development")
