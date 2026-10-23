import importlib
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import bench
from bench.prepare import prepare
from bench.storage import read_json, write_json
from test_features import GlobalAdapter


def cpu_execution(work_dir=None):
    return bench.Execution(device="cpu", numerics="fp32-v1", workers=0, batch_size=4, work_dir=work_dir)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
def test_classification_scoring_preserves_rows_and_partial_batch_loss(batch_size, tmp_path):
    import math
    from bench.features import FeatureBank, save_features
    from bench.probes import score

    points = [(2, 0), (0, 2), (-1, -1), (1, 1), (-2, 0), (0, -2), (0, 0), (3, 1), (-1, 2)]
    targets = [0, 0, 2, 1, 2, 0, 0, 0, 1]
    predictions = [0, 1, 2, 0, 2, 2, 0, 0, 1]
    rows = [{"id": f"sample-{index}", "target": target} for index, target in enumerate(targets)]
    path = tmp_path / "features.safetensors"
    save_features(path, {"values": torch.tensor(points, dtype=torch.float32), "costs": {}})
    bank = FeatureBank(tmp_path, {"shape": [2], "shards": [{"file": path.name, "ids": [row["id"] for row in rows]}]})
    probe = torch.nn.Linear(2, 3, bias=False)
    with torch.no_grad():
        probe.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]))
    task = {"kind": "classification", "classes": 3, "probe": {"batch_size": batch_size}}
    statistics = {"mean": torch.zeros(2), "std": torch.ones(2)}
    metrics, records = score(probe, bank, rows, None, task, statistics, "cpu", 7)
    bank.close()
    losses = []
    for (x, y), target in zip(points, targets):
        logits = [x, y, -x - y]
        losses.append(math.log(sum(math.exp(value) for value in logits)) - logits[target])
    assert metrics["score"] == pytest.approx(100 * 6 / 9)
    assert metrics["loss"] == pytest.approx(sum(losses) / len(losses), abs=1e-6)
    assert metrics["sample_count"] == metrics["eligible_count"] == 9
    assert records == [{"id": row["id"], "seed": 7, "label": row["target"], "prediction": prediction} for row, prediction in zip(rows, predictions)]


def test_standalone_result_reuse_and_cleanup(clevr_data, tmp_path):
    root, _ = clevr_data
    prepare(names=["clevr-count"], data_root=root)
    model = GlobalAdapter()
    model.train()
    model.layer.eval()
    rng = torch.random.get_rng_state().clone()
    output = tmp_path / "result"
    result = bench.run(model, names=["clevr-count"], data_root=root, output_dir=output, execution=cpu_execution())
    assert result.status == "complete" and "clevr-count-v1" in result.metrics
    assert not (output / ".work").exists()
    assert torch.equal(rng, torch.random.get_rng_state())
    assert model.training and not model.layer.training
    artifact = read_json(output / "results.json")
    assert artifact["tasks"]["clevr-count-v1"]["diagnostics"]["sample_count"] == 9
    assert artifact["storage"]["retained_workspace_bytes"] == 0
    assert artifact["storage"]["retained_temporary_bytes"] == 0
    assert artifact["storage"]["peak_temporary_bytes"] > 0
    assert artifact["storage"]["estimated_peak_temporary_bytes"] >= artifact["storage"]["peak_temporary_bytes"]
    timing = artifact["tasks"]["clevr-count-v1"]["extraction_work"][0]["timing"]
    assert timing["first_batch_examples"] == 4 and timing["steady_state_examples"] == 8
    assert 0 < artifact["timing"]["time_to_first_task_seconds"] < artifact["timing"]["total_seconds"]
    count = model.calls
    repeated = bench.run(model, names=["clevr-count"], data_root=root, output_dir=output, execution=cpu_execution())
    assert repeated.metrics == result.metrics and model.calls == count
    with pytest.raises(ValueError, match="Conflicting run identity"):
        bench.run(model, names=["clevr-count"], evaluation="final", data_root=root, output_dir=output, execution=cpu_execution())


@pytest.mark.parametrize("default_dtype", [torch.float16, torch.float64])
def test_probe_precision_is_independent_of_caller_default_dtype(clevr_data, tmp_path, default_dtype):
    root, _ = clevr_data
    prepare(names=["clevr-count"], data_root=root)
    model = GlobalAdapter().float()
    original_dtype = torch.get_default_dtype()
    reports = []
    try:
        for name, dtype in [("baseline", torch.float32), ("changed-default", default_dtype)]:
            torch.set_default_dtype(dtype)
            output = tmp_path / name
            result = bench.run(model, names=["clevr-count"], data_root=root, output_dir=output, execution=cpu_execution())
            assert result.status == "complete" and not (output / ".work").exists()
            assert torch.get_default_dtype() == dtype
            reports.append(read_json(output / "results.json"))
    finally:
        torch.set_default_dtype(original_dtype)
    before, after = [report["tasks"]["clevr-count-v1"] for report in reports]
    assert reports[0]["model"] == reports[1]["model"]
    assert before["metrics"] == after["metrics"]
    assert before["statistics"]["sha256"] == after["statistics"]["sha256"]
    assert before["selection"] == after["selection"]


@pytest.mark.parametrize("change", ["pillow", "preprocessing-source", "safetensors"])
def test_preprocessing_changes_invalidate_completed_results_and_probes(clevr_data, tmp_path, monkeypatch, change):
    import bench.features as features
    import bench.images as images

    root, _ = clevr_data
    prepare(names=["clevr-count"], data_root=root)
    model = GlobalAdapter()
    execution = cpu_execution(tmp_path / "workspace")
    output = tmp_path / "original"
    bench.run(model, names=["clevr-count"], data_root=root, output_dir=output, execution=execution)
    previous_calls = model.calls
    if change == "pillow":
        monkeypatch.setattr(features, "pillow_version", "changed-version")
    elif change == "safetensors":
        monkeypatch.setattr(features, "safetensors_version", "changed-version")
    else:
        original_digest = features.file_digest
        monkeypatch.setattr(features, "file_digest", lambda path: "changed-source" if Path(path) == Path(images.__file__) else original_digest(path))
    with pytest.raises(ValueError, match="Conflicting run identity"):
        bench.run(model, names=["clevr-count"], data_root=root, output_dir=output, execution=execution)
    replacement = tmp_path / "replacement"
    bench.run(model, names=["clevr-count"], data_root=root, output_dir=replacement, execution=execution)
    assert model.calls > previous_calls
    assert read_json(replacement / "results.json")["tasks"]["clevr-count-v1"]["probe_status"] == "fitted"


@pytest.mark.parametrize("changed_source", ["probes.py", "runner.py"])
def test_evaluation_source_changes_reject_old_results_and_scope_probe_reuse(clevr_data, tmp_path, monkeypatch, changed_source):
    from bench import probes, protocols, runner
    from bench.storage import file_digest

    root, _ = clevr_data
    prepare(names=["clevr-count"], data_root=root)
    model = GlobalAdapter()
    execution = cpu_execution(tmp_path / "workspace")
    output = tmp_path / "original"
    bench.run(model, names=["clevr-count"], data_root=root, output_dir=output, execution=execution)
    original = read_json(output / "results.json")
    changed = Path(runner.__file__).parent / changed_source
    for module in (protocols, probes, runner):
        monkeypatch.setattr(module, "file_digest", lambda path: "f" * 64 if Path(path) == changed else file_digest(path), raising=False)
    with pytest.raises(ValueError, match="Conflicting run identity"):
        bench.run(model, names=["clevr-count"], data_root=root, output_dir=output, execution=execution)
    replacement = tmp_path / "replacement"
    bench.run(model, names=["clevr-count"], data_root=root, output_dir=replacement, execution=execution)
    updated = read_json(replacement / "results.json")
    assert updated["model"] == original["model"]
    before, after = original["tasks"]["clevr-count-v1"], updated["tasks"]["clevr-count-v1"]
    assert after["probe_status"] == ("fitted" if changed_source == "probes.py" else "reused")
    assert (before["selection"]["digest"] != after["selection"]["digest"]) == (changed_source == "probes.py")


@pytest.mark.parametrize("changed_source, affected", [
    ("probes.py", {"classification", "segmentation"}),
    ("geometry.py", {"segmentation", "correspondence"}),
    ("tasks/segmentation.py", {"segmentation"}),
    ("tasks/correspondence.py", {"correspondence"}),
])
def test_evaluation_identity_tracks_only_relevant_scoring_sources(monkeypatch, changed_source, affected):
    from bench import protocols

    before = {kind: protocols.evaluation_source_digest({"kind": kind}) for kind in ("classification", "segmentation", "correspondence")}
    original_digest = protocols.file_digest
    changed = Path(protocols.__file__).parent / changed_source
    monkeypatch.setattr(protocols, "file_digest", lambda path: "f" * 64 if Path(path) == changed else original_digest(path))
    for kind, identity in before.items():
        assert (protocols.evaluation_source_digest({"kind": kind}) != identity) == (kind in affected)


def test_cli_evaluation_preserves_nested_model_provenance(clevr_data, tmp_path, monkeypatch):
    from torchvision.models import resnet18
    from bench.cli.main import main

    root, _ = clevr_data
    prepare(names=["clevr-count"], data_root=root)
    checkpoint = tmp_path / "resnet.pt"
    torch.save(resnet18(weights=None).state_dict(), checkpoint)
    output = tmp_path / "cli-result"
    monkeypatch.setattr(sys, "argv", ["bench", "model=resnet18", f"model.checkpoint={checkpoint}", "model.provenance={pretraining:{dataset:fixture,partitions:[train]},model_selection:{rule:fixed}}", "names=[clevr-count]", f"data_root={root}", f"output_dir={output}", "execution.device=cpu", "execution.numerics=fp32-v1", "execution.workers=0", "execution.batch_size=4"])
    main()
    result = read_json(output / "results.json")
    assert result["status"] == "complete"
    assert result["exposure"] == {"pretraining": {"dataset": "fixture", "partitions": ["train"]}, "model_selection": {"rule": "fixed"}}
    assert not (output / ".work").exists()


def test_shared_clevr_bank_and_final_selection_reuse(clevr_data, tmp_path, monkeypatch):
    root, _ = clevr_data
    prepare(names=["clevr-count", "clevr-distance"], data_root=root)
    workspace = tmp_path / "workspace"
    model = GlobalAdapter()
    bench.run(model, names=["clevr-count", "clevr-distance"], data_root=root, output_dir=tmp_path / "development", execution=cpu_execution(workspace))
    assert model.calls == 3  # 12 distinct training/development images, shared by both labels.
    assert not list((workspace / "banks").glob("*/shard-*"))
    assert len(list((workspace / "probes").glob("*/selection.json"))) == 2
    artifact = read_json(tmp_path / "development" / "results.json")
    retained = sum(path.stat().st_size for path in (workspace / "probes").rglob("*") if path.is_file())
    assert artifact["storage"]["retained_temporary_bytes"] == retained
    module = importlib.import_module("bench.runner")

    def forbidden(*args, **kwargs):
        raise AssertionError("Final evaluation must reuse the completed development selection")

    monkeypatch.setattr(module, "fit_probe", forbidden)
    final = bench.run(model, names=["clevr-count"], evaluation="final", data_root=root, output_dir=tmp_path / "final", execution=cpu_execution(workspace))
    assert final.status == "complete"
    assert model.calls == 4  # Only the four final images are newly extracted.
    assert not workspace.exists()
    artifact = read_json(tmp_path / "final" / "results.json")
    assert artifact["tasks"]["clevr-count-v1"]["probe_status"] == "reused"
    assert "pretraining" in artifact["exposure"]


def test_full_search_temporary_estimate_covers_probe_metadata(clevr_data, tmp_path, monkeypatch):
    from copy import deepcopy
    from bench.storage import digest

    root, _ = clevr_data
    protocols = importlib.import_module("bench.protocols")
    recipes = deepcopy(protocols.recipes())
    for name in ("clevr-count", "clevr-distance"):
        recipe = recipes[name]
        recipe["probe"].update(learning_rates=[.001, .003, .01], weight_decays=[0., .1], seeds=[0, 1, 2])
        recipe["digest"] = digest({key: value for key, value in recipe.items() if key != "digest"})
    monkeypatch.setattr(protocols, "recipes", lambda: recipes)
    names = ["clevr-count", "clevr-distance"]
    prepare(names=names, data_root=root)
    output = tmp_path / "full-search"
    bench.run(GlobalAdapter(), names=names, data_root=root, output_dir=output, execution=cpu_execution())
    result = read_json(output / "results.json")
    assert all(len(task["selection"]["candidates"]) == 6 and len(task["selection"]["seeds"]) == 3 for task in result["tasks"].values())
    assert 0 < result["storage"]["peak_temporary_bytes"] <= result["storage"]["estimated_peak_temporary_bytes"]
    assert result["storage"]["retained_temporary_bytes"] == result["storage"]["retained_workspace_bytes"] == 0


def test_partial_metrics_survive_later_failure_and_retry(clevr_data, tmp_path, monkeypatch):
    root, _ = clevr_data
    prepare(names=["clevr-count", "clevr-distance"], data_root=root)
    module = importlib.import_module("bench.runner")
    evaluate = module.evaluate_selection
    output = tmp_path / "result"

    def fail_second(selection, directory, bank, rows, source, task, execution):
        if task["name"] == "clevr-distance":
            current = read_json(output / "results.json")
            assert current["status"] == "running"
            assert current["tasks"]["clevr-count-v1"]["status"] == "complete"
            raise RuntimeError("second task failed")
        return evaluate(selection, directory, bank, rows, source, task, execution)

    monkeypatch.setattr(module, "evaluate_selection", fail_second)
    model = GlobalAdapter()
    with pytest.raises(RuntimeError, match="second task failed"):
        bench.run(model, names=["clevr-count", "clevr-distance"], data_root=root, output_dir=output, execution=cpu_execution())
    artifact = read_json(output / "results.json")
    assert artifact["status"] == "failed"
    assert artifact["tasks"]["clevr-count-v1"]["status"] == "complete"
    count_statistics = (output / "statistics" / "clevr-count-v1.json").read_bytes()
    monkeypatch.setattr(module, "evaluate_selection", evaluate)
    bench.run(model, names=["clevr-count", "clevr-distance"], data_root=root, output_dir=output, execution=cpu_execution())
    assert (output / "statistics" / "clevr-count-v1.json").read_bytes() == count_statistics
    assert not (output / ".work").exists()


def test_unsupported_tasks_fail_before_any_extraction(tmp_path):
    model = GlobalAdapter()
    with pytest.raises(ValueError, match="lacks the requested output"):
        bench.run(model, names=["pascal"], data_root=tmp_path, output_dir=tmp_path / "out", execution=cpu_execution())
    assert model.calls == 0


def test_probe_training_inside_caller_inference_context(clevr_data, tmp_path):
    root, _ = clevr_data
    prepare(names=["clevr-count"], data_root=root)
    model = GlobalAdapter()
    precision = torch.get_float32_matmul_precision()
    with torch.inference_mode():
        result = bench.run(model, names=["clevr-count"], data_root=root, output_dir=tmp_path / "result", execution=cpu_execution())
        assert torch.is_inference_mode_enabled()
        assert result.status == "complete"
    assert torch.get_float32_matmul_precision() == precision


def test_retry_finishes_cleanup_without_extraction(clevr_data, tmp_path, monkeypatch):
    root, _ = clevr_data
    prepare(names=["clevr-count"], data_root=root)
    module = importlib.import_module("bench.runner")
    remove = module.shutil.rmtree
    failures = []

    def fail_once(path, *args, **kwargs):
        if Path(path).parent.name == "banks" and not failures:
            failures.append(path)
            raise OSError("injected cleanup failure")
        return remove(path, *args, **kwargs)

    monkeypatch.setattr(module.shutil, "rmtree", fail_once)
    output = tmp_path / "result"
    model = GlobalAdapter()
    with pytest.raises(OSError, match="cleanup failure"):
        bench.run(model, names=["clevr-count"], data_root=root, output_dir=output, execution=cpu_execution())
    assert read_json(output / "results.json")["tasks"]["clevr-count-v1"]["status"] == "complete"
    calls = model.calls
    result = bench.run(model, names=["clevr-count"], data_root=root, output_dir=output, execution=cpu_execution())
    assert result.status == "complete" and model.calls == calls
    assert not (output / ".work").exists()


def test_all_input_shapes_are_checked_before_fitting(clevr_data, tmp_path, monkeypatch):
    from bench.protocols import recipes
    from bench.storage import digest

    root, _ = clevr_data
    tasks = recipes()
    tasks["clevr-distance"]["canvas"] = 112
    tasks["clevr-distance"]["digest"] = digest({key: value for key, value in tasks["clevr-distance"].items() if key != "digest"})
    prepare(names=["clevr-count", "clevr-distance"], data_root=root)
    model = GlobalAdapter()
    encode = model.encode_global

    def fixed_shape(batch):
        if batch.images.shape[-1] != 224:
            raise ValueError("unsupported canvas")
        return encode(batch)

    model.encode_global = fixed_shape

    def forbidden(*args, **kwargs):
        raise AssertionError("No probe may fit before every requested input shape passes")

    monkeypatch.setattr("bench.runner.fit_probe", forbidden)
    with pytest.raises(ValueError, match="unsupported canvas"):
        bench.run(model, names=["clevr-count", "clevr-distance"], data_root=root, output_dir=tmp_path / "result", execution=cpu_execution())
    assert model.calls == 1


def test_development_does_not_read_final_sources(clevr_data, tmp_path):
    root, source = clevr_data
    prepare(names=["clevr-count"], data_root=root)
    for image in (source / "images" / "val").iterdir():
        image.unlink()
    result = bench.run(GlobalAdapter(), names=["clevr-count"], data_root=root, output_dir=tmp_path / "result", execution=cpu_execution())
    assert result.status == "complete"


@pytest.mark.parametrize("location", ["output", "workspace", "preparation"])
def test_concurrent_process_owner_refuses_mutation(clevr_data, tmp_path, location):
    root, _ = clevr_data
    prepare(names=["clevr-count"], data_root=root)
    output, workspace = tmp_path / "result", tmp_path / "workspace"
    paths = {"output": tmp_path / ".result.bench-output-lock", "workspace": tmp_path / ".workspace.bench-lock", "preparation": root / ".bench" / "locks" / "clevr.lock"}
    script = "import sys\nfrom bench.storage import lock\nwith lock(sys.argv[1]):\n print('locked', flush=True)\n sys.stdin.readline()\n"
    process = subprocess.Popen([sys.executable, "-c", script, str(paths[location])], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert process.stdout.readline().strip() == "locked"
        if location == "preparation":
            report = prepare(names=["clevr-count"], data_root=root)
            assert not report["ready"] and "Busy" in report["tasks"]["clevr-count-v1"]["error"]
        else:
            model = GlobalAdapter()
            with pytest.raises(RuntimeError, match="Busy"):
                bench.run(model, names=["clevr-count"], data_root=root, output_dir=output, execution=cpu_execution(workspace))
            assert model.calls == 0 and not (output / "results.json").exists()
    finally:
        process.communicate("\n", timeout=5)
    assert process.returncode == 0


def test_interrupted_probe_reuses_completed_candidates_and_verifies_selection(clevr_data, tmp_path, monkeypatch):
    from bench.probes import read_selection

    root, _ = clevr_data
    prepare(names=["clevr-count"], data_root=root)
    module = importlib.import_module("bench.probes")
    train = module.train_candidate
    calls = []

    def interrupt_second(*args, **kwargs):
        calls.append((args[7], args[8]))
        if len(calls) == 2:
            raise RuntimeError("interrupted second candidate")
        return train(*args, **kwargs)

    monkeypatch.setattr(module, "train_candidate", interrupt_second)
    model = GlobalAdapter()
    workspace, output = tmp_path / "work", tmp_path / "result"
    arguments = dict(names=["clevr-count"], data_root=root, output_dir=output, execution=cpu_execution(workspace))
    with pytest.raises(RuntimeError, match="interrupted second candidate"):
        bench.run(model, **arguments)
    result = bench.run(model, **arguments)
    assert result.status == "complete" and calls.count((0, 0.01)) == 1
    selection_path = next((workspace / "probes").glob("*/selection.json"))
    selection = read_selection(selection_path.parent)
    assert selection is not None
    assert len(list(selection_path.parent.glob("*.pt"))) == 3  # Two selected heads and normalization.
    selection["seeds"][0]["epoch"] += 1
    write_json(selection_path, selection)
    assert read_selection(selection_path.parent) is None
