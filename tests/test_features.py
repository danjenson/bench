import random
from collections import namedtuple
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import torch

from bench import Execution, Extraction, ImageBatch, ModelInfo, SpatialFeatures, VisionModel
from bench.features import build_bank, extract, feature_diagnostics, model_identity, numerical_mode, validate_execution
from bench.prepare import load_prepared, prepare
from bench.protocols import resolve_names
from bench.storage import read_json


class GlobalAdapter(VisionModel):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 5)
        self.info = ModelInfo("fixture", "test-1", {"global": "channel-linear"}, {"normalization": "none", "seed": 0})
        self.calls = 0
        self.fail = False

    def encode_global(self, batch):
        self.calls += 1
        # The runner restores ambient RNG even when a faulty adapter consumes it.
        torch.rand(2)
        random.random()
        np.random.rand()
        if self.fail:
            raise RuntimeError("adapter failure")
        return Extraction(self.layer(batch.images.mean((2, 3))), {"encoder_passes_count": torch.ones(len(batch.sample_ids))})


def image_batch():
    return ImageBatch(torch.ones(2, 3, 8, 8), torch.ones(2, 1, 8, 8, dtype=torch.bool), torch.eye(3).repeat(2, 1, 1), ("opaque-0", "opaque-1"))


@pytest.fixture
def cuda_devices(monkeypatch):
    state = {"current": 0, "bf16": [False, True]}

    @contextmanager
    def select(device):
        previous = state["current"]
        index = torch.device(device).index
        state["current"] = previous if index is None else index
        try:
            yield
        finally:
            state["current"] = previous

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: state["current"])
    monkeypatch.setattr(torch.cuda, "device", select)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: state["bf16"][state["current"]])
    return state


@pytest.mark.parametrize("supported", [False, True])
def test_bf16_validation_checks_requested_gpu_and_restores_device(cuda_devices, supported):
    cuda_devices["bf16"] = [not supported, supported]
    execution = Execution(device="cuda:1")
    if supported:
        validate_execution(execution)
    else:
        with pytest.raises(ValueError, match="bf16-v1 requires"):
            validate_execution(execution)
    assert cuda_devices["current"] == 0


@pytest.mark.parametrize("fail", [False, True])
def test_numerical_mode_uses_requested_gpu_and_restores_device(cuda_devices, fail):
    try:
        with numerical_mode(Execution(device="cuda:1")):
            assert torch.cuda.current_device() == 1
            # Autocast also checks the current device when enabling BF16.
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if fail:
                    raise RuntimeError("evaluation failed")
    except RuntimeError as error:
        assert fail and str(error) == "evaluation failed"
    assert cuda_devices["current"] == 0


def test_implicit_cuda_device_rejects_model_on_another_gpu(cuda_devices, monkeypatch):
    model = GlobalAdapter()
    parameter = namedtuple("parameter", "device")(torch.device("cuda:1"))
    monkeypatch.setattr(model, "named_parameters", lambda: [("weight", parameter)])
    with pytest.raises(ValueError, match="caller must place the model on cuda:0"):
        model_identity(model, Execution(device="cuda"))


def test_adapter_state_rng_and_flags_survive_success_and_failure():
    model = GlobalAdapter()
    model.train()
    model.layer.eval()
    model.layer.weight.grad = torch.ones_like(model.layer.weight)
    state = {key: value.clone() for key, value in model.state_dict().items()}
    gradient = model.layer.weight.grad.clone()
    execution = Execution(device="cpu", numerics="fp32-v1", workers=0)
    for fail in (False, True):
        model.fail = fail
        torch_before = torch.random.get_rng_state().clone()
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        if fail:
            with pytest.raises(RuntimeError, match="adapter failure"):
                extract(model, image_batch(), "global", execution)
        else:
            result = extract(model, image_batch(), "global", execution)
            assert not result["values"].is_inference()
            head = torch.zeros(5, 2, requires_grad=True)
            (result["values"] @ head).sum().backward()
            assert head.grad is not None
        assert model.training and not model.layer.training
        assert torch.equal(torch_before, torch.random.get_rng_state())
        assert random.getstate() == python_before
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
        assert torch.equal(model.layer.weight.grad, gradient)
        for key, value in state.items():
            assert torch.equal(model.state_dict()[key], value)


def test_spatial_extent_accepts_inset_centers_and_rejects_crop():
    class SpatialAdapter(VisionModel):
        def encode_spatial(self, batch):
            matrix = torch.tensor([[0.8, 0, 0.1], [0, 0.8, 0.1], [0, 0, 1]]).repeat(2, 1, 1)
            return Extraction(SpatialFeatures(torch.ones(2, 7, 2, 2), matrix, torch.tensor(self.extent).repeat(2, 1)))

    model = SpatialAdapter()
    model.extent = [0., 0., 1., 1.]
    execution = Execution(device="cpu", numerics="fp32-v1")
    result = extract(model, image_batch(), "spatial", execution)
    assert result["values"].shape == (2, 7, 2, 2)
    model.extent = [0.1, 0.1, 0.9, 0.9]
    with pytest.raises(ValueError, match="does not cover"):
        extract(model, image_batch(), "spatial", execution)


def test_bank_reuse_resumption_and_recipe_invalidation(clevr_data, tmp_path, monkeypatch):
    root, _ = clevr_data
    prepare(names=["clevr-count"], data_root=root)
    task = resolve_names(["clevr-count"])[0]
    manifest, rows = load_prepared(root, task, "train")
    model = GlobalAdapter()
    execution = Execution(device="cpu", numerics="fp32-v1", batch_size=2, workers=0)
    metadata = model_identity(model, execution)
    arguments = dict(model_metadata=metadata, task=task, root=manifest["source_root"], rows=rows, work_dir=tmp_path / "work", execution=execution)
    bank, status = build_bank(model, **arguments)
    expected = [bank[row["id"]]["values"].clone() for row in rows]
    assert status["extracted_examples"] == len(rows)
    assert status["timing"]["first_batch_examples"] + status["timing"]["steady_state_examples"] == len(rows)
    calls = model.calls

    def forbidden(*args, **kwargs):
        raise AssertionError("Verified reuse must not construct a data loader")

    with monkeypatch.context() as patch:
        patch.setattr("bench.features.DataLoader", forbidden)
        reused, status = build_bank(model, **arguments)
    assert model.calls == calls
    assert status["reused_examples"] == len(rows)
    assert status["timing"]["first_batch_examples"] == status["timing"]["steady_state_examples"] == 0
    assert feature_diagnostics(reused, [row["id"] for row in rows])["sample_count"] == len(rows)
    # A corrupted final shard is regenerated; valid preceding shards survive.
    last = bank.manifest["shards"][-1]
    (bank.directory / last["file"]).write_bytes(b"interrupted")
    resumed, status = build_bank(model, **arguments)
    assert status["extracted_examples"] == len(last["ids"])
    for row, original in zip(rows, expected):
        torch.testing.assert_close(resumed[row["id"]]["values"], original)
    model.info = ModelInfo("fixture", "test-1", {"global": "channel-linear"}, {"seed": 1})
    assert model_identity(model, execution)["digest"] != metadata["digest"]
    model.info = ModelInfo("fixture", "test-1", {"global": "channel-linear", "spatial": "other"}, {"normalization": "none", "seed": 0})
    assert model_identity(model, execution)["digest"] != metadata["digest"]


def test_overlapping_banks_and_different_labels_extract_each_image_once(clevr_data, tmp_path):
    root, _ = clevr_data
    prepare(names=["clevr-count"], data_root=root)
    task = resolve_names(["clevr-count"])[0]
    manifest, rows = load_prepared(root, task, "train")
    model = GlobalAdapter()
    execution = Execution(device="cpu", numerics="fp32-v1", batch_size=2, workers=0)
    arguments = dict(model_metadata=model_identity(model, execution), task=task, root=manifest["source_root"], work_dir=tmp_path / "work", execution=execution)
    first, _ = build_bank(model, rows=rows[:6], **arguments)
    altered_labels = [{**row, "target": 999} for row in rows[3:]]
    second, usage = build_bank(model, rows=altered_labels, **arguments)
    assert first.directory == second.directory
    assert usage["extracted_examples"] == 3 and usage["reused_examples"] == 3
    assert model.calls == 5


@pytest.mark.parametrize("interruption", ["shard", "manifest"])
def test_shard_manifest_commits_resume_every_verified_shard_with_linear_writes(clevr_data, tmp_path, monkeypatch, interruption):
    import bench.features as features

    root, _ = clevr_data
    prepare(names=["clevr-count"], data_root=root)
    task = resolve_names(["clevr-count"])[0]
    manifest, original = load_prepared(root, task, "train")
    rows = [{**original[index % len(original)], "id": f"fixture-row-{index}"} for index in range(67)]
    task = {**task, "canvas": 16}
    model = GlobalAdapter()
    execution = Execution(device="cpu", numerics="fp32-v1", batch_size=1, workers=0)
    arguments = dict(model_metadata=model_identity(model, execution), task=task, root=manifest["source_root"], rows=rows, execution=execution)
    reference, _ = build_bank(model, work_dir=tmp_path / "reference", **arguments)
    work = tmp_path / "interrupted"
    written_bytes = []
    write_json, save_features = features.write_json, features.save_features

    def record_commit(path, value, **kwargs):
        write_json(path, value, **kwargs)
        written_bytes.append(Path(path).stat().st_size)

    saved = 0

    def interrupt(path, value, **kwargs):
        nonlocal saved
        if interruption == "shard" and saved == 48:
            raise RuntimeError("interrupted extraction")
        save_features(path, value, **kwargs)
        saved += 1

    def interrupt_commit(path, value, **kwargs):
        if interruption == "manifest" and saved == 49:
            raise RuntimeError("interrupted extraction")
        record_commit(path, value, **kwargs)

    monkeypatch.setattr(features, "write_json", record_commit)
    with monkeypatch.context() as patch:
        patch.setattr(features, "save_features", interrupt)
        patch.setattr(features, "write_json", interrupt_commit)
        with pytest.raises(RuntimeError, match="interrupted extraction"):
            build_bank(model, work_dir=work, **arguments)
    path = next(work.glob("banks/*/manifest.json"))
    resumed, usage = build_bank(model, work_dir=work, **arguments)
    assert usage["reused_examples"] == 48
    assert usage["extracted_examples"] == len(rows) - 48
    # Allow ample per-entry metadata, but reject rewriting the growing index.
    assert sum(written_bytes) <= 4096 + 512 * len(rows)
    assert read_json(path) == {key: value for key, value in resumed.manifest.items() if key != "shards"}
    assert [read_json(p) for p in sorted(path.parent.glob("shard-*.json"))] == resumed.manifest["shards"]
    assert {p.name for p in path.parent.glob("shard-*.safetensors")} == {shard["file"] for shard in resumed.manifest["shards"]}
    for row in rows:
        assert torch.equal(reference[row["id"]]["values"], resumed[row["id"]]["values"])
        assert reference[row["id"]]["costs"] == resumed[row["id"]]["costs"]
    reference.close()
    resumed.close()


def test_missing_shard_index_entry_recovers_prefix_and_removes_unrequested_tail(clevr_data, tmp_path):
    root, _ = clevr_data
    prepare(names=["clevr-count"], data_root=root)
    task = {**resolve_names(["clevr-count"])[0], "canvas": 16}
    manifest, rows = load_prepared(root, resolve_names(["clevr-count"])[0], "train")
    model = GlobalAdapter()
    execution = Execution(device="cpu", numerics="fp32-v1", batch_size=1, workers=0)
    arguments = dict(model_metadata=model_identity(model, execution), task=task, root=manifest["source_root"], work_dir=tmp_path / "work", execution=execution)
    bank, _ = build_bank(model, rows=rows, **arguments)
    expected = [bank[row["id"]]["values"].clone() for row in rows[:5]]
    bank.close()
    (bank.directory / "shard-000003.json").unlink()
    resumed, usage = build_bank(model, rows=rows[:5], **arguments)
    assert usage["reused_examples"] == 3 and usage["extracted_examples"] == 2
    assert len(list(bank.directory.glob("shard-*.json"))) == 5
    assert len(list(bank.directory.glob("shard-*.safetensors"))) == 5
    for row, values in zip(rows, expected):
        assert torch.equal(resumed[row["id"]]["values"], values)
    resumed.close()


def test_expanding_reused_bank_checks_space_before_extraction(clevr_data, tmp_path, monkeypatch):
    root, _ = clevr_data
    prepare(names=["clevr-count"], data_root=root)
    task = resolve_names(["clevr-count"])[0]
    manifest, rows = load_prepared(root, task, "train")
    model = GlobalAdapter()
    execution = Execution(device="cpu", numerics="fp32-v1", batch_size=2, workers=0)
    arguments = dict(model_metadata=model_identity(model, execution), task=task, root=manifest["source_root"], work_dir=tmp_path / "work", execution=execution)
    build_bank(model, rows=rows[:2], **arguments)
    calls = model.calls
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr("bench.features.shutil.disk_usage", lambda path: usage(100, 100, 0))
    with pytest.raises(OSError, match="Insufficient temporary space"):
        build_bank(model, rows=rows, **arguments)
    assert model.calls == calls


def test_small_global_bank_reuses_shards_across_probe_passes(tmp_path, monkeypatch):
    import bench.features as features

    read_text = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda path, *args, **kwargs: "MemAvailable: 67108864 kB\n" if path == Path("/proc/meminfo") else read_text(path, *args, **kwargs))
    manifest = {"shape": [8], "shards": []}
    for index in range(1030):
        name = f"shard-{index}.safetensors"
        features.save_features(tmp_path / name, {"values": torch.full((1, 8), float(index)), "costs": {}})
        manifest["shards"].append({"file": name, "ids": [str(index)]})
    load = features.load_file
    calls = []

    def counted(*arguments, **keywords):
        calls.append(arguments[0])
        return load(*arguments, **keywords)

    monkeypatch.setattr(features, "load_file", counted)
    bank = features.FeatureBank(tmp_path, manifest)
    order = torch.randperm(1030, generator=torch.Generator().manual_seed(3)).tolist()
    for index in order + order[::-1]:
        torch.testing.assert_close(bank[str(index)]["values"], torch.full((8,), float(index)))
    assert len(calls) == 1030
    bank.close()
    bank["0"]
    assert len(calls) == 1031
    bank.close()


@pytest.mark.parametrize("shape,memory,shard_bytes,expected_loads", [
    ([2], "MemAvailable: 8388608 kB\n", 200 * 1024 ** 2, 2),
    ([2], "MemAvailable: 1048576 kB\n", 200 * 1024 ** 2, 3),
    ([2], "MemAvailable: 67108864 kB\n", 5 * 1024 ** 3 // 2, 3),
    ([2, 1, 1], "MemAvailable: 67108864 kB\n", 200 * 1024 ** 2, 3),
    ([2], "MemTotal: 67108864 kB\n", 200 * 1024 ** 2, 3),
    ([2], None, 200 * 1024 ** 2, 3),
])
def test_mapping_cache_obeys_available_memory_and_preserves_dense_limits(tmp_path, monkeypatch, shape, memory, shard_bytes, expected_loads):
    from types import SimpleNamespace

    import bench.features as features

    read_text, stat = Path.read_text, Path.stat

    def memory_info(path, *args, **kwargs):
        if path != Path("/proc/meminfo"):
            return read_text(path, *args, **kwargs)
        if memory is None:
            raise PermissionError("Memory information unavailable")
        return memory

    # Report realistic shard sizes without allocating gigabytes in a unit test.
    def shard_stat(path, *args, **kwargs):
        if path.parent == tmp_path and path.name in ("a.safetensors", "b.safetensors"):
            return SimpleNamespace(st_size=shard_bytes)
        return stat(path, *args, **kwargs)

    calls = []

    def load(path, *, device):
        calls.append(path)
        return {"values": torch.full((1, *shape), 1.0 if path.stem == "a" else 2.0)}

    monkeypatch.setattr(Path, "read_text", memory_info)
    monkeypatch.setattr(Path, "stat", shard_stat)
    monkeypatch.setattr(features, "load_file", load)
    bank = features.FeatureBank(tmp_path, {"shape": shape, "shards": [{"file": key + ".safetensors", "ids": [key]} for key in ("a", "b")]})
    for key in ("a", "b", "a"):
        torch.testing.assert_close(bank[key]["values"], torch.full(shape, 1.0 if key == "a" else 2.0), rtol=0, atol=0)
    assert len(calls) == expected_loads
    bank.close()
    bank["a"]
    assert len(calls) == expected_loads + 1
    bank.close()
    expected = torch.stack([torch.full(shape, value) for value in (1.0, 2.0, 1.0)])
    torch.testing.assert_close(bank.read_values(["a", "b", "a"]), expected, rtol=0, atol=0)
    assert len(calls) == 2 * expected_loads + 1
    bank.close()
    bank.read_values(["a"])
    assert len(calls) == 2 * expected_loads + 2
    bank.close()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("spatial", [False, True])
def test_feature_shards_preserve_values_geometry_costs_and_dtype(tmp_path, dtype, spatial):
    from bench.features import FeatureBank, save_features

    values = torch.arange(120, dtype=dtype).reshape(2, 3, 4, 5)
    tensors = {"values": values.transpose(-1, -2) if spatial else values.reshape(2, 60)[:, ::2], "costs": {"encoder_passes_count": torch.tensor([2, 3]), "observed_area_fraction": torch.tensor([0.5, 1.0])}}
    if spatial:
        tensors.update(grid_to_original=torch.eye(3).repeat(2, 1, 1), extent_in_original=torch.tensor([[0., 0., 1., 1.], [0.1, 0.2, 0.8, 0.9]]))
    path = tmp_path / "shard.safetensors"
    save_features(path, tensors)
    bank = FeatureBank(tmp_path, {"shape": list(tensors["values"].shape[1:]), "shards": [{"file": path.name, "ids": ["a", "b"]}]})
    for index, key in enumerate(["a", "b"]):
        restored = bank[key]
        assert restored.keys() == tensors.keys()
        for name in tensors:
            if name == "costs":
                assert restored[name].keys() == tensors[name].keys()
                for cost in tensors[name]:
                    torch.testing.assert_close(restored[name][cost], tensors[name][cost][index], rtol=0, atol=0, check_dtype=True)
            else:
                torch.testing.assert_close(restored[name], tensors[name][index], rtol=0, atol=0, check_dtype=True)
    torch.testing.assert_close(bank.read_values(["b", "a", "b"]), tensors["values"][[1, 0, 1]], rtol=0, atol=0, check_dtype=True)
    bank.close()
