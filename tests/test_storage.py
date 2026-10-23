from pathlib import Path

import pytest
import torch

from bench.features import save_features, save_tensors
from bench.storage import TemporaryStorage, atomic_write, bytes_used, remove, write_json


def test_storage_high_water_includes_atomic_copies_and_failed_replacements(tmp_path, monkeypatch):
    import bench.storage as module

    work = tmp_path / "work"
    head = work / "probes" / "head.pt"
    unrelated = work / "checkpoint.pt"
    atomic_write(unrelated, b"caller checkpoint" * 1000)
    atomic_write(head, b"old head")
    storage = TemporaryStorage(work)
    observed = [len(b"old head")]
    replace = module.os.replace

    def inspect(temporary, destination):
        if Path(destination).is_relative_to(work / "probes") or Path(destination).is_relative_to(work / "banks"):
            observed.append(bytes_used(work / "probes") + bytes_used(work / "banks"))
        return replace(temporary, destination)

    monkeypatch.setattr(module.os, "replace", inspect)
    save_tensors(head, {"weight": torch.ones(30, 10)}, storage=storage)
    save_tensors(head, {"weight": torch.ones(40, 20)}, storage=storage)
    save_features(work / "banks" / "shard.safetensors", {"values": torch.ones(128, 512), "costs": {}}, storage=storage)
    write_json(work / "banks" / "manifest.json", {"samples": ["a", "b"]}, storage=storage)
    assert storage.peak == max(observed)
    assert storage.current == bytes_used(work / "probes") + bytes_used(work / "banks")
    previous_bytes, previous_current = head.read_bytes(), storage.current

    def fail(temporary, destination):
        observed.append(bytes_used(work / "probes") + bytes_used(work / "banks"))
        raise OSError("interrupted replacement")

    monkeypatch.setattr(module.os, "replace", fail)
    with pytest.raises(OSError, match="interrupted replacement"):
        atomic_write(head, b"new head" * 10000, storage=storage)
    assert head.read_bytes() == previous_bytes
    assert storage.current == previous_current and storage.peak == max(observed)
    remove(work / "banks", storage=storage)
    remove(work / "probes", storage=storage)
    assert storage.current == 0 and unrelated.exists()


def test_failed_feature_write_preserves_committed_shard_and_accounts_temporary_bytes(tmp_path, monkeypatch):
    import bench.features as features

    path = tmp_path / "banks" / "shard.safetensors"
    storage = TemporaryStorage(tmp_path)
    tensors = {"values": torch.ones(2, 3), "costs": {}}
    save_features(path, tensors, storage=storage)
    previous = path.read_bytes()

    def fail(tensors, temporary):
        Path(temporary).write_bytes(b"partial tensor file")
        raise OSError("interrupted feature write")

    monkeypatch.setattr(features, "save_file", fail)
    with pytest.raises(OSError, match="interrupted feature write"):
        save_features(path, tensors, storage=storage)
    assert path.read_bytes() == previous
    assert storage.current == len(previous)
    assert storage.peak == len(previous) + len(b"partial tensor file")
    assert not list(path.parent.glob(".bench-tmp-*"))
