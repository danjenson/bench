import subprocess
import sys

import pytest

from bench.workspace import initialize, prune, workspace_lock


def test_pruning_is_owned_idempotent_and_preserves_unrelated_files(tmp_path):
    workspace = tmp_path / "work"
    initialize(workspace, "model-a")
    (workspace / "banks").mkdir()
    (workspace / "banks" / "features.pt").write_text("features")
    (workspace / "checkpoint.pt").write_text("caller checkpoint")
    prune(work_dir=workspace)
    assert not (workspace / "banks").exists()
    assert (workspace / "checkpoint.pt").read_text() == "caller checkpoint"
    prune(work_dir=workspace)
    assert (workspace / "checkpoint.pt").exists()
    unowned = tmp_path / "unowned"
    unowned.mkdir()
    (unowned / "checkpoint.pt").write_text("caller checkpoint")
    with pytest.raises(ValueError, match="unowned"):
        prune(work_dir=unowned)
    empty = tmp_path / "empty"
    prune(work_dir=empty)
    prune(work_dir=empty)


def test_new_model_replaces_old_state_and_active_prune_fails(tmp_path):
    workspace = tmp_path / "work"
    initialize(workspace, "model-a")
    (workspace / "probes").mkdir()
    (workspace / "probes" / "winner.pt").write_text("old head")
    with workspace_lock(workspace):
        with pytest.raises(RuntimeError, match="Busy"):
            prune(work_dir=workspace)
    initialize(workspace, "model-b")
    assert not (workspace / "probes").exists()
    prune(work_dir=workspace)
    assert not workspace.exists()


def test_hard_interruption_discards_only_uncommitted_temporary_writes(tmp_path):
    from bench.storage import atomic_write

    workspace = tmp_path / "work"
    initialize(workspace, "model-a")
    selected = workspace / "probes" / "selected.pt"
    atomic_write(selected, b"verified previous selection")
    script = "import os,sys\nfrom bench.storage import atomic_file\nwith atomic_file(sys.argv[1]) as stream:\n stream.write(b'incomplete replacement')\n stream.flush()\n os._exit(17)\n"
    result = subprocess.run([sys.executable, "-c", script, str(selected)], check=False)
    assert result.returncode == 17
    assert list((workspace / "probes").glob(".bench-tmp-*"))
    initialize(workspace, "model-a")
    assert selected.read_bytes() == b"verified previous selection"
    assert not list((workspace / "probes").glob(".bench-tmp-*"))
