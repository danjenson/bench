"""One-model temporary workspaces. Deletion is confined to owned state."""

from .storage import lock, read_json, remove, resolve_path, write_json


def workspace_lock(path):
    path = resolve_path(path)
    return lock(path.parent / f".{path.name}.bench-lock")


def initialize(path, model_digest, run_id=None):
    path = resolve_path(path)
    marker = path / "owner.json"
    if path.exists() and any(path.iterdir()) and not marker.exists():
        raise ValueError(f"Refusing nonempty unowned workspace: {path}; choose an empty work_dir")
    if marker.exists():
        previous = read_json(marker)
        if previous.get("owner") != "bench-workspace-v1":
            raise ValueError(f"Unrecognized workspace ownership: {path}")
        if previous["model_digest"] != model_digest:
            clear(path)
    path.mkdir(parents=True, exist_ok=True)
    for name in ("banks", "probes"):
        child = path / name
        if child.is_symlink():
            raise ValueError(f"Workspace-owned directory cannot be a symlink: {child}")
        for temporary in child.rglob(".bench-tmp-*"):
            if temporary.is_file() or temporary.is_symlink():
                temporary.unlink()
    write_json(marker, {"owner": "bench-workspace-v1", "model_digest": model_digest, "run_id": run_id})
    return path


def clear(path, *, storage=None):
    path = resolve_path(path)
    marker = path / "owner.json"
    if not path.exists():
        return
    if not marker.exists():
        if not any(path.iterdir()):
            path.rmdir()
            return
        raise ValueError(f"Refusing to prune an unowned workspace: {path}")
    if read_json(marker).get("owner") != "bench-workspace-v1":
        raise ValueError(f"Unrecognized workspace ownership: {path}")
    for name in ("banks", "probes"):
        child = path / name
        remove(child, storage=storage)
    if any(child != marker for child in path.iterdir()):
        write_json(marker, {"owner": "bench-workspace-v1", "model_digest": None, "run_id": None})
    else:
        marker.unlink()
        path.rmdir()


def prune(*, work_dir):
    path = resolve_path(work_dir)
    with workspace_lock(path):
        clear(path)
    print(f"Pruned bench-owned temporary state: {path}")
