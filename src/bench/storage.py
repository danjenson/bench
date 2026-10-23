"""Atomic local artifacts, content identities, and process locks."""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def resolve_path(path):
    return Path(path).expanduser().resolve()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path, algorithm="sha256"):
    hasher = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def bytes_used(path):
    path = Path(path)
    if path.is_file() or path.is_symlink():
        return path.lstat().st_size
    return sum(file.lstat().st_size for file in path.rglob("*") if file.is_file() or file.is_symlink()) if path.exists() else 0


class TemporaryStorage:
    """Logical file-byte high water, including atomic replacements, under one lock."""

    def __init__(self, workspace):
        self.roots = [Path(workspace) / name for name in ("banks", "probes")]
        self.current = sum(bytes_used(path) for path in self.roots)
        self.peak = self.current

    def contains(self, path):
        return any(Path(path).is_relative_to(root) for root in self.roots)


@contextmanager
def atomic_path(path, *, storage=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".bench-tmp-", dir=path.parent)
    os.close(fd)
    tracked = storage is not None and storage.contains(path)
    try:
        yield Path(temporary)
        with Path(temporary).open("rb") as stream:
            os.fsync(stream.fileno())
        new_size = Path(temporary).stat().st_size
        if tracked:
            storage.peak = max(storage.peak, storage.current + new_size)
            old_size = bytes_used(path)
        os.replace(temporary, path)
        if tracked:
            storage.current += new_size - old_size
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if tracked and Path(temporary).exists():
            storage.peak = max(storage.peak, storage.current + Path(temporary).stat().st_size)
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def atomic_file(path, *, storage=None):
    with atomic_path(path, storage=storage) as temporary:
        with temporary.open("wb") as stream:
            yield stream


def atomic_write(path, data, *, storage=None):
    with atomic_file(path, storage=storage) as stream:
        stream.write(data)


def write_json(path, data, *, storage=None):
    atomic_write(path, (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(), storage=storage)


def remove(path, *, storage=None):
    path = Path(path)
    tracked = storage is not None and storage.contains(path)
    before = bytes_used(path) if tracked else 0
    try:
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.exists():
            shutil.rmtree(path)
    finally:
        if tracked:
            storage.current -= before - bytes_used(path)


@contextmanager
def lock(path):
    """The lock inode is persistent: deleting it would allow two owners."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Busy: {path}. Retry when its current bench operation finishes.") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def contained_path(root, relative):
    root = Path(root).resolve()
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Path escapes dataset root: {relative}")
    return path
