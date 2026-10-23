"""Source acquisition; never imports a model or initializes a compute runtime."""

from importlib.resources import files
import os
from pathlib import Path
import shutil
import stat
import tarfile
import urllib.error
import urllib.request
import zipfile

from omegaconf import OmegaConf

from .storage import contained_path, file_digest, read_json, write_json


def source_recipes():
    return OmegaConf.to_container(OmegaConf.load(files("bench.cli.configs").joinpath("sources.yaml")), resolve=True)


def download(url, destination, checksum):
    """Resume only against the same server entity, then verify the whole file."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    metadata_path = partial.with_suffix(partial.suffix + ".json")
    metadata = read_json(metadata_path) if metadata_path.exists() else {}
    offset = partial.stat().st_size if partial.exists() else 0
    validator = metadata.get("etag") or metadata.get("last_modified")
    headers = {"Range": f"bytes={offset}-", "If-Range": validator} if offset and validator and metadata.get("url") == url else {}
    try:
        response = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60)
    except urllib.error.HTTPError as error:
        if error.code != 416:
            raise
        # A completed download may have been interrupted before its rename.
        if partial.exists() and file_digest(partial, checksum["algorithm"]) == checksum["value"]:
            os.replace(partial, destination)
            metadata_path.unlink(missing_ok=True)
            return
        response = urllib.request.urlopen(url, timeout=60)
        headers = {}
    with response:
        resumed = response.status == 206 and bool(headers)
        if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError(f"Invalid partial response while downloading {url}")
        write_json(metadata_path, {"url": url, "etag": response.headers.get("ETag"), "last_modified": response.headers.get("Last-Modified")})
        with partial.open("ab" if resumed else "wb") as stream:
            shutil.copyfileobj(response, stream, length=1024 * 1024)
            stream.flush()
            os.fsync(stream.fileno())
    if file_digest(partial, checksum["algorithm"]) != checksum["value"]:
        partial.unlink()
        metadata_path.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch for {url}; corrupt bench-owned download removed; rerun prepare")
    os.replace(partial, destination)
    metadata_path.unlink(missing_ok=True)


def extract_archive(archive, destination):
    """Extract regular files/directories only; reject links and traversal."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist():
                target = contained_path(destination, member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
                    raise ValueError(f"Unsupported archive member: {member.filename}")
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(member) as incoming, target.open("wb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
    else:
        with tarfile.open(archive) as source:
            for member in source:
                target = contained_path(destination, member.name)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as incoming, target.open("wb") as outgoing:
                        shutil.copyfileobj(incoming, outgoing)
                else:
                    raise ValueError(f"Unsupported archive member: {member.name}")


def source_path(root, name, recipe, configured=None):
    choices = [Path(configured).expanduser().resolve()] if configured else [root / layout for layout in recipe["layouts"]]
    for path in choices:
        if all((path / item).exists() for item in recipe["required"]):
            return path
    return choices[0]


def acquire(root, name, recipe, configured=None, *, validate=None):
    target = source_path(root, name, recipe, configured)
    if all((target / item).exists() for item in recipe["required"]):
        return target
    layout = ", ".join(recipe["required"])
    if not recipe.get("checksum") or not recipe.get("url"):
        raise FileNotFoundError(f"{name}: expected {target} containing {layout}. {recipe['manual']} Official route: {recipe['official_page']}")
    print(f"{name}: expected download {recipe['download_bytes']:,} bytes; peak preparation {recipe['peak_bytes']:,} bytes", flush=True)
    if shutil.disk_usage(root).free < recipe["peak_bytes"]:
        raise OSError(f"Insufficient space at {root}; need approximately {recipe['peak_bytes']:,} bytes")
    staging = root / ".bench" / "staging" / name
    staging.mkdir(parents=True, exist_ok=True)
    # Only the archive within staging is bench-owned; user archives survive.
    candidates = [root / recipe["archive"], target.parent / recipe["archive"], target.parent.parent / recipe["archive"]]
    archive = next((path for path in candidates if path.is_file()), staging / recipe["archive"])
    if not archive.exists():
        download(recipe["url"], archive, recipe["checksum"])
    checksum = recipe["checksum"]
    if file_digest(archive, checksum["algorithm"]) != checksum["value"]:
        raise ValueError(f"Checksum mismatch: {archive}. Expected {checksum['algorithm']}:{checksum['value']}")
    unpacked = staging / "extracted"
    if unpacked.exists():
        shutil.rmtree(unpacked)
    extract_archive(archive, unpacked)
    extracted = unpacked / recipe["archive_root"]
    if not all((extracted / item).exists() for item in recipe["required"]):
        raise ValueError(f"Incomplete archive layout: {archive}; expected {layout}")
    if validate is not None:
        validate(extracted)
    if target.exists():
        raise FileExistsError(f"Incomplete destination already exists: {target}; repair it or choose a new per-dataset path")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.replace(extracted, target)
    shutil.rmtree(unpacked)
    if archive.parent == staging:
        archive.unlink()
    return target
