"""Deterministic source indexing and atomic task sample manifests."""

from collections import defaultdict
from pathlib import Path
import time

from PIL import Image

from .acquisition import acquire, source_recipes
from .protocols import prepare_command, resolve_names
from .storage import contained_path, digest, file_digest, lock, read_json, resolve_path, write_json
from .tasks.classification import index_clevr, index_imagenet
from .tasks.correspondence import distinct_images, index_spair
from .tasks.segmentation import index_segmentation, read_mask


def split_samples(partitions, task):
    if "development" in partitions:
        return {split: sorted(rows, key=lambda row: row["id"]) for split, rows in partitions.items()}
    groups = defaultdict(list)
    stratified = task["source"] == "imagenet"
    for row in partitions["train"]:
        groups[row["target"] if stratified else None].append(row)
    bank = task.get("bank", {})
    training, development = [], []
    for rows in groups.values():
        rows.sort(key=lambda row: digest([task["split_seed"], row["id"]]))
        dev_count = bank.get("development_per_class" if stratified else "development", max(1, round(len(rows) * task["development_fraction"])))
        train_count = bank.get("train_per_class" if stratified else "train", len(rows) - dev_count)
        if train_count < 1 or len(rows) < train_count + dev_count:
            raise ValueError(f"Insufficient training images for fixed bank in {task['id']}: need {train_count + dev_count}, found {len(rows)}")
        development.extend(rows[:dev_count])
        training.extend(rows[dev_count:dev_count + train_count])
    final_groups = defaultdict(list)
    for row in partitions["final"]:
        final_groups[row["target"] if stratified else None].append(row)
    final = []
    for rows in final_groups.values():
        rows.sort(key=lambda row: digest([task["split_seed"], row["id"]]))
        count = bank.get("final_per_class" if stratified else "final", len(rows))
        if len(rows) < count:
            raise ValueError(f"Insufficient final images for fixed bank in {task['id']}: need {count}, found {len(rows)}")
        final.extend(rows[:count])
    return {split: sorted(rows, key=lambda row: row["id"]) for split, rows in [("train", training), ("development", development), ("final", final)]}


def index_source(root, task):
    if task["source"] == "imagenet":
        return index_imagenet(root, task)
    if task["source"] == "clevr":
        return index_clevr(root, task)
    if task["source"] == "spair71k":
        return index_spair(root, task)
    return index_segmentation(root, task)


def verify_file(root, relative, inventory, kind=None, task=None):
    path = contained_path(root, relative)
    stat = path.stat()
    stamp = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
    previous = inventory.get(relative, {})
    if any(previous.get(key) != value for key, value in stamp.items()):
        if kind == "image":
            with Image.open(path) as image:
                image.verify()
        elif kind == "mask":
            read_mask(path, task)
        inventory[relative] = {**stamp, "sha256": file_digest(path)}
    return inventory[relative]["sha256"]


def validated_partitions(source_root, task, source, inventory):
    partitions, labels = index_source(source_root, task)
    for split, rows in partitions.items():
        expected = source["counts"].get(split)
        if task["name"] != "imagenet-100" and expected is not None and len(rows) != expected:
            raise ValueError(f"Incomplete {task['source']} {split}: expected {expected} examples, found {len(rows)} at {source_root}")
        identifiers = [row["id"] for row in rows]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"Duplicate source IDs in {task['id']} {split}")
    partitions = split_samples(partitions, task)
    if task["kind"] != "correspondence":
        seen = set()
        for rows in partitions.values():
            ids = {row["id"] for row in rows}
            if seen & ids:
                raise ValueError(f"Overlapping partitions in {task['id']}")
            seen.update(ids)
    for split, rows in partitions.items():
        if not rows:
            raise ValueError(f"Empty {split} partition in {task['id']}")
        for row in rows:
            row["files"] = {}
            for field in ("image", "target_image", "mask", "annotation"):
                if field in row:
                    kind = "image" if field in ("image", "target_image") else field
                    row["files"][row[field]] = verify_file(source_root, row[field], inventory, kind, task)
    return partitions, labels


def build_manifest(data_root, source_root, task, source, inventory):
    partitions, labels = validated_partitions(source_root, task, source, inventory)
    directory = data_root / ".bench" / "prepared" / task["id"]
    splits = {}
    for split, rows in partitions.items():
        split_digest = digest(rows)
        filename = f"{split}-{split_digest}.json"
        path = directory / filename
        if not path.exists():
            write_json(path, rows)
        splits[split] = {"file": filename, "digest": split_digest, "count": len(rows)}
    diagnostic_rows = distinct_images(partitions["train"]) if task["kind"] == "correspondence" else partitions["train"]
    manifest = {"schema_version": "1.0.0", "task": task, "source": source, "labels": labels, "splits": splits, "diagnostic_sample_ids": [row["id"] for row in diagnostic_rows[:task["diagnostics"]["maximum_images"]]]}
    manifest["digest"] = digest(manifest)
    manifest["source_root"] = str(source_root)
    write_json(directory / "manifest.json", manifest)
    # Only obsolete bench-owned split manifests are removed after commit.
    keep = {entry["file"] for entry in splits.values()}
    for pattern in ("train-*.json", "development-*.json", "final-*.json"):
        for path in directory.glob(pattern):
            if path.name not in keep:
                path.unlink()
    return manifest


def prepare(*, names, data_root=Path("~/datasets"), datasets=None):
    tasks = resolve_names(names)
    root = resolve_path(data_root)
    root.mkdir(parents=True, exist_ok=True)
    print(f"Data root: {root}", flush=True)
    sources = source_recipes()
    report = {"data_root": str(root), "requested_names": names, "tasks": {}}
    started = time.monotonic()
    for source_name in dict.fromkeys(task["source"] for task in tasks):
        source_tasks = [task for task in tasks if task["source"] == source_name]
        try:
            with lock(root / ".bench" / "locks" / f"{source_name}.lock"):
                source = sources[source_name]
                staged_inventory = {}

                def validate_staged(path):
                    for task in source_tasks:
                        validated_partitions(path, task, source, staged_inventory)

                acquisition_started = time.monotonic()
                source_root = acquire(root, source_name, source, (datasets or {}).get(source_name), validate=validate_staged)
                acquisition_seconds = time.monotonic() - acquisition_started
                inventory_path = root / ".bench" / "sources" / f"{source_name}.json"
                old = read_json(inventory_path) if inventory_path.exists() else {}
                inventory = old.get("files", {}) if old.get("root") == str(source_root) else {}
                inventory.update(staged_inventory)
                for task in source_tasks:
                    task_started = time.monotonic()
                    try:
                        manifest = build_manifest(root, source_root, task, source, inventory)
                        report["tasks"][task["id"]] = {"status": "ready", "release_status": task["release_status"], "manifest": str(root / ".bench" / "prepared" / task["id"] / "manifest.json"), "source_root": str(source_root), "digest": manifest["digest"], "counts": {split: entry["count"] for split, entry in manifest["splits"].items()}, "seconds": time.monotonic() - task_started, "source_acquisition_and_validation_seconds": acquisition_seconds, "verified_source_bytes": sum(entry["size"] for entry in inventory.values())}
                        print(f"{task['id']}: prepared {report['tasks'][task['id']]['counts']}", flush=True)
                    except (OSError, ValueError, KeyError) as error:
                        report["tasks"][task["id"]] = {"status": "incomplete", "error": str(error)}
                    finally:
                        write_json(inventory_path, {"root": str(source_root), "files": inventory})
        except (OSError, ValueError, RuntimeError) as error:
            for task in source_tasks:
                report["tasks"][task["id"]] = {"status": "incomplete", "error": str(error)}
    report["seconds"] = time.monotonic() - started
    report["ready"] = all(task["status"] == "ready" for task in report["tasks"].values())
    report["prepared_bytes"] = sum(path.stat().st_size for path in (root / ".bench" / "prepared").rglob("*") if path.is_file()) if (root / ".bench" / "prepared").exists() else 0
    return report


def load_prepared(data_root, task, split, *, verify=True):
    root = resolve_path(data_root)
    directory = root / ".bench" / "prepared" / task["id"]
    remedy = prepare_command([task["name"]], root)
    try:
        manifest = read_json(directory / "manifest.json")
        identity = {key: value for key, value in manifest.items() if key not in ("digest", "source_root")}
        if digest(identity) != manifest["digest"] or manifest["task"]["digest"] != task["digest"]:
            raise ValueError("Prepared recipe or manifest digest changed")
        entry = manifest["splits"][split]
        rows = read_json(contained_path(directory, entry["file"]))
        if digest(rows) != entry["digest"] or len(rows) != entry["count"]:
            raise ValueError(f"Corrupt {split} sample manifest")
        source_root = Path(manifest["source_root"])
        if verify:
            inventory_path = root / ".bench" / "sources" / f"{task['source']}.json"
            inventory = read_json(inventory_path)["files"]
            checked = set()
            for row in rows:
                for relative, expected in row["files"].items():
                    if relative not in checked:
                        if verify_file(source_root, relative, inventory) != expected:
                            raise ValueError(f"Source checksum changed: {relative}")
                        checked.add(relative)
        return manifest, rows
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"Missing, incomplete, or changed data for {task['id']} ({split}): {error}. Expected manifest: {directory / 'manifest.json'}. Run {remedy}") from error
