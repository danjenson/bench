"""ImageNet and CLEVR source definitions and target rules."""

from bisect import bisect_left
from importlib.resources import files
import hashlib
import json
import math

from ..storage import contained_path, digest, file_digest, read_json


def clevr_target(scene, task):
    objects = scene["objects"]
    if not 3 <= len(objects) <= 10:
        raise ValueError("CLEVR v1.0 scenes must contain 3 through 10 objects")
    if task["name"] == "clevr-count":
        return len(objects) - 3
    distance = min(float(item["pixel_coords"][2]) for item in objects)
    if not math.isfinite(distance) or distance < 0 or distance > 100:
        raise ValueError(f"Invalid CLEVR camera distance: {distance}")
    return bisect_left(task["distance_thresholds"], distance)


def index_clevr(root, task):
    partitions = {}
    for official, split in [("train", "train"), ("val", "final")]:
        annotations = f"scenes/CLEVR_{official}_scenes.json"
        rows = []
        for scene_index, scene in enumerate(read_json(root / annotations)["scenes"]):
            image = f"images/{official}/{scene['image_filename']}"
            contained_path(root, image)
            rows.append({"id": digest(["clevr", image]), "image": image, "target": clevr_target(scene, task), "scene_index": scene_index, "annotation": annotations})
        partitions[split] = rows
    labels = [f"{count} objects" for count in range(3, 11)] if task["name"] == "clevr-count" else [f"depth <= {value:g}" for value in task["distance_thresholds"]]
    return partitions, labels


def index_imagenet(root, task):
    if "class_list" in task:
        path = files("bench.cli.configs.task").joinpath(task["class_list"])
        if file_digest(path) != task["class_list_sha256"]:
            raise ValueError("The packaged ImageNet-100 class list checksum does not match its protocol")
        classes = sorted(path.read_text().splitlines())
    else:
        classes = sorted(path.name for path in (root / "train").iterdir() if path.is_dir() and path.name.startswith("n"))
    if len(classes) != task["classes"]:
        raise ValueError(f"Expected {task['classes']} ImageNet classes, found {len(classes)}")
    partitions = {}
    for official, split in [("train", "train"), ("val", "final")]:
        rows = []
        for target, label in enumerate(classes):
            directory = root / official / label
            images = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
            if not images:
                raise ValueError(f"No ImageNet images in {directory}; validation must be organized by synset")
            for path in images:
                image = path.relative_to(root).as_posix()
                rows.append({"id": digest(["imagenet", image]), "image": image, "target": target, "synset": label})
        partitions[split] = rows
    mapping_path = files("bench.cli.configs.task").joinpath("imagenet_classes.json")
    if file_digest(mapping_path) != "a1e7a966a1f601d39e4b43e119b3e7dd4a2ad3ea08cf69847cbaf021013767bc":
        raise ValueError("The packaged ImageNet class-name mapping checksum does not match its source")
    mapping = {synset: name.replace("_", " ") for synset, name in read_json(mapping_path).values()}
    return partitions, [f"{mapping[synset]} ({synset})" for synset in classes]


def apply_reimagenet(root, rows, task):
    """Attach final-only targets from the pinned, caller-obtained annotations."""
    protocol = task["final_labels"]
    directory = root / "reimagenet"
    contents, checksums = {}, {}
    for name, pin in protocol["files"].items():
        path = directory / name
        if not path.is_file():
            raise FileNotFoundError(f"ReImageNet requires {path}. Accept access at https://huggingface.co/datasets/vrg-prague/ReImageNet and download the three annotation files at revision {protocol['revision']} into {directory}.")
        payload = path.read_bytes()
        checksums[f"reimagenet/{name}"] = hashlib.sha256(payload).hexdigest()
        if checksums[f"reimagenet/{name}"] != pin:
            raise ValueError(f"ReImageNet checksum mismatch: {path}; expected release {protocol['revision']}")
        contents[name] = payload
    synsets = json.loads(contents["label_names.json"])
    classes = sorted({row["synset"] for row in rows})
    if not isinstance(synsets, list) or len(synsets) != task["classes"] or set(synsets) != set(classes):
        raise ValueError("ReImageNet synsets must match the ImageNet class set exactly")
    indices = {synset: index for index, synset in enumerate(classes)}
    remap = [indices[synset] for synset in synsets]

    def labels(values, *, uncertain=False):
        if not isinstance(values, list) or any(type(value) is not int or not (-int(uncertain) <= value < len(synsets)) for value in values):
            raise ValueError("Invalid ReImageNet class indices")
        return {remap[value] for value in values if value >= 0}

    equivalents = [{index} for index in range(len(classes))]
    for pair in json.loads(contents["class_update_config.json"])["eq_classes"].values():
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("ReImageNet equivalent classes must be pairs")
        group = set().union(*(equivalents[index] for index in labels(pair)))
        for index in group:
            equivalents[index] = group
    expected = {row["image"].removeprefix("val/"): row for row in rows}
    seen = set()
    counts = {"eligible": 0, "no_valid_label": 0, "uncertain": 0}
    for line in contents["reannotation.jsonl"].splitlines():
        annotation = json.loads(line)
        key = annotation["file_path"]
        if key not in expected or key in seen or annotation["image_name"] != key.rsplit("/", 1)[-1]:
            raise ValueError(f"Missing, duplicate, or mismatched ReImageNet image: {key}")
        seen.add(key)
        row = expected[key]
        if row["target"] not in labels(annotation["original_class"]):
            raise ValueError(f"ReImageNet original class disagrees with ImageNet source: {key}")
        supplied = annotation["reannotated_labels"]
        accepted = labels(supplied, uncertain=True)
        status = "eligible" if accepted else "uncertain" if -1 in supplied else "no_valid_label"
        row["accepted_labels"] = sorted(set().union(*(equivalents[index] for index in accepted))) if status == "eligible" else []
        row["label_status"] = status
        counts[status] += 1
    if seen != set(expected):
        raise ValueError(f"Incomplete ReImageNet annotations: expected {len(expected)}, found {len(seen)}")
    if not counts["eligible"]:
        raise ValueError("ReImageNet has no eligible final images")
    return {"dataset": protocol["dataset"], "revision": protocol["revision"], "files": checksums, "counts": counts, "scoring": protocol["scoring"], "uncertain": protocol["uncertain"]}


def classification_correct(record):
    accepted = record.get("accepted_labels", [record["label"]])
    return record["prediction"] in accepted if accepted else None


def classification_score(records):
    if not records or len({"accepted_labels" in row for row in records}) != 1:
        raise ValueError("Classification records must use one consistent label protocol")
    outcomes = [classification_correct(row) for row in records]
    eligible = [value for value in outcomes if value is not None]
    if not eligible:
        raise ValueError("No eligible classification examples")
    result = {"score": 100 * sum(eligible) / len(eligible), "eligible_count": len(eligible)}
    if "accepted_labels" in records[0]:
        result["original_top1"] = 100 * sum(row["prediction"] == row["label"] for row in records) / len(records)
    return result
