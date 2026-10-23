"""ImageNet and CLEVR source definitions and target rules."""

from bisect import bisect_left
from importlib.resources import files
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
