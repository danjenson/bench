"""VOC2012 and ADE20K scene parsing source and label definitions."""

import numpy as np
from PIL import Image

from ..storage import digest


def index_segmentation(root, task):
    partitions = {}
    if task["source"] == "voc2012":
        labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "TV/monitor"]
        for official, split in [("train", "train"), ("val", "final")]:
            annotation = f"ImageSets/Segmentation/{official}.txt"
            names = (root / annotation).read_text().split()
            partitions[split] = [{"id": digest(["voc2012", name]), "image": f"JPEGImages/{name}.jpg", "mask": f"SegmentationClass/{name}.png", "annotation": annotation} for name in names]
    else:
        lines = (root / "objectInfo150.txt").read_text().splitlines()[1:]
        labels = [line.split("\t")[-1].strip() for line in lines if line.strip()]
        if len(labels) != 150:
            raise ValueError("ADEChallengeData2016/objectInfo150.txt must define 150 classes")
        for official, split in [("training", "train"), ("validation", "final")]:
            partitions[split] = []
            for path in sorted((root / "images" / official).glob("*.jpg")):
                image = path.relative_to(root).as_posix()
                partitions[split].append({"id": digest(["ade20k", image]), "image": image, "mask": f"annotations/{official}/{path.stem}.png", "annotation": "objectInfo150.txt"})
    return partitions, labels


def read_mask(path, task):
    with Image.open(path) as image:
        mask = np.array(image, dtype=np.int64)
    if mask.ndim != 2:
        raise ValueError(f"Expected an indexed segmentation mask: {path}")
    if task["source"] == "ade20k":
        if mask.min() < 0 or mask.max() > 150:
            raise ValueError(f"ADE20K mask has labels outside 0..150: {path}")
        mask -= 1
        mask[mask < 0] = 255
    elif not ((mask == 255) | ((mask >= 0) & (mask < task["classes"]))).all():
        raise ValueError(f"VOC mask has labels outside 0..20 and ignore=255: {path}")
    return mask
