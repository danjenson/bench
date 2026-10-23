"""SPair pair annotations and scoring eligibility, independent of adapters."""

import numpy as np

from ..storage import digest, read_json


def pck_radius(annotation, threshold):
    box = np.asarray(annotation["trg_bndbox"], dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or (box[2:] <= box[:2]).any():
        raise ValueError("SPair target bounding box must be finite with positive width and height")
    return threshold * float((box[2:] - box[:2]).max())


def eligible_keypoints(annotation):
    source = np.asarray(annotation["src_kps"], dtype=np.float64)
    target = np.asarray(annotation["trg_kps"], dtype=np.float64)
    if source.ndim != 2 or source.shape[1] != 2 or source.shape != target.shape:
        raise ValueError("SPair source and target keypoints must be matching [N, 2] arrays")
    sw, sh = annotation["src_imsize"][:2]
    tw, th = annotation["trg_imsize"][:2]
    valid = np.isfinite(source).all(1) & np.isfinite(target).all(1)
    valid &= (source >= 0).all(1) & (source < [sw, sh]).all(1)
    valid &= (target >= 0).all(1) & (target < [tw, th]).all(1)
    return source, target, valid


def index_spair(root, task):
    partitions = {}
    for official, split in [("trn", "train"), ("val", "development"), ("test", "final")]:
        rows = []
        layout = root / "Layout" / "large" / f"{official}.txt"
        for pair in layout.read_text().splitlines():
            annotation = f"PairAnnotation/{official}/{pair}.json"
            data = read_json(root / annotation)
            eligible_keypoints(data)
            pck_radius(data, task["threshold"])
            category = data["category"]
            source = f"JPEGImages/{category}/{data['src_imname']}"
            target = f"JPEGImages/{category}/{data['trg_imname']}"
            rows.append({"id": digest(["spair71k", pair]), "image": source, "target_image": target, "source_id": digest(["spair71k", source]), "target_id": digest(["spair71k", target]), "annotation": annotation, "category": category, "viewpoint_variation": data["viewpoint_variation"]})
        partitions[split] = rows
    return partitions, sorted({row["category"] for row in partitions["train"]})


def distinct_images(rows):
    images = {}
    for row in rows:
        for field, identity in [("image", "source_id"), ("target_image", "target_id")]:
            images[row[identity]] = {"id": row[identity], "image": row[field], "files": {row[field]: row["files"][row[field]]}}
    return [images[key] for key in sorted(images)]


def match_pair(source, target, annotation, threshold=0.1):
    import torch
    from torch.nn import functional as F

    from ..geometry import grid_centers, sample_original, valid_centers

    source_points, target_points, eligible = eligible_keypoints(annotation)
    radius = pck_radius(annotation, threshold)
    source_points = torch.from_numpy(source_points[eligible]).float()
    target_points = torch.from_numpy(target_points[eligible]).float()
    if not len(source_points):
        return {"correct": 0, "eligible": 0, "grid_ceiling_correct": 0}
    source_size = torch.tensor(annotation["src_imsize"][:2]).float()
    target_size = torch.tensor(annotation["trg_imsize"][:2]).float()
    source_vectors = sample_original(source["values"], source["grid_to_original"], (source_points / source_size)[:, None])[:, :, 0].T
    valid = valid_centers(target)
    if not valid.any():
        raise ValueError("SPair spatial grid has no native feature center inside target image content")
    target_vectors = target["values"][:, valid].T.float()
    similarities = F.normalize(source_vectors, dim=1) @ F.normalize(target_vectors, dim=1).T
    centers = grid_centers(target)[valid] * target_size
    matches = centers[similarities.argmax(1)]  # argmax chooses the first raster-order tie.
    return {"correct": int(((matches - target_points).norm(dim=1) <= radius).sum()), "eligible": len(target_points), "grid_ceiling_correct": int((torch.cdist(target_points, centers).min(1).values <= radius).sum())}


def evaluate_pairs(bank, rows, root, task):
    from collections import defaultdict

    from ..storage import contained_path

    records, groups = [], defaultdict(lambda: {"correct": 0, "eligible": 0, "grid_ceiling_correct": 0})
    for row in rows:
        annotation = read_json(contained_path(root, row["annotation"]))
        counts = match_pair(bank[row["source_id"]], bank[row["target_id"]], annotation, task["threshold"])
        breakdown = {key: annotation.get(key) for key in ("category", "viewpoint_variation", "scale_variation", "occlusion", "truncation")}
        records.append({"id": row["id"], "source_id": row["source_id"], "target_id": row["target_id"], **counts, "breakdown": breakdown})
        for group in ["all"] + [f"{key}={value}" for key, value in breakdown.items()]:
            for key, value in counts.items():
                groups[group][key] += value
    totals = groups["all"]
    if not totals["eligible"]:
        raise ValueError("SPair partition has no eligible correspondences")
    metrics = {"pck_at_0.1": 100 * totals["correct"] / totals["eligible"], "grid_limited_pck_ceiling": 100 * totals["grid_ceiling_correct"] / totals["eligible"]}
    supporting = {name: {**counts, "pck_at_0.1": 100 * counts["correct"] / counts["eligible"] if counts["eligible"] else None} for name, counts in groups.items()}
    return metrics, supporting, records
