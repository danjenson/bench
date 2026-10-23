"""Frozen linear probes, bounded feature reads, and development selection."""

from itertools import product
import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from .features import save_tensors
from .geometry import logits_on_canvas, valid_centers
from .images import fit_pad_geometry, fit_pad_mask
from .protocols import evaluation_source_digest
from .storage import contained_path, digest, file_digest, read_json, remove, write_json
from .tasks.segmentation import read_mask


class LinearProbe(nn.Module):
    def __init__(self, width, classes, seed, device):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        weight = torch.empty(classes, width, dtype=torch.float32).uniform_(-1 / math.sqrt(width), 1 / math.sqrt(width), generator=generator)
        self.weight = nn.Parameter(weight.to(device))
        self.bias = nn.Parameter(torch.zeros(classes, device=device, dtype=torch.float32))

    def forward(self, features):
        if features.ndim == 4:
            return F.conv2d(features, self.weight[:, :, None, None], self.bias)
        return F.linear(features, self.weight, self.bias)


def normalization(bank, rows, task):
    total, mean, centered_sum = 0, None, None
    for row in rows:
        feature = bank[row["id"]]
        values = feature["values"].float()
        if values.ndim == 3:
            values = values[:, valid_centers(feature)].T
        else:
            values = values[None]
        count = len(values)
        if not count:
            continue
        local_mean = values.mean(0)
        local_sum = (values - local_mean).square().sum(0)
        if mean is None:
            mean, centered_sum, total = local_mean, local_sum, count
        else:
            delta = local_mean - mean
            centered_sum += local_sum + delta.square() * (total * count / (total + count))
            mean += delta * (count / (total + count))
            total += count
    if total == 0:
        raise ValueError("No valid feature locations in the probe-training bank")
    return {"mean": mean, "std": (centered_sum / total).clamp_min(task["standardization"]["variance_floor"]).sqrt(), "count": total}


def standardize(values, statistics):
    shape = (-1, 1, 1) if values.ndim == 3 else (-1,)
    return (values.float() - statistics["mean"].reshape(shape).to(values.device)) / statistics["std"].reshape(shape).to(values.device)


def label_canvas(root, row, task):
    with Image.open(contained_path(root, row["image"])) as image:
        resized, offset, transform = fit_pad_geometry(image.size, task["canvas"])
    mask = read_mask(contained_path(root, row["mask"]), task)
    return torch.from_numpy(fit_pad_mask(mask, task["canvas"], resized, offset)), torch.from_numpy(transform)


def segmentation_counts(prediction, target, classes):
    eligible = target != 255
    target, prediction = target[eligible], prediction[eligible]
    counts = torch.bincount(target * classes + prediction, minlength=classes * classes).reshape(classes, classes)
    intersection = counts.diag()
    union = counts.sum(0) + counts.sum(1) - intersection
    return intersection, union, int(eligible.sum())


def miou(intersection, union):
    eligible = union > 0
    if not eligible.any():
        raise ValueError("No eligible pixels for segmentation scoring")
    return float((intersection[eligible].double() / union[eligible]).mean() * 100)


def score(probe, bank, rows, root, task, statistics, device, seed):
    total_loss, denominator, records = 0.0, 0, []
    intersections = torch.zeros(task["classes"], dtype=torch.int64)
    unions = torch.zeros_like(intersections)
    with torch.no_grad():
        if task["kind"] == "classification":
            batch_size = task["probe"]["batch_size"]
            for offset in range(0, len(rows), batch_size):
                batch = rows[offset:offset + batch_size]
                values = standardize(bank.read_values([row["id"] for row in batch]).to(device), statistics)
                logits = probe(values)
                targets = torch.tensor([row["target"] for row in batch], device=device)
                total_loss += float(F.cross_entropy(logits, targets, reduction="sum"))
                denominator += len(batch)
                predictions = logits.argmax(1).cpu().tolist()
                records.extend({"id": row["id"], "seed": seed, "label": row["target"], "prediction": prediction} for row, prediction in zip(batch, predictions))
        else:
            for row in rows:
                feature = bank[row["id"]]
                values = standardize(feature["values"].to(device), statistics)
                logits = probe(values[None])[0]
                target, transform = label_canvas(root, row, task)
                target = target.to(device)
                logits = logits_on_canvas(logits, feature["grid_to_original"], transform, target.shape)
                total_loss += float(F.cross_entropy(logits[None], target[None], ignore_index=255, reduction="sum"))
                prediction = logits.argmax(0)
                intersection, union, eligible = segmentation_counts(prediction, target, task["classes"])
                denominator += eligible
                intersections += intersection.cpu()
                unions += union.cpu()
                records.append({"id": row["id"], "seed": seed, "intersection": intersection.cpu().tolist(), "union": union.cpu().tolist(), "eligible_pixels": eligible, "prediction_sha256": hashlib.sha256(prediction.cpu().numpy().astype(np.uint8).tobytes()).hexdigest()})
    metric = 100 * sum(record["label"] == record["prediction"] for record in records) / len(records) if task["kind"] == "classification" else miou(intersections, unions)
    return {"score": metric, "loss": total_loss / denominator, "sample_count": len(rows), "eligible_count": denominator}, records


def train_candidate(bank, rows, development, root, task, statistics, device, seed, learning_rate, weight_decay, destination):
    recipe = task["probe"]
    probe = LinearProbe(bank.manifest["shape"][0], task["classes"], seed, device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=tuple(recipe["betas"]), eps=recipe["epsilon"], amsgrad=recipe["amsgrad"], foreach=recipe["foreach"], fused=recipe["fused"])
    generator = torch.Generator().manual_seed(seed)
    batch_size = recipe["batch_size"]
    updates_per_epoch = math.ceil(len(rows) / batch_size)
    total_updates = recipe.get("updates", recipe.get("epochs", 1) * updates_per_epoch)
    warmup = max(1, round(total_updates * recipe["warmup_fraction"]))
    step, epoch, best = 0, 0, None
    while step < total_updates:
        order = torch.randperm(len(rows), generator=generator).tolist()
        epoch += 1
        loss_sum, loss_count = 0.0, 0
        for offset in range(0, len(rows), batch_size):
            if step == total_updates:
                break
            batch = [rows[index] for index in order[offset:offset + batch_size]]
            step += 1
            scale = step / warmup if step <= warmup else recipe["final_lr_fraction"] + (1 - recipe["final_lr_fraction"]) * (1 + math.cos(math.pi * (step - warmup) / max(1, total_updates - warmup))) / 2
            for group in optimizer.param_groups:
                group["lr"] = learning_rate * scale
            optimizer.zero_grad(set_to_none=True)
            if task["kind"] == "classification":
                values = standardize(bank.read_values([row["id"] for row in batch]), statistics).to(device)
                labels = torch.tensor([row["target"] for row in batch], device=device)
                loss = F.cross_entropy(probe(values), labels, reduction="sum")
                (loss / len(batch)).backward()
                loss_sum += float(loss.detach())
                loss_count += len(batch)
            else:
                labels = [label_canvas(root, row, task) for row in batch]
                valid_count = sum(int((target != 255).sum()) for target, _ in labels)
                if not valid_count:
                    raise ValueError("Effective probe batch contains no eligible segmentation pixels")
                # Each image is a microbatch; all losses share the effective-batch denominator.
                for row, (target, transform) in zip(batch, labels):
                    feature = bank[row["id"]]
                    values = standardize(feature["values"].to(device), statistics)
                    logits = logits_on_canvas(probe(values[None])[0], feature["grid_to_original"], transform, target.shape)
                    loss = F.cross_entropy(logits[None], target[None].to(device), ignore_index=255, reduction="sum")
                    (loss / valid_count).backward()
                    loss_sum += float(loss.detach())
                loss_count += valid_count
            optimizer.step()
        if recipe["selection"] == "final-update" and step < total_updates:
            continue
        development_result, _ = score(probe, bank, development, root, task, statistics, device, seed)
        # Strictly greater preserves earlier-epoch ties.
        if best is None or development_result["score"] > best["development_score"] or recipe["selection"] == "final-update":
            best = {"seed": seed, "epoch": epoch, "updates": step, "learning_rate": learning_rate, "weight_decay": weight_decay, "development_score": development_result["score"], "development_loss": development_result["loss"], "training_loss": loss_sum / loss_count, "probe_parameters": sum(parameter.numel() for parameter in probe.parameters())}
            save_tensors(destination, {key: value.detach().cpu() for key, value in probe.state_dict().items()}, storage=bank.storage)
    best["head_sha256"] = file_digest(destination)
    best["head_file"] = destination.name
    return best


def probe_identity(task, preparation, model_metadata):
    return digest({"task": task["digest"], "model": model_metadata["digest"], "train": preparation["splits"]["train"]["digest"], "development": preparation["splits"]["development"]["digest"], "numerics": model_metadata["numerics"], "seeds": task["probe"]["seeds"], "implementation": evaluation_source_digest(task)})


def read_selection(directory):
    directory = Path(directory)
    path = directory / "selection.json"
    if not path.exists():
        return None
    selection = read_json(path)
    if selection.get("identity") != directory.name or selection.get("digest") != digest({key: value for key, value in selection.items() if key != "digest"}):
        return None
    artifacts = [("normalization.pt", selection["normalization_sha256"])] + [(seed["head_file"], seed["head_sha256"]) for seed in selection["seeds"]]
    for filename, expected in artifacts:
        artifact = contained_path(directory, filename)
        if not artifact.exists() or file_digest(artifact) != expected:
            return None
    return selection


def fit_probe(bank, *, training, development, root, task, directory, execution, diagnostics):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ready = read_selection(directory)
    if ready is not None:
        return ready
    statistics = normalization(bank, training, task)
    save_tensors(directory / "normalization.pt", statistics, storage=bank.storage)
    search_path = directory / "search.json"
    committed = read_json(search_path) if search_path.exists() else {}
    search = committed.get("record", {})
    if search.get("identity") != directory.name or committed.get("sha256") != digest(search):
        search = {"identity": directory.name, "candidates": [], "winner": None}
    # Completed losing candidates need only their scores. The winner must retain weights.
    if search["winner"] is not None:
        winner = search["winner"]
        path = directory / winner["head_file"]
        if not path.exists() or file_digest(path) != winner["head_sha256"]:
            search = {"identity": directory.name, "candidates": [], "winner": None}
    # Finish an interrupted search-commit/prune boundary before another candidate.
    keep_winner = search["winner"]["head_file"] if search["winner"] else None
    for path in directory.glob("candidate-*.pt"):
        if path.name != keep_winner:
            remove(path, storage=bank.storage)
    recipe = task["probe"]
    candidates = list(product(recipe["learning_rates"], recipe["weight_decays"]))
    for index, (learning_rate, weight_decay) in enumerate(candidates):
        if index < len(search["candidates"]):
            continue
        candidate = train_candidate(bank, training, development, root, task, statistics, execution.device, 0, learning_rate, weight_decay, directory / f"candidate-{index}.pt")
        search["candidates"].append(candidate)
        previous = search["winner"]
        if previous is None or (-candidate["development_score"], candidate["epoch"], index) < (-previous["development_score"], previous["epoch"], search["winner_index"]):
            search["winner"], search["winner_index"] = candidate, index
        write_json(search_path, {"record": search, "sha256": digest(search)}, storage=bank.storage)
        for path in directory.glob("candidate-*.pt"):
            if path.name != search["winner"]["head_file"]:
                remove(path, storage=bank.storage)
    seeds = [search["winner"]]
    for seed in recipe["seeds"][1:]:
        metadata_path = directory / f"seed-{seed}.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else None
        path = directory / f"seed-{seed}.pt"
        if metadata is None or metadata.get("digest") != digest({key: value for key, value in metadata.items() if key != "digest"}) or not path.exists() or file_digest(path) != metadata["head_sha256"]:
            metadata = train_candidate(bank, training, development, root, task, statistics, execution.device, seed, seeds[0]["learning_rate"], seeds[0]["weight_decay"], path)
            metadata["digest"] = digest(metadata)
            write_json(metadata_path, metadata, storage=bank.storage)
        seeds.append(metadata)
    selection = {"identity": directory.name, "seeds": seeds, "candidates": search["candidates"], "shape": bank.manifest["shape"], "dtype": bank.manifest["dtype"], "bytes_per_sample": bank.manifest["bytes_per_sample"], "normalization_sha256": file_digest(directory / "normalization.pt"), "diagnostics": diagnostics}
    selection["digest"] = digest(selection)
    write_json(directory / "selection.json", selection, storage=bank.storage)
    keep = {seed["head_file"] for seed in seeds} | {"normalization.pt"}
    for path in directory.glob("*.pt"):
        if path.name not in keep:
            remove(path, storage=bank.storage)
    return selection


def evaluate_selection(selection, directory, bank, rows, root, task, execution):
    statistics = torch.load(Path(directory) / "normalization.pt", weights_only=True, map_location="cpu")
    results, records = [], []
    for seed in selection["seeds"]:
        probe = LinearProbe(selection["shape"][0], task["classes"], seed["seed"], execution.device)
        probe.load_state_dict(torch.load(Path(directory) / seed["head_file"], weights_only=True, map_location=execution.device))
        result, observations = score(probe, bank, rows, root, task, statistics, execution.device, seed["seed"])
        results.append({**result, "seed": seed["seed"]})
        records.extend(observations)
    metrics = {task["metric"]: float(np.mean([result["score"] for result in results]))}
    if len(results) > 1:
        metrics[task["metric"] + "_std"] = float(np.std([result["score"] for result in results], ddof=0))
    return metrics, results, records
