"""Frozen linear probes, bounded feature reads, and development selection."""

from collections import OrderedDict
from contextlib import nullcontext
from copy import deepcopy
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



def small_affine_weights(grid, source_shape):
    """Separable border/bilinear weights for bounded, axis-aligned small grids."""
    if max(source_shape) > 8:
        return None
    x, y = grid[0, :, 0], grid[:, 0, 1]
    if not (torch.equal(grid[..., 0], x[None, :].expand(grid.shape[:2]))
            and torch.equal(grid[..., 1], y[:, None].expand(grid.shape[:2]))):
        return None
    weights = []
    for coordinates, length in zip((y, x), source_shape):
        positions = (((coordinates + 1) * length - 1) / 2).clamp(0, length - 1)
        centers = torch.arange(length, device=grid.device, dtype=torch.float32)
        weights.append((1 - (positions[:, None] - centers[None]).abs()).clamp_min(0))
    return weights


class FitResampling:
    """Bounded, one-fit reuse of fixed affine sampling tensors only."""

    gpu_byte_limit = 1024 ** 3
    cpu_byte_limit = 32 * 1024 ** 2

    def __init__(self, task, device):
        self.entries, self.current_bytes, self.peak_bytes = {}, 0, 0
        self.device = torch.device(device)
        self.enabled = task["kind"] == "segmentation"
        self.limit = self.cpu_byte_limit
        if self.enabled and self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            reusable = torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
            self.limit = min(self.gpu_byte_limit, (free + reusable) // 8)

    def __call__(self, sample, logits, grid_to_original, canvas_to_original, size):
        if not self.enabled or not self.limit:
            return logits_on_canvas(logits, grid_to_original, canvas_to_original, size)
        shape = tuple(logits.shape[-2:]), tuple(size)
        key = "sample", sample, shape
        payload = self.entries.get(key)
        if payload is None:
            shared = ("aligned-shape", None, shape)
            if max(logits.shape[-2:]) <= 32 and grid_to_original.device == canvas_to_original.device and torch.equal(grid_to_original, canvas_to_original):
                payload = self.entries.get(shared)
                if payload is None:
                    weights = []
                    for source, target in zip(logits.shape[-2:], size):
                        positions = ((torch.arange(target, device=logits.device, dtype=torch.float32) + 0.5) * source / target - 0.5).clamp(0, source - 1)
                        centers = torch.arange(source, device=logits.device, dtype=torch.float32)
                        weights.append((1 - (positions[:, None] - centers[None]).abs()).clamp_min(0))
                    payload = "weights", weights
                    self.store(shared, payload)
            else:
                height, width = size
                y, x = torch.meshgrid((torch.arange(height, device=logits.device, dtype=torch.float32) + 0.5) / height, (torch.arange(width, device=logits.device, dtype=torch.float32) + 0.5) / width, indexing="ij")
                points = torch.stack([x, y, torch.ones_like(x)], dim=-1)
                original = (points @ canvas_to_original.to(logits.device).float().T)[..., :2]
                matrix = torch.linalg.inv(grid_to_original.float()).to(logits.device)
                original = original.to(logits.device, torch.float32)
                homogeneous = torch.cat([original, torch.ones_like(original[..., :1])], dim=-1)
                payload = "grid", (homogeneous @ matrix.T)[..., :2] * 2 - 1
                weights = small_affine_weights(payload[1], logits.shape[-2:])
                if weights is not None:
                    payload = "small-weights", weights
            if payload[0] == "weights" and shared in self.entries:
                self.entries[key] = payload
            else:
                self.store(key, payload)
        kind, values = payload
        if kind in ("weights", "small-weights"):
            with torch.autocast(logits.device.type, enabled=False):
                return values[0] @ logits.float() @ values[1].T
        return F.grid_sample(logits[None].float(), values[None], mode="bilinear", padding_mode="border", align_corners=False)[0]

    def store(self, key, payload):
        tensors = payload[1] if payload[0] in ("weights", "small-weights") else [payload[1]]
        needed = sum(t.numel() * t.element_size() for t in tensors)
        if self.current_bytes + needed <= self.limit:
            self.entries[key] = payload
            self.current_bytes += needed
            self.peak_bytes = max(self.peak_bytes, self.current_bytes)

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.entries.clear()
        self.current_bytes = 0


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


class StandardizedFeatures:
    """Bounded fit-local reuse, preserving the original standardization sites."""

    gpu_byte_limit = 12 * 1024 ** 3
    cpu_byte_limit = 32 * 1024 ** 2

    def __init__(self, bank, training, development, statistics, device):
        self.bank, self.training, self.development, self.statistics = bank, training, development, statistics
        self.rows = list({row["id"]: row for row in training + development}.values())
        self.device = torch.device(device)
        self.values, self.indices, self.geometry = None, {}, []
        self.info = {"enabled": False, "rows": len(self.rows), "peak_allocated_bytes": 0,
            "standardization": "unchanged: global-training CPU; scoring/spatial execution device",
            "lifetime": "one-probe-fit"}

    def __enter__(self):
        shape = self.bank.manifest["shape"]
        required = len(self.rows) * math.prod(shape) * 4
        limit = self.cpu_byte_limit
        if self.device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(self.device)
            reusable = torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
            limit = min(self.gpu_byte_limit, (free + reusable) // 2)
        self.info["byte_limit"] = limit
        overlap = {row["id"] for row in self.training} & {row["id"] for row in self.development}
        if not self.rows or required > limit or (len(shape) == 1 and self.device.type == "cuda" and overlap):
            self.info["reason"] = "streaming-fallback-byte-limit-or-overlapping-roles"
            return self
        self.indices = {row["id"]: index for index, row in enumerate(self.rows)}
        try:
            self.values = torch.empty((len(self.rows), *shape), dtype=torch.float32, device=self.device)
        except torch.cuda.OutOfMemoryError:
            self.close()
            self.info["reason"] = "streaming-fallback-allocation"
            return self
        try:
            with torch.no_grad():
                if len(shape) == 1:
                    trained = {row["id"] for row in self.training}
                    for scoring in [False, True]:
                        rows = [row for row in self.rows if (row["id"] not in trained) is scoring]
                        for start in range(0, len(rows), 1024):
                            chunk = rows[start:start + 1024]
                            raw = self.bank.read_values([row["id"] for row in chunk])
                            normalized = standardize(raw.to(self.device), self.statistics) if scoring else standardize(raw, self.statistics).to(self.device)
                            indices = torch.tensor([self.indices[row["id"]] for row in chunk], device=self.device)
                            self.values.index_copy_(0, indices, normalized)
                else:
                    for index, row in enumerate(self.rows):
                        feature = self.bank[row["id"]]
                        self.values[index].copy_(standardize(feature["values"].to(self.device), self.statistics))
                        self.geometry.append({key: feature[key].detach().cpu().clone()
                            for key in ["grid_to_original", "extent_in_original"] if key in feature})
            self.info.update(enabled=True, peak_allocated_bytes=required)
            return self
        except BaseException:
            self.close()
            raise

    @property
    def enabled(self):
        return self.values is not None

    def read_values(self, ids):
        indices = torch.tensor([self.indices[sample] for sample in ids], dtype=torch.long, device=self.device)
        return self.values.index_select(0, indices)

    def read_feature(self, sample):
        index = self.indices[sample]
        return {**self.geometry[index], "values": self.values[index].clone()}

    def close(self):
        self.values = None
        self.indices.clear()
        self.geometry.clear()

    def __exit__(self, *error):
        self.close()


def global_values(bank, ids, statistics, device, features, *, scoring=False):
    if features is not None and features.enabled:
        return features.read_values(ids)
    values = bank.read_values(ids)
    return standardize(values.to(device), statistics) if scoring else standardize(values, statistics).to(device)


def spatial_values(bank, sample, statistics, device, features):
    if features is not None and features.enabled:
        feature = features.read_feature(sample)
        return feature, feature["values"]
    feature = bank[sample]
    return feature, standardize(feature["values"].to(device), statistics)


def label_canvas(root, row, task):
    with Image.open(contained_path(root, row["image"])) as image:
        resized, offset, transform = fit_pad_geometry(image.size, task["canvas"])
    mask = read_mask(contained_path(root, row["mask"]), task)
    return torch.from_numpy(fit_pad_mask(mask, task["canvas"], resized, offset)), torch.from_numpy(transform)


class PreparedTargets:
    """Bounded, lossless CPU targets for one active probe fit or scoring call."""

    byte_limit = 512 * 1024 ** 2

    def __init__(self, root, task):
        self.root, self.task = root, task
        self.entries = OrderedDict()
        self.current_bytes = 0

    def read(self, row):
        key = (row["id"], row["image"], row["mask"], tuple(sorted(row.get("files", {}).items())))
        if key in self.entries:
            value = self.entries.pop(key)
            self.entries[key] = value
            return value[:2]
        target, transform = label_canvas(self.root, row, self.task)
        if target.numel() + transform.numel() * transform.element_size() > self.byte_limit:
            return target, transform
        # Canonical segmentation labels and the ignore value fit in one byte.
        # Preserve out-of-range targets so the ordinary loss still rejects them.
        packed = target.to(torch.uint8) if int(target.min()) >= 0 and int(target.max()) <= 255 else target
        size = packed.numel() * packed.element_size() + transform.numel() * transform.element_size()
        if size > self.byte_limit:
            return target, transform
        while self.entries and self.current_bytes + size > self.byte_limit:
            _, (_, _, removed) = self.entries.popitem(last=False)
            self.current_bytes -= removed
        self.entries[key] = packed, transform, size
        self.current_bytes += size
        return packed, transform

    def close(self):
        self.entries.clear()
        self.current_bytes = 0

    def __enter__(self):
        return self

    def __exit__(self, *error):
        self.close()


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


def score(probe, bank, rows, root, task, statistics, device, seed, targets=None, features=None):
    total_loss, denominator, records = 0.0, 0, []
    intersections = torch.zeros(task["classes"], dtype=torch.int64)
    unions = torch.zeros_like(intersections)
    with torch.no_grad():
        if task["kind"] == "classification":
            batch_size = task["probe"]["batch_size"]
            for offset in range(0, len(rows), batch_size):
                batch = rows[offset:offset + batch_size]
                values = global_values(bank, [row["id"] for row in batch], statistics, device, features, scoring=True)
                logits = probe(values)
                targets = torch.tensor([row["target"] for row in batch], device=device)
                total_loss += float(F.cross_entropy(logits, targets, reduction="sum"))
                denominator += len(batch)
                predictions = logits.argmax(1).cpu().tolist()
                records.extend({"id": row["id"], "seed": seed, "label": row["target"], "prediction": prediction} for row, prediction in zip(batch, predictions))
        else:
            for row in rows:
                feature, values = spatial_values(bank, row["id"], statistics, device, features)
                logits = probe(values[None])[0]
                target, transform = targets.read(row) if targets is not None else label_canvas(root, row, task)
                target = target.to(device).long()
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


def train_candidate(bank, rows, development, root, task, statistics, device, seed, learning_rate, weight_decay, destination, targets=None, features=None):
    recipe = task["probe"]
    probe = LinearProbe(bank.manifest["shape"][0], task["classes"], seed, device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=tuple(recipe["betas"]), eps=recipe["epsilon"], amsgrad=recipe["amsgrad"], foreach=recipe["foreach"], fused=recipe["fused"])
    generator = torch.Generator().manual_seed(seed)
    batch_size = recipe["batch_size"]
    updates_per_epoch = math.ceil(len(rows) / batch_size)
    total_updates = recipe.get("updates", recipe.get("epochs", 1) * updates_per_epoch)
    warmup = max(1, round(total_updates * recipe["warmup_fraction"]))
    step, epoch, best = 0, 0, None
    with FitResampling(task, device) as resampling:
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
                    values = global_values(bank, [row["id"] for row in batch], statistics, device, features)
                    labels = torch.tensor([row["target"] for row in batch], device=device)
                    loss = F.cross_entropy(probe(values), labels, reduction="sum")
                    (loss / len(batch)).backward()
                    loss_sum += float(loss.detach())
                    loss_count += len(batch)
                else:
                    labels = [targets.read(row) if targets is not None else label_canvas(root, row, task) for row in batch]
                    device_labels = torch.stack([target for target, _ in labels]).to(device=device, dtype=torch.long, non_blocking=True)
                    valid_count = sum(int((target != 255).sum()) for target, _ in labels)
                    if not valid_count:
                        raise ValueError("Effective probe batch contains no eligible segmentation pixels")
                    # Each image is a microbatch; all losses share the effective-batch denominator.
                    batch_losses = []
                    for row, (target, transform), device_target in zip(batch, labels, device_labels):
                        feature, values = spatial_values(bank, row["id"], statistics, device, features)
                        logits = resampling(row["id"], probe(values[None])[0], feature["grid_to_original"], transform, target.shape)
                        loss = F.cross_entropy(logits[None], device_target[None], ignore_index=255, reduction="sum")
                        (loss / valid_count).backward()
                        batch_losses.append(loss.detach())
                    for value in torch.stack(batch_losses).cpu().tolist():
                        loss_sum += value
                    loss_count += valid_count
                optimizer.step()
            if recipe["selection"] == "final-update" and step < total_updates:
                continue
            development_result, _ = score(probe, bank, development, root, task, statistics, device, seed, targets=targets, features=features)
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
    with (PreparedTargets(root, task) if task["kind"] == "segmentation" else nullcontext()) as targets:
        return _fit_probe(bank, training=training, development=development, root=root, task=task,
                          directory=directory, execution=execution, diagnostics=diagnostics, targets=targets)


def _fit_probe(bank, *, training, development, root, task, directory, execution, diagnostics, targets):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ready = read_selection(directory)
    if ready is not None:
        return ready
    statistics = normalization(bank, training, task)
    save_tensors(directory / "normalization.pt", statistics, storage=bank.storage)
    with StandardizedFeatures(bank, training, development, statistics, execution.device) as features:
        return _fit_candidates(bank, training=training, development=development, root=root, task=task,
            directory=directory, execution=execution, diagnostics=diagnostics, targets=targets,
            statistics=statistics, features=features)


def _fit_candidates(bank, *, training, development, root, task, directory, execution, diagnostics, targets, statistics, features):
    if 'search_plan' in task['probe']:
        from .probe_search import fit_full_probe

        def train(seed, epochs, rate, decay, destination):
            variant = deepcopy(task)
            variant['probe'].update(epochs=epochs, learning_rates=[rate], weight_decays=[decay], seeds=[seed])
            variant['digest'] = digest({key: value for key, value in variant.items() if key != 'digest'})
            return train_candidate(bank, training, development, root, variant, statistics, execution.device,
                seed, rate, decay, destination, targets=targets, features=features)

        header = {'shape': bank.manifest['shape'], 'dtype': bank.manifest['dtype'],
            'bytes_per_sample': bank.manifest['bytes_per_sample'],
            'normalization_sha256': file_digest(directory / 'normalization.pt'),
            'diagnostics': diagnostics, 'feature_cache': dict(features.info)}
        return fit_full_probe(task['probe'], directory=directory, storage=bank.storage, train=train, header=header)
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
        candidate = train_candidate(bank, training, development, root, task, statistics, execution.device, 0, learning_rate, weight_decay, directory / f"candidate-{index}.pt", targets=targets, features=features)
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
            metadata = train_candidate(bank, training, development, root, task, statistics, execution.device, seed, seeds[0]["learning_rate"], seeds[0]["weight_decay"], path, targets=targets, features=features)
            metadata["digest"] = digest(metadata)
            write_json(metadata_path, metadata, storage=bank.storage)
        seeds.append(metadata)
    selection = {"identity": directory.name, "seeds": seeds, "candidates": search["candidates"], "shape": bank.manifest["shape"], "dtype": bank.manifest["dtype"], "bytes_per_sample": bank.manifest["bytes_per_sample"], "normalization_sha256": file_digest(directory / "normalization.pt"), "diagnostics": diagnostics}
    selection["feature_cache"] = dict(features.info)
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
    with (PreparedTargets(root, task) if task["kind"] == "segmentation" else nullcontext()) as targets:
        for seed in selection["seeds"]:
            probe = LinearProbe(selection["shape"][0], task["classes"], seed["seed"], execution.device)
            probe.load_state_dict(torch.load(Path(directory) / seed["head_file"], weights_only=True, map_location=execution.device))
            result, observations = score(probe, bank, rows, root, task, statistics, execution.device, seed["seed"], targets=targets)
            results.append({**result, "seed": seed["seed"]})
            records.extend(observations)
    metrics = {task["metric"]: float(np.mean([result["score"] for result in results]))}
    if len(results) > 1:
        metrics[task["metric"] + "_std"] = float(np.std([result["score"] for result in results], ddof=0))
    return metrics, results, records
