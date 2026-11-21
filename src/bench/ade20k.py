"""Online frozen-feature ADE20K probing following the pinned DINOv3 recipe."""

import hashlib
import math
import time
from itertools import product

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from .api import ImageBatch
from .features import adapter_mode, extract, save_tensors
from .geometry import logits_on_canvas
from .images import resize_short_size, window_boxes
from .probes import miou, read_selection, segmentation_counts
from .progress import progress_bar
from .storage import contained_path, digest, file_digest, remove, write_json
from .tasks.segmentation import read_mask


def photometric(pixels, rng):
    # uint8 clipping between operations matches the reference augmentation.
    if rng.integers(2):
        pixels = (pixels.float() + rng.uniform(-32, 32)).clamp(0, 255).to(torch.uint8)
    contrast_first = bool(rng.integers(2))

    def contrast(value):
        return (value.float() * rng.uniform(0.5, 1.5)).clamp(0, 255).to(torch.uint8) if rng.integers(2) else value

    if contrast_first:
        pixels = contrast(pixels)
    if rng.integers(2):
        pixels = TF.adjust_saturation(pixels, float(rng.uniform(0.5, 1.5)))
    if rng.integers(2):
        pixels = TF.adjust_hue(pixels, float(rng.uniform(-0.5, 0.5)))
    return pixels if contrast_first else contrast(pixels)


def training_view(image, mask, task, rng):
    side = task["canvas"]
    width, height = resize_short_size(image.size, int(side * rng.uniform(*task["augmentation"]["scale_range"])))
    pixels = torch.from_numpy(np.array(image.resize((width, height), Image.Resampling.BILINEAR))).permute(2, 0, 1)
    # Tensor nearest interpolation follows the reference's mask backend.
    target = F.interpolate(torch.from_numpy(mask)[None, None].float(), size=(height, width), mode="nearest")[0, 0].long()
    for attempt in range(task["augmentation"]["crop_attempts"]):
        top = int(rng.integers(max(height - side, 0) + 1))
        left = int(rng.integers(max(width - side, 0) + 1))
        crop = target[top:top + side, left:left + side]
        counts = torch.bincount(crop[crop != 255], minlength=task["classes"])
        if attempt == task["augmentation"]["crop_attempts"] - 1 or ((counts > 0).sum() > 1 and counts.max() < task["augmentation"]["category_max_ratio"] * counts.sum()):
            break
    pixels = pixels[:, top:top + side, left:left + side]
    flipped = rng.random() < task["augmentation"]["flip_probability"]
    if flipped:
        pixels, crop = pixels.flip(-1), crop.flip(-1)
    pixels = photometric(pixels, rng).float() / 255
    crop_height, crop_width = crop.shape
    padded = pixels.new_tensor(task["augmentation"]["padding_rgb"])[..., None, None].expand(3, side, side).clone()
    padded[:, :crop_height, :crop_width] = pixels
    valid = torch.zeros((1, side, side), dtype=torch.bool)
    valid[:, :crop_height, :crop_width] = True
    labels = F.pad(crop, (0, side - crop_width, 0, side - crop_height), value=255)
    transform = torch.tensor([[(-side if flipped else side) / width, 0, (left + crop_width if flipped else left) / width],
                              [0, side / height, top / height], [0, 0, 1]], dtype=torch.float32)
    return padded, valid, transform, labels


class TrainingImages(Dataset):
    def __init__(self, root, rows, task, seed):
        self.root, self.rows, self.task, self.seed = root, rows, task, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        index, draw = item
        row = self.rows[index]
        with Image.open(contained_path(self.root, row["image"])) as stream:
            image = stream.convert("RGB")
        mask = read_mask(contained_path(self.root, row["mask"]), self.task)
        return (*training_view(image, mask, self.task, np.random.default_rng([self.seed, draw])), row["id"])


def training_batches(count, recipe, seed):
    generator = torch.Generator().manual_seed(seed)
    batch_size, draw, updates = recipe["batch_size"], 0, 0
    if count < batch_size:
        raise ValueError("ADE20K training needs at least one complete probe batch")
    while updates < recipe["updates"]:
        order = torch.randperm(count, generator=generator).tolist()
        for start in range(0, count - batch_size + 1, batch_size):
            yield [(index, draw + offset) for offset, index in enumerate(order[start:start + batch_size])]
            draw += batch_size
            updates += 1
            if updates == recipe["updates"]:
                return


class LinearHead(nn.Module):
    def __init__(self, width, classes, dropout, device):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout)
        # A single process sees the whole effective batch, matching SyncBatchNorm.
        self.norm = nn.BatchNorm2d(width, device=device, dtype=torch.float32)
        self.linear = nn.Conv2d(width, classes, 1, device=device, dtype=torch.float32)
        nn.init.normal_(self.linear.weight, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, values):
        return self.linear(self.norm(self.dropout(values)))


class Encoder:
    def __init__(self, model, execution):
        self.model, self.execution = model, execution
        self.examples, self.seconds, self.costs = 0, 0.0, {}
        self.shape, self.dtype = None, None

    def __call__(self, images, valid, transforms, ids):
        device = self.execution.device
        batch = ImageBatch(images.to(device), valid.to(device), transforms.to(device), tuple(ids))
        started = time.monotonic()
        values = extract(self.model, batch, "spatial", self.execution, output_device=device)
        self.seconds += time.monotonic() - started
        self.examples += len(ids)
        shape = list(values["values"].shape[1:])
        if self.shape is not None and shape != self.shape:
            raise ValueError("ADE20K requires a fixed spatial grid shape for its fixed-size crops")
        self.shape, self.dtype = shape, str(values["values"].dtype)
        for name, costs in values["costs"].items():
            for cost in costs.tolist():
                item = self.costs.setdefault(name, {"count": 0, "sum": 0.0, "min": cost, "max": cost})
                item.update(count=item["count"] + 1, sum=item["sum"] + cost, min=min(item["min"], cost), max=max(item["max"], cost))
        return values

    def report(self):
        return {"extracted_examples": self.examples, "reused_examples": 0, "total_seconds": self.seconds,
                "cost_unit": "encoder-input-view", "costs": {name: {**item, "mean": item["sum"] / item["count"]} for name, item in self.costs.items()}}


def evaluation_image(root, row, task):
    with Image.open(contained_path(root, row["image"])) as stream:
        image = stream.convert("RGB")
    pixels = torch.from_numpy(np.array(image.resize(resize_short_size(image.size, task["canvas"]), Image.Resampling.BILINEAR)))
    return pixels.permute(2, 0, 1).float() / 255


def window_input(pixels, box):
    top, bottom, left, right = box
    height, width = pixels.shape[-2:]
    crop = pixels[:, top:bottom, left:right]
    transform = torch.tensor([[(right - left) / width, 0, left / width], [0, (bottom - top) / height, top / height], [0, 0, 1]], dtype=torch.float32)
    return crop[None], torch.ones((1, 1, *crop.shape[-2:]), dtype=torch.bool), transform[None]


@torch.no_grad()
def predict(encoder, head, pixels, original_shape, sample_id, task):
    height, width = pixels.shape[-2:]
    # Keep full-image aggregation on the host; only one window's logits is on GPU.
    logits = torch.zeros((task["classes"], height, width), dtype=torch.float32)
    counts = torch.zeros((height, width), dtype=torch.float32)
    head.eval()
    for box in window_boxes(height, width, task["canvas"], task["evaluation_stride"]):
        images, valid, transform = window_input(pixels, box)
        feature = encoder(images, valid, transform, [sample_id])
        coarse = head(feature["values"].float())[0]
        dense = logits_on_canvas(coarse, feature["grid_to_original"][0], transform[0].to(coarse.device), images.shape[-2:]).cpu()
        top, bottom, left, right = box
        logits[:, top:bottom, left:right] += dense
        counts[top:bottom, left:right] += 1
    logits /= counts
    return F.interpolate(logits[None], size=original_shape, mode="bilinear", align_corners=False)[0]


def score(encoder, head, rows, root, task, seed, *, progress=None):
    intersections = torch.zeros(task["classes"], dtype=torch.int64)
    unions = torch.zeros_like(intersections)
    records, eligible_count = [], 0
    with progress_bar(rows, total=len(rows), desc=f"{task['id']} sliding evaluation", unit="image", enabled=progress) as bar:
        for row in bar:
            target = torch.from_numpy(read_mask(contained_path(root, row["mask"]), task))
            prediction = predict(encoder, head, evaluation_image(root, row, task), target.shape, row["id"], task).argmax(0)
            intersection, union, eligible = segmentation_counts(prediction, target, task["classes"])
            intersections += intersection
            unions += union
            eligible_count += eligible
            records.append({"id": row["id"], "seed": seed, "intersection": intersection.tolist(), "union": union.tolist(),
                            "eligible_pixels": eligible, "prediction_sha256": hashlib.sha256(prediction.numpy().astype(np.uint8).tobytes()).hexdigest()})
    return {"seed": seed, "score": miou(intersections, unions), "sample_count": len(rows), "eligible_count": eligible_count}, records


def learning_rate_scale(update, recipe):
    # The reference scheduler is first advanced after the second optimizer step.
    step = max(update - 1, 0)
    if step < recipe["warmup_updates"]:
        return (recipe["warmup_ratio"] + (1 - recipe["warmup_ratio"]) * step / recipe["warmup_updates"]) * (1 - step / recipe["updates"])
    return (1 + math.cos(math.pi * (step + 1) / recipe["updates"])) / 2


def backward_pixels(coarse, transforms, labels, canvas_transforms):
    # Backpropagate dense logits one image at a time, then through the shared BN
    # graph once. This preserves full-batch statistics without a B×150×512² graph.
    detached = coarse.detach().requires_grad_()
    total = 0.0
    for index, target in enumerate(labels):
        logits = logits_on_canvas(detached[index], transforms[index], canvas_transforms[index], target.shape)
        # Reference CE zeros ignored pixels but averages over ALL canvas pixels.
        loss = F.cross_entropy(logits[None], target[None], ignore_index=255, reduction="sum") / labels.numel()
        loss.backward()
        total += float(loss.detach())
    coarse.backward(detached.grad)
    return total


def train_candidate(encoder, training, development, root, task, width, rate, decay, destination, storage):
    recipe, execution = task["probe"], encoder.execution
    seed = recipe["seeds"][0]
    with adapter_mode(encoder.model):
        torch.manual_seed(seed)
        head = LinearHead(width, task["classes"], recipe["dropout"], execution.device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=rate, weight_decay=decay, betas=tuple(recipe["betas"]), eps=recipe["epsilon"], foreach=False, fused=False)
        loader = DataLoader(TrainingImages(root, training, task, seed), batch_sampler=training_batches(len(training), recipe, seed),
                            num_workers=execution.workers, generator=torch.Generator().manual_seed(seed))
        best = None
        with progress_bar(loader, total=recipe["updates"], desc=f"{task['id']} lr={rate:g} wd={decay:g}", unit="update", enabled=execution.progress) as bar:
            for step, (images, valid, transforms, labels, ids) in enumerate(bar):
                head.train()
                features = []
                for offset in range(0, len(ids), execution.batch_size):
                    end = offset + execution.batch_size
                    features.append(encoder(images[offset:end], valid[offset:end], transforms[offset:end], ids[offset:end]))
                values = torch.cat([item["values"] for item in features]).float()
                geometry = torch.cat([item["grid_to_original"] for item in features])
                for group in optimizer.param_groups:
                    group["lr"] = rate * learning_rate_scale(step, recipe)
                optimizer.zero_grad(set_to_none=True)
                loss = backward_pixels(head(values), geometry, labels.to(execution.device), transforms.to(execution.device))
                optimizer.step()
                if (step + 1) % recipe["evaluation_interval"] == 0 or step + 1 == recipe["updates"]:
                    measured, _ = score(encoder, head, development, root, task, seed, progress=execution.progress)
                    if best is None or measured["score"] > best["development_score"]:
                        best = {"seed": seed, "updates": step + 1, "learning_rate": rate, "weight_decay": decay,
                                "development_score": measured["score"], "training_loss": loss,
                                "probe_parameters": sum(value.numel() for value in head.parameters()), "head_file": destination.name}
                        save_tensors(destination, {name: value.detach().cpu() for name, value in head.state_dict().items()}, storage=storage)
        best["head_sha256"] = file_digest(destination)
        return best


def preflight(model, execution, root, row, task):
    encoder = Encoder(model, execution)
    pixels = evaluation_image(root, row, task)
    box = next(window_boxes(*pixels.shape[-2:], task["canvas"], task["evaluation_stride"]))
    images, valid, transform = window_input(pixels, box)
    encoder(images, valid, transform, [row["id"]])
    return encoder


def run_probe(model, execution, root, training, development, scoring, task, directory, storage, evaluation, encoder):
    started = time.monotonic()
    selection = read_selection(directory)
    reused = selection is not None
    if selection is None:
        candidates, winner = [], None
        for path in directory.glob("*.pt"):
            remove(path, storage=storage)
        for index, (rate, decay) in enumerate(product(task["probe"]["learning_rates"], task["probe"]["weight_decays"])):
            candidate = train_candidate(encoder, training, development, root, task, encoder.shape[0], rate, decay, directory / f"candidate-{index}.pt", storage)
            candidates.append(candidate)
            if winner is None or (-candidate["development_score"], candidate["updates"]) < (-winner["development_score"], winner["updates"]):
                if winner is not None:
                    remove(directory / winner["head_file"], storage=storage)
                winner = candidate
            else:
                remove(directory / candidate["head_file"], storage=storage)
        selection = {"identity": directory.name, "shape": encoder.shape, "dtype": encoder.dtype,
                     "normalization_sha256": None, "normalization": "trained-batchnorm-in-head",
                     "seeds": [winner], "candidates": candidates}
        selection["digest"] = digest(selection)
        write_json(directory / "selection.json", selection, storage=storage)
    selection_seconds = time.monotonic() - started
    fitting_usage = encoder.report()
    encoder = Encoder(model, execution)
    seed = selection["seeds"][0]
    with adapter_mode(model):
        head = LinearHead(selection["shape"][0], task["classes"], task["probe"]["dropout"], execution.device)
        head.load_state_dict(torch.load(directory / seed["head_file"], weights_only=True, map_location=execution.device))
        measured, records = score(encoder, head, scoring, root, task, seed["seed"], progress=execution.progress)
    scoring_usage = encoder.report()
    details = {"probe_status": "reused" if reused else "regenerated" if evaluation == "final" else "fitted",
               "probe_fitting_seconds": 0.0 if reused else selection_seconds, "selection_seconds": selection_seconds,
               "selection": {**{key: value for key, value in selection.items() if key not in ("seeds", "candidates")},
                             **{key: [{k: v for k, v in item.items() if k != "head_file"} for item in selection[key]] for key in ("seeds", "candidates")}},
               "seed_results": [measured], "sample_count": len(scoring), "feature_shape": selection["shape"], "feature_dtype": selection["dtype"],
               "diagnostics": {"status": "not_computed", "reason": "online augmented features; no fixed training bank"},
               "extraction_costs": scoring_usage["costs"], "extraction_seconds": fitting_usage["total_seconds"] + scoring_usage["total_seconds"],
               "extraction_usage": {"extracted_examples": fitting_usage["extracted_examples"] + scoring_usage["extracted_examples"], "reused_examples": 0},
               "extraction_work": [{"scope": "preflight-and-selection", **fitting_usage}, {"scope": "scoring", **scoring_usage}]}
    return {"miou": measured["score"]}, records, details
