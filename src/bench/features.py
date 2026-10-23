"""Validated adapter execution and disposable, verified tensor shards."""

from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
from PIL import Image, __version__ as pillow_version
from safetensors import __version__ as safetensors_version
from safetensors.torch import load_file, save_file
import torch
from torch.utils.data import DataLoader, Dataset

from .api import Extraction, ImageBatch, SpatialFeatures
from .images import make_canvas
from .storage import atomic_file, atomic_path, contained_path, digest, file_digest, read_json, remove, write_json


def model_identity(model, execution):
    from . import images

    info = asdict(model.info)
    if not info["name"] or not info["revision"] or not info["readouts"]:
        raise ValueError("ModelInfo must declare a name, source revision, and at least one readout")
    if set(info["readouts"]) - {"global", "spatial"} or any(not isinstance(value, str) or not value for value in info["readouts"].values()):
        raise ValueError("ModelInfo.readouts must map global and/or spatial to nonempty readout names")
    digest(info)  # Reject unserializable metadata, including nonfinite numbers.
    state = hashlib.sha256()
    tensors = [("parameter:" + name, tensor) for name, tensor in model.named_parameters()]
    tensors += [("buffer:" + name, tensor) for name, tensor in model.named_buffers()]
    device = torch.device(execution.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    for name, tensor in sorted(tensors):
        if tensor.device.type != device.type or device.index is not None and tensor.device.index != device.index:
            raise ValueError(f"{name} is on {tensor.device}; the caller must place the model on {device}")
        state.update(digest([name, list(tensor.shape), str(tensor.dtype)]).encode())
        state.update(tensor.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
    metadata = {"info": info, "state_digest": state.hexdigest(), "numerics": execution.numerics, "torch": str(torch.__version__), "numpy": np.__version__, "pillow": pillow_version, "safetensors": safetensors_version, "preprocessing_implementation_sha256": file_digest(images.__file__), "extraction_implementation_sha256": file_digest(__file__)}
    metadata["digest"] = digest(metadata)
    return metadata


def validate_execution(execution):
    if execution.batch_size < 1 or execution.workers < 0:
        raise ValueError("Execution.batch_size must be positive and workers must be nonnegative")
    device = torch.device(execution.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("v1 supports CPU or CUDA execution")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use Execution(device='cpu', numerics='fp32-v1') or a CUDA host")
    if execution.numerics not in ("bf16-v1", "fp32-v1"):
        raise ValueError("Choose numerical profile bf16-v1 or fp32-v1")
    if execution.numerics == "bf16-v1":
        supported = False
        if device.type == "cuda":
            with torch.cuda.device(device):
                supported = torch.cuda.is_bf16_supported()
        if not supported:
            raise ValueError("bf16-v1 requires a supported CUDA GPU; choose numerics='fp32-v1'")


@contextmanager
def numerical_mode(execution):
    device = torch.device(execution.device)
    precision = torch.get_float32_matmul_precision()
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        with (torch.cuda.device(device) if device.type == "cuda" else nullcontext()), torch.inference_mode(False), torch.enable_grad(), torch.autocast(device.type, enabled=False):
            yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


@contextmanager
def adapter_mode(model):
    flags = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            yield
    finally:
        for module, training in flags:
            module.training = training
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def validate_affine(matrix, batch_size, name):
    if matrix.shape != (batch_size, 3, 3) or not matrix.is_floating_point() or not torch.isfinite(matrix).all():
        raise ValueError(f"{name} must be finite floating-point [B, 3, 3]")
    expected = matrix.new_tensor([0, 0, 1]).expand(batch_size, -1)
    if not torch.allclose(matrix[:, 2], expected) or (torch.linalg.det(matrix.float()).abs() < 1e-12).any():
        raise ValueError(f"{name} must contain invertible homogeneous affine matrices")


def supplied_extent(batch):
    extents = []
    height, width = batch.images.shape[-2:]
    with torch.autocast(batch.images.device.type, enabled=False):
        for valid, transform in zip(batch.valid_pixels, batch.image_to_original):
            ys, xs = valid[0].nonzero(as_tuple=True)
            if not len(xs):
                raise ValueError("An ImageBatch entry has no valid image content")
            left, right = xs.min() / width, (xs.max() + 1) / width
            top, bottom = ys.min() / height, (ys.max() + 1) / height
            corners = torch.stack([torch.stack([x, y, x.new_ones(())]) for x, y in [(left, top), (left, bottom), (right, top), (right, bottom)]])
            points = corners.float() @ transform.float().T
            extents.append(torch.cat([points[:, :2].min(0).values, points[:, :2].max(0).values]))
    return torch.stack(extents)


def extract(model, batch, method, execution):
    with adapter_mode(model):
        autocast = torch.autocast(torch.device(execution.device).type, dtype=torch.bfloat16, enabled=execution.numerics == "bf16-v1")
        with torch.inference_mode(), autocast:
            output = getattr(model, "encode_" + method)(batch)
        if not isinstance(output, Extraction):
            raise ValueError(f"encode_{method} must return Extraction")
        feature = output.features
        if method == "global":
            if not isinstance(feature, torch.Tensor) or feature.ndim != 2:
                raise ValueError("encode_global must return finite [B, D] features, not logits or a spatial readout")
            values = feature
        else:
            if not isinstance(feature, SpatialFeatures) or feature.values.ndim != 4:
                raise ValueError("encode_spatial must return SpatialFeatures with [B, D, Hf, Wf] values")
            values = feature.values
        batch_size = len(batch.sample_ids)
        if values.shape[0] != batch_size or any(size < 1 for size in values.shape) or not values.is_floating_point() or not torch.isfinite(values).all():
            raise ValueError(f"Invalid encode_{method} feature shape or nonfinite values")
        # Clone outside inference mode: detached inference tensors cannot train a head.
        result = {"values": values.detach().cpu().clone()}
        if method == "spatial":
            validate_affine(feature.grid_to_original, batch_size, "grid_to_original")
            extent = feature.extent_in_original
            if extent.shape != (batch_size, 4) or not extent.is_floating_point() or not torch.isfinite(extent).all() or (extent[:, :2] >= extent[:, 2:]).any():
                raise ValueError("extent_in_original must be finite [B, 4] rectangles with positive width and height")
            required = supplied_extent(batch).to(extent)
            if (extent[:, :2] > required[:, :2] + 1e-5).any() or (extent[:, 2:] < required[:, 2:] - 1e-5).any():
                raise ValueError("Spatial output extent does not cover all supplied image content; cropped readouts are unsupported")
            result["grid_to_original"] = feature.grid_to_original.detach().cpu().clone()
            result["extent_in_original"] = extent.detach().cpu().clone()
        result["costs"] = {}
        for name, cost in (output.costs or {}).items():
            if not name.endswith(("_count", "_fraction", "_seconds", "_bytes", "_joules", "_flops")):
                raise ValueError(f"Extraction cost must have a unit-qualified name: {name}")
            if cost.shape != (batch_size,) or not torch.isfinite(cost).all() or (cost < 0).any():
                raise ValueError(f"Extraction cost {name} must be nonnegative finite [B]")
            if name.endswith("_fraction") and (cost > 1).any():
                raise ValueError(f"Extraction fraction {name} exceeds one")
            result["costs"][name] = cost.detach().cpu().clone()
        return result


class ImageDataset(Dataset):
    def __init__(self, root, rows, task):
        self.root, self.rows, self.task = root, rows, task

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(contained_path(self.root, row["image"])) as image:
            canvas = make_canvas(image, self.task)
        return (torch.from_numpy(np.array(canvas.image)).permute(2, 0, 1).float() / 255, torch.from_numpy(canvas.valid[None]), torch.from_numpy(canvas.to_original), row["id"])


def collate_images(entries):
    images, valid, transforms, ids = zip(*entries)
    return torch.stack(images), torch.stack(valid), torch.stack(transforms), tuple(ids)


def save_tensors(path, value, *, storage=None):
    with atomic_file(path, storage=storage) as stream:
        torch.save(value, stream)


def save_features(path, tensors, *, storage=None):
    flattened = {name: value.contiguous() for name, value in tensors.items() if name != "costs"}
    flattened.update({"costs." + name: value.contiguous() for name, value in tensors["costs"].items()})
    with atomic_path(path, storage=storage) as temporary:
        save_file(flattened, temporary)


class FeatureBank:
    def __init__(self, directory, manifest, storage=None):
        self.directory, self.manifest = Path(directory), manifest
        self.storage = storage
        self.locations = {sample_id: (shard["file"], index) for shard in manifest["shards"] for index, sample_id in enumerate(shard["ids"])}
        self._mapped = OrderedDict()
        self._mapped_bytes = 0
        self._mapping_bytes_limit, self._mapping_count_limit = 256 * 1024 ** 2, 1024
        if len(manifest["shape"]) == 1:
            try:
                available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
            except (OSError, StopIteration):
                available = 0
            self._mapping_bytes_limit = max(self._mapping_bytes_limit, min(4 * 1024 ** 3, available // 8))
            if self._mapping_bytes_limit > 256 * 1024 ** 2:
                self._mapping_count_limit = 16384

    def __getitem__(self, sample_id):
        filename, index = self.locations[sample_id]
        tensors = self._read_shard(filename)
        result = {key: value[index] for key, value in tensors.items() if key != "costs"}
        result["costs"] = {name: value[index] for name, value in tensors["costs"].items()}
        return result

    def read_values(self, sample_ids):
        values = []
        for sample_id in sample_ids:
            filename, index = self.locations[sample_id]
            values.append(self._read_shard(filename)["values"][index])
        return torch.stack(values)

    def _read_shard(self, filename):
        if filename in self._mapped:
            tensors, size = self._mapped.pop(filename)
            self._mapped[filename] = tensors, size
        else:
            path = self.directory / filename
            size = path.stat().st_size
            while self._mapped and (self._mapped_bytes + size > self._mapping_bytes_limit or len(self._mapped) >= self._mapping_count_limit):
                _, (_, removed_size) = self._mapped.popitem(last=False)
                self._mapped_bytes -= removed_size
            flattened = load_file(path, device="cpu")
            tensors = {name: value for name, value in flattened.items() if not name.startswith("costs.")}
            tensors["costs"] = {name.removeprefix("costs."): value for name, value in flattened.items() if name.startswith("costs.")}
            if size <= self._mapping_bytes_limit:
                self._mapped[filename] = tensors, size
                self._mapped_bytes += size
        return tensors

    def close(self):
        self._mapped.clear()
        self._mapped_bytes = 0


def feature_identity(model_metadata, task, rows):
    from . import images

    transforms = {key: task[key] for key in ("canvas", "transform", "resize_short", "image_interpolation") if key in task}
    return {"model": model_metadata["digest"], "source": task["source"], "method": task["method"], "preprocessing": transforms, "preprocessing_implementation_sha256": file_digest(images.__file__), "pillow": pillow_version, "samples": [[row["id"], row["files"][row["image"]]] for row in rows]}


def storage_estimate(manifest, sample_count, task, batch_size):
    # Include indexed metadata, tensor serialization, and one atomic shard copy.
    features = manifest["bytes_per_sample"] * sample_count + 512 * sample_count + 4096 * math.ceil(sample_count / batch_size)
    atomic_shard = (manifest["bytes_per_sample"] + 512) * min(sample_count, batch_size) + 4096
    probe_bytes = 0
    if recipe := task.get("probe"):
        seeds, candidates = len(recipe["seeds"]), len(recipe["learning_rates"]) * len(recipe["weight_decays"])
        heads = (task["classes"] * (manifest["shape"][0] + 1) * 4 + 4096) * (seeds + 2)
        diagnostics = 1024 + 128 * task["diagnostics"]["maximum_images"] + 256 * math.prod(manifest["shape"][1:])
        # Include tensor-container headers, compact search/selection JSON and atomic copies.
        probe_bytes = heads + 2 * (8 * manifest["shape"][0] + 4096 + diagnostics + 1024 * (candidates + seeds + 2))
    return {"features_bytes": features, "atomic_shard_bytes": atomic_shard, "probe_bytes": probe_bytes, "total_bytes": features + atomic_shard + probe_bytes}


def build_bank(model, *, model_metadata, task, root, rows, work_dir, execution, preflight=False, storage=None):
    started = time.monotonic()
    identity = feature_identity(model_metadata, task, rows)
    context = {key: value for key, value in identity.items() if key != "samples"}
    directory = Path(work_dir) / "banks" / digest(context)
    path = directory / "manifest.json"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = read_json(path) if path.exists() else {"identity": context}
    if manifest["identity"] != context:
        raise ValueError(f"Feature bank identity mismatch: {path}")
    verified = []
    while (entry := directory / f"shard-{len(verified):06d}.json").exists():
        shard = read_json(entry)
        tensor_path = contained_path(directory, shard["file"])
        if not tensor_path.exists() or file_digest(tensor_path) != shard["sha256"]:
            break
        verified.append(shard)
    manifest["shards"] = verified
    known = {sample: checksum for shard in verified for sample, checksum in zip(shard["ids"], shard["sample_digests"])}
    requested = dict(identity["samples"])
    completed = {sample for sample, checksum in requested.items() if known.get(sample) == checksum}
    remaining = [row for row in rows if row["id"] not in completed]
    if "bytes_per_sample" in manifest and remaining and not preflight:
        needed = storage_estimate(manifest, len(remaining), task, execution.batch_size)["total_bytes"]
        if shutil.disk_usage(directory).free < needed:
            raise OSError(f"Insufficient temporary space for {task['id']}; need about {needed:,} additional bytes at {directory}")
    if preflight:
        remaining = [] if verified else remaining[:execution.batch_size]
    generator = torch.Generator().manual_seed(0)
    loader_options = {"prefetch_factor": 2} if execution.workers else {}
    loader = DataLoader(ImageDataset(root, remaining, task), batch_size=execution.batch_size, num_workers=execution.workers, collate_fn=collate_images, pin_memory=torch.device(execution.device).type == "cuda", generator=generator, **loader_options) if remaining else []
    current_examples = 0
    extracted_ids = []
    last_batch_finished = time.monotonic()
    timing = {"setup_seconds": last_batch_finished - started, "first_batch_seconds": 0.0, "first_batch_examples": 0, "steady_state_seconds": 0.0, "steady_state_examples": 0}
    for images, valid, transforms, ids in loader:
        device = execution.device
        batch = ImageBatch(images.to(device, non_blocking=True), valid.to(device, non_blocking=True), transforms.to(device, non_blocking=True), ids)
        tensors = extract(model, batch, task["method"], execution)
        shape, dtype = list(tensors["values"].shape[1:]), str(tensors["values"].dtype)
        if "shape" in manifest and (manifest["shape"] != shape or manifest["dtype"] != dtype):
            raise ValueError("Adapter feature width, grid, or dtype changed within a bank")
        if "bytes_per_sample" not in manifest:
            per_sample = sum(value[0].numel() * value.element_size() for key, value in tensors.items() if key != "costs")
            per_sample += sum(value[0].numel() * value.element_size() for value in tensors["costs"].values())
            manifest.update(shape=shape, dtype=dtype, bytes_per_sample=per_sample)
            estimate = storage_estimate(manifest, len(rows), task, execution.batch_size)
            print(f"{task['id']} {task['method']}: estimated temporary storage {estimate['total_bytes']:,} bytes", flush=True)
            if shutil.disk_usage(directory).free < estimate["total_bytes"]:
                raise OSError(f"Insufficient temporary space for {task['id']}; need about {estimate['total_bytes']:,} bytes at {directory}")
            write_json(path, {key: value for key, value in manifest.items() if key != "shards"}, storage=storage)
        filename = f"shard-{len(manifest['shards']):06d}.safetensors"
        save_features(directory / filename, tensors, storage=storage)
        shard = {"file": filename, "sha256": file_digest(directory / filename), "ids": list(ids), "sample_digests": [requested[sample] for sample in ids]}
        write_json((directory / filename).with_suffix(".json"), shard, storage=storage)
        manifest["shards"].append(shard)
        finished = time.monotonic()
        stage = "first_batch" if current_examples == 0 else "steady_state"
        timing[stage + "_seconds"] += finished - last_batch_finished
        timing[stage + "_examples"] += len(ids)
        last_batch_finished = finished
        current_examples += len(ids)
        extracted_ids.extend(ids)
    keep = {shard["file"] for shard in manifest["shards"]}
    keep |= {Path(filename).with_suffix(".json").name for filename in keep}
    for pattern in ("shard-*.safetensors", "shard-*.json"):
        for extra in directory.glob(pattern):
            if extra.name not in keep:
                remove(extra, storage=storage)
    timing["teardown_seconds"] = time.monotonic() - last_batch_finished
    timing["total_seconds"] = time.monotonic() - started
    return FeatureBank(directory, manifest, storage), {"extracted_examples": current_examples, "reused_examples": len(completed), "extracted_ids": extracted_ids, "timing": timing}


def feature_diagnostics(bank, sample_ids):
    values = torch.stack([bank[sample]["values"].float() for sample in sample_ids])

    def across_images(matrix):
        centered = matrix - matrix.mean(0)
        normalized = torch.nn.functional.normalize(matrix, dim=1)
        count = len(matrix)
        cosine = ((normalized.sum(0).square().sum() - normalized.square().sum()) / (count * (count - 1))).item() if count > 1 else None
        singular = torch.linalg.svdvals(centered)
        probabilities = singular / singular.sum() if singular.sum() else singular
        rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-30).log()).sum()).item() if singular.sum() else 0.0
        return {"rms_variation": centered.square().mean().sqrt().item(), "mean_pairwise_cosine": cosine, "effective_rank": rank}

    result = {"sample_ids": sample_ids, "sample_count": len(values), "feature_width": values.shape[1]}
    if values.ndim == 2:
        result.update(across_images(values))
    else:
        result["per_position"] = [across_images(values[:, :, y, x]) for y in range(values.shape[2]) for x in range(values.shape[3])]
        tokens = values.flatten(2).transpose(1, 2)
        normalized = torch.nn.functional.normalize(tokens, dim=2)
        count = tokens.shape[1]
        result["within_image_token_similarity"] = ((normalized.sum(1).square().sum(1) - normalized.square().sum((1, 2))) / (count * (count - 1))).tolist() if count > 1 else [None] * len(tokens)
    return result
