"""Measure ADE-sized probe kernels on synthetic features, without task scores."""

import argparse
from pathlib import Path
import resource
from statistics import median
import tempfile
import time

import torch
from torch.nn import functional as F

from bench import Execution
from bench.features import numerical_mode, save_tensors
from bench.geometry import logits_on_canvas
from bench.probes import LinearProbe
from bench.protocols import resolve_names
from bench.storage import TemporaryStorage, file_digest, remove, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("This diagnostic requires CUDA and at least 10 GiB of available GPU memory")
    if torch.cuda.mem_get_info()[0] < 10 * 1024 ** 3:
        parser.error("The fixed ADE-sized diagnostic needs at least 10 GiB of available GPU memory")
    task = resolve_names(["ade20k"])[0]
    count, canvas, classes = task["probe"]["batch_size"], task["canvas"], task["classes"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    generator = torch.Generator().manual_seed(19)
    labels = torch.randint(classes, (count, canvas, canvas), generator=generator)
    for index, target in enumerate(labels):
        target[:index * 13] = 255
    denominator = int((labels != 255).sum())
    axis = (torch.arange(canvas, device="cuda") + 0.5) / canvas * 2 - 1
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    sampling_grid = torch.stack([x, y], -1)[None]

    def measured(function):
        torch.cuda.synchronize()
        begin = time.monotonic()
        result = function()
        torch.cuda.synchronize()
        return result, time.monotonic() - begin

    report = {"scope": "synthetic dense-kernel diagnostic; no benchmark task scores or calibration claim", "task_recipe_digest": task["digest"], "feature_shape": [768, 32, 32], "feature_dtype": "bfloat16", "canvas": canvas, "classes": classes, "effective_batch": count, "eligible_pixels": denominator, "hardware": torch.cuda.get_device_name(), "torch": str(torch.__version__), "modes": {}}
    report["timing_scope"] = "first effective batch with synchronized phase timers, then two warmup and five measured updates of the same synthetic batch without phase barriers"
    report["script_sha256"] = file_digest(__file__)
    with tempfile.TemporaryDirectory(prefix=".dense-probe-", dir=args.output.parent) as temporary, numerical_mode(Execution(device="cuda", numerics="fp32-v1")):
        work = Path(temporary)
        storage = TemporaryStorage(work)
        write_started = time.monotonic()
        paths = [work / "banks" / f"{index}.pt" for index in range(count)]
        for path in paths:
            save_tensors(path, {"values": torch.randn(768, 32, 32, generator=generator).bfloat16()}, storage=storage)
        report["feature_generate_and_write_seconds"] = time.monotonic() - write_started
        read_started = time.monotonic()
        values = torch.stack([torch.load(path, mmap=True, weights_only=True)["values"] for path in paths])
        report["feature_read_seconds"] = time.monotonic() - read_started
        report["read_scope"] = "new mappings of recently written files; OS page cache may be warm"
        heads, gradients = {}, {}
        for name, microbatch in [("full_effective_batch", count), ("one_image_microbatches", 1)]:
            probe = LinearProbe(768, classes, 0, "cuda")
            recipe = task["probe"]
            optimizer = torch.optim.AdamW(probe.parameters(), lr=recipe["learning_rates"][0], weight_decay=recipe["weight_decays"][0], betas=tuple(recipe["betas"]), eps=recipe["epsilon"], amsgrad=recipe["amsgrad"], foreach=recipe["foreach"], fused=recipe["fused"])
            torch.cuda.reset_peak_memory_stats()

            def update(phase_timers):
                def phase(function):
                    return measured(function) if phase_timers else (function(), 0.0)

                optimizer.zero_grad(set_to_none=True)
                totals = {"transfer_seconds": 0.0, "forward_resample_loss_seconds": 0.0, "backward_seconds": 0.0, "loss": 0.0}
                for offset in range(0, count, microbatch):
                    (features, targets), seconds = phase(lambda: (values[offset:offset + microbatch].float().cuda(), labels[offset:offset + microbatch].cuda()))
                    totals["transfer_seconds"] += seconds

                    def forward():
                        native = probe(features)
                        if microbatch == 1:
                            logits = logits_on_canvas(native[0], torch.eye(3), torch.eye(3), (canvas, canvas))[None]
                        else:
                            logits = F.grid_sample(native, sampling_grid.expand(len(native), -1, -1, -1), mode="bilinear", padding_mode="border", align_corners=False)
                        return F.cross_entropy(logits, targets, ignore_index=255, reduction="sum") / denominator

                    loss, seconds = phase(forward)
                    totals["forward_resample_loss_seconds"] += seconds
                    _, seconds = phase(loss.backward)
                    totals["backward_seconds"] += seconds
                    totals["loss"] += float(loss.detach())
                    del loss, features, targets
                _, totals["optimizer_seconds"] = phase(optimizer.step)
                return totals

            totals = update(True)
            gradients[name] = {key: parameter.grad.detach().cpu().clone() for key, parameter in probe.named_parameters()}
            totals["peak_gpu_bytes"] = torch.cuda.max_memory_allocated()
            heads[name] = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}
            durations = []
            for index in range(7):
                _, seconds = measured(lambda: update(False))
                if index >= 2:
                    durations.append(seconds)
            totals["steady_state"] = {"warmup_updates": 2, "measured_updates": 5, "effective_batch_seconds": durations, "median_effective_batch_seconds": median(durations), "images_per_second": count / median(durations), "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
            report["modes"][name] = totals
            del optimizer, probe
        for key in gradients["full_effective_batch"]:
            torch.testing.assert_close(gradients["full_effective_batch"][key], gradients["one_image_microbatches"][key], rtol=2e-4, atol=2e-7)
        first, second = [LinearProbe(768, classes, 0, "cuda") for _ in range(2)]
        first.load_state_dict(heads["full_effective_batch"])
        second.load_state_dict(heads["one_image_microbatches"])
        disagreements = 0
        scoring_started = time.monotonic()
        with torch.no_grad():
            for value, target in zip(values, labels):
                features = value.float().cuda()[None]
                predictions = [logits_on_canvas(head(features)[0], torch.eye(3), torch.eye(3), (canvas, canvas)).argmax(0).cpu() for head in (first, second)]
                disagreements += int(((predictions[0] != predictions[1]) & (target != 255)).sum())
        report["prediction_comparison_seconds"] = time.monotonic() - scoring_started
        report["gradient_parity"] = {"rtol": 2e-4, "atol": 2e-7, "passed": True}
        report["prediction_disagreement_pixels"] = disagreements
        report["prediction_disagreement_fraction"] = disagreements / denominator
        cleanup_started = time.monotonic()
        remove(work / "banks", storage=storage)
        report["cleanup_seconds"] = time.monotonic() - cleanup_started
        report["storage"] = {"peak_temporary_bytes": storage.peak, "retained_temporary_bytes": storage.current}
    report["total_seconds"] = time.monotonic() - started
    report["peak_host_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    write_json(args.output, report)
    print(f"Dense diagnostic: {args.output}; gradient agreement passed; {disagreements}/{denominator} predicted pixels differ")


if __name__ == "__main__":
    main()
