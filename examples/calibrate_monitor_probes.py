"""Measure probe budgets, feature scales and learning rates under calibration IDs."""

import argparse
from copy import deepcopy
import math
import os
from pathlib import Path
import platform
import resource
import time

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

import bench
from bench.features import build_bank, feature_diagnostics, model_identity, numerical_mode
from bench.prepare import build_manifest, load_prepared
from bench.probes import evaluate_selection, fit_probe, probe_identity
from bench.protocols import resolve_names
from bench.storage import TemporaryStorage, digest, file_digest, lock, read_json, remove, resolve_path, write_json
from bench.workspace import clear, initialize, workspace_lock


class ScaledBank:
    def __init__(self, bank, factor):
        self.bank, self.factor = bank, factor
        self.manifest, self.storage = bank.manifest, bank.storage

    def __getitem__(self, sample):
        feature = self.bank[sample]
        return {**feature, "values": feature["values"].float() * self.factor}

    def read_values(self, sample_ids):
        return self.bank.read_values(sample_ids).float() * self.factor


def measure(model, *, names, data_root, output, execution, stage="budget-scale"):
    if stage not in ("budget-scale", "variance-floor", "learning-rate", "convergence", "fuller-reference", "fuller-convergence", "fuller-selection", "search-boundary", "search-extension", "search-weight-decay", "search-grid"):
        raise ValueError("Choose the budget-scale, variance-floor, learning-rate, convergence, fuller-reference, fuller-convergence, fuller-selection, search-boundary, search-extension, search-weight-decay or search-grid calibration stage")
    tasks = resolve_names(names)
    weight_search = stage in ("search-weight-decay", "search-grid")
    full_probe = stage in ("search-boundary", "search-extension") or weight_search
    scale_check = stage in ("budget-scale", "variance-floor")
    long_reference = stage in ("fuller-convergence", "fuller-selection")
    allowed = ("imagenet-1k", "ade20k", "clevr-count", "clevr-distance") if full_probe else ("imagenet-100", "pascal")
    if any(task["name"] not in allowed for task in tasks):
        raise ValueError(f"The {stage} stage supports only {', '.join(allowed)}")
    if weight_search and any(len(task["probe"]["weight_decays"]) < 2 or max(task["probe"]["weight_decays"]) <= 0 for task in tasks):
        raise ValueError("Weight-decay calibration requires a searched grid with a positive upper bound")
    if any(task["method"] not in model.info.readouts for task in tasks):
        raise ValueError("The model must support every requested calibration readout")
    output, data_root = resolve_path(output), resolve_path(data_root)
    preparation_started = time.monotonic()
    reference_tasks = {task["id"]: prepare_fuller_reference(data_root, task) for task in tasks} if stage == "fuller-reference" or long_reference else {}
    preparation_seconds = time.monotonic() - preparation_started
    work = output / ".work"
    started = time.monotonic()
    with lock(output.parent / f".{output.name}.bench-output-lock"), workspace_lock(work), numerical_mode(execution):
        if output.exists() and any(path != work.parent / f".{work.name}.bench-lock" for path in output.iterdir()):
            raise ValueError("Choose a fresh calibration output; prune abandoned work explicitly")
        metadata = model_identity(model, execution)
        initialize(work, metadata["digest"])
        storage = TemporaryStorage(work)
        plan = {"model": metadata, "script_sha256": file_digest(__file__), "data_root": str(data_root), "numerics": execution.numerics, "stage": stage, "variants": [], "scope": "exploratory development controls; not reserved-checkpoint monitoring acceptance"}
        source_root = Path(bench.__file__).parent
        plan["software"] = {str(path.relative_to(source_root)): file_digest(path) for path in source_root.rglob("*.py")}
        plan["hardware"] = {"platform": platform.platform(), "logical_cpus": os.cpu_count(), "host_memory_bytes": os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"), "gpu": torch.cuda.get_device_name(execution.device), "cuda_runtime": torch.version.cuda}
        plan["execution"] = {"device": execution.device, "batch_size": execution.batch_size, "workers": execution.workers}
        if reference_tasks:
            plan["reference_preparation"] = reference_tasks
            plan["scope"] = "larger training-bank development budget diagnostic; budget validation and reserved-checkpoint agreement remain required"
            plan["preparation_seconds_excluded_from_evaluation"] = preparation_seconds
        rates = (1, 3, 10, 30, 100) if stage == "search-extension" else (1, 3, 10) if full_probe else (1, 3, 10, 30)
        budgets = (2, 4) if stage == "search-extension" else (4, 8) if stage in ("convergence", "fuller-reference") else (1, 2)
        cases = [(1, 1.0, 1), (2, 1.0, 1), (1, 0.1, 1), (1, 10.0, 1)] if scale_check else [(budget, 1.0, rate) for rate in rates for budget in budgets]
        if weight_search:
            cases = [(budget, 1.0, 1) for budget in budgets]
        for task in tasks:
            budget_key = "epochs" if full_probe else "updates"
            base_rates = [max(task["probe"]["learning_rates"])] if full_probe and not weight_search else task["probe"]["learning_rates"]
            if long_reference:
                prepared, _ = load_prepared(data_root, reference_tasks[task["id"]], "train", verify=False)
                reference_epochs = 50 if task["kind"] == "classification" else 40
                reference_updates = reference_epochs * math.ceil(prepared["splits"]["train"]["count"] / task["probe"]["batch_size"])
            for multiplier, scale, rate_multiplier in cases:
                variant = deepcopy(reference_tasks.get(task["id"], task))
                variant["probe"][budget_key] = (reference_updates if long_reference else variant["probe"][budget_key]) * multiplier
                variant["probe"]["learning_rates"] = [rate * rate_multiplier for rate in base_rates]
                variant["probe"]["seeds"] = [0]
                variant["id"] = f"{variant['id']}-calibration-{budget_key[0]}{variant['probe'][budget_key]}-s{scale:g}"
                if not scale_check:
                    variant["id"] += f"-lrm{rate_multiplier}"
                if weight_search:
                    variant["probe"]["weight_decays"] += [max(task["probe"]["weight_decays"]) * 10]
                    variant["id"] += "-wdm10"
                if stage == "search-grid":
                    variant["probe"]["learning_rates"] += [max(base_rates) * extension for extension in (3, 10)]
                    variant["id"] += "-joint"
                if stage == "variance-floor":
                    variant["standardization"]["variance_floor"] = 1e-8
                    variant["id"] += "-floor1e-08"
                if stage == "fuller-selection":
                    variant["probe"].update(selection="best-development-score-then-earlier-epoch-then-candidate-order", scoring="each-epoch")
                    variant["id"] += "-bestdev"
                variant["name"], variant["release_status"] = variant["id"], "calibration-only"
                variant["calibration"] = {"base_task": task["id"], "base_task_digest": task["digest"], "feature_scale_before_standardization": scale, "scale_arithmetic": "float32", "budget_multiplier": multiplier, "learning_rate_multiplier": rate_multiplier, "learning_rate_reference": base_rates}
                if long_reference:
                    variant["calibration"]["reference_epochs"] = reference_epochs * multiplier
                if stage == "fuller-selection":
                    variant["calibration"]["selection_reference"] = task["probe"]["selection"]
                if stage == "variance-floor":
                    variant["calibration"]["variance_floor_reference"] = task["standardization"]["variance_floor"]
                if weight_search:
                    variant["calibration"]["weight_decay_reference"] = task["probe"]["weight_decays"]
                    variant["calibration"]["weight_decay_multiplier"] = 10
                if stage == "search-grid":
                    variant["calibration"]["learning_rate_extension_multipliers"] = [3, 10]
                variant["digest"] = digest({key: value for key, value in variant.items() if key != "digest"})
                plan["variants"].append(variant)
        plan["gates"] = {"budget_absolute_percentage_points": {"imagenet100-linear-v1": 0.5, "voc2012-linear-v1": 0.25}, "scale_absolute_percentage_points": {task["id"]: task["numerical_tolerance"]["absolute_percentage_points"] for task in tasks}}
        if full_probe:
            plan["gates"]["budget_absolute_percentage_points"] = {}
            plan["scope"] = "development search coverage and budget diagnostic with seed zero; monitoring acceptance thresholds do not certify full probes"
        write_json(output / "plan.json", plan)
        report = {"status": "running", "plan_sha256": file_digest(output / "plan.json"), "results": {}, "comparisons": {}}
        write_json(output / "calibration.json", report)
        torch.cuda.reset_peak_memory_stats(execution.device)
        bank = None
        try:
            for task in tasks:
                preparation_task = reference_tasks.get(task["id"], task)
                prepared, training = load_prepared(data_root, preparation_task, "train")
                _, development = load_prepared(data_root, preparation_task, "development")
                source = Path(prepared["source_root"])
                bank, usage = build_bank(model, model_metadata=metadata, task=task, root=source, rows=training + development, work_dir=work, execution=execution, storage=storage)
                diagnostics = feature_diagnostics(bank, prepared["diagnostic_sample_ids"])
                variants = [variant for variant in plan["variants"] if variant["calibration"]["base_task"] == task["id"]]
                scores = {}
                for variant in variants:
                    begin = time.monotonic()
                    scale = variant["calibration"]["feature_scale_before_standardization"]
                    scaled = bank if scale == 1.0 else ScaledBank(bank, scale)
                    directory = work / "probes" / probe_identity(variant, prepared, metadata)
                    selection = fit_probe(scaled, training=training, development=development, root=source, task=variant, directory=directory, execution=execution, diagnostics={"raw_encoder": diagnostics, "calibration": variant["calibration"]})
                    metrics, seeds, records = evaluate_selection(selection, directory, scaled, development, source, variant, execution)
                    statistics = output / "statistics" / f"{variant['id']}.json"
                    write_json(statistics, {"task": variant["id"], "evaluation": "development", "records": records})
                    report["results"][variant["id"]] = {"metrics": metrics, "seed_results": seeds, "selection": [{key: value for key, value in seed.items() if key != "head_file"} for seed in selection["seeds"]], "preparation_digest": prepared["digest"], "source_root": str(source), "sample_digests": prepared["splits"], "statistics": {"file": str(statistics.relative_to(output)), "sha256": file_digest(statistics)}, "fitting_and_scoring_seconds": time.monotonic() - begin}
                    report["results"][variant["id"]]["candidate_scores"] = [{key: value for key, value in candidate.items() if key != "head_file"} for candidate in selection["candidates"]]
                    calibration = variant["calibration"]
                    scores[calibration["budget_multiplier"], calibration["feature_scale_before_standardization"], calibration["learning_rate_multiplier"]] = metrics[task["metric"]]
                    write_json(output / "calibration.json", report)
                    remove(directory, storage=storage)
                    print(f"{variant['id']}: {metrics}", flush=True)
                budget_tolerance = plan["gates"]["budget_absolute_percentage_points"].get(task["id"])
                report["comparisons"][task["id"]] = {
                    "extraction": {key: value for key, value in usage.items() if key != "extracted_ids"},
                    "raw_encoder_diagnostics": diagnostics,
                }
                checks = {}
                for rate in sorted({case[2] for case in cases}):
                    base, doubled = scores[budgets[0], 1.0, rate], scores[budgets[1], 1.0, rate]
                    checks[str(rate)] = {"base_score": base, "doubled_budget_score": doubled, "doubled_budget_difference": doubled - base}
                    if budget_tolerance is not None:
                        checks[str(rate)].update(budget_tolerance=budget_tolerance, budget_passed=abs(doubled - base) <= budget_tolerance)
                if scale_check:
                    tolerance = plan["gates"]["scale_absolute_percentage_points"][task["id"]]
                    differences = {str(scale): scores[1, scale, 1] - scores[1, 1.0, 1] for scale in (0.1, 10.0)}
                    report["comparisons"][task["id"]].update(checks["1"], scale_score_differences=differences, scale_tolerance=tolerance, scale_passed=all(abs(delta) <= tolerance for delta in differences.values()))
                else:
                    report["comparisons"][task["id"]]["learning_rate_checks"] = checks
                bank.close()
                remove(bank.directory, storage=storage)
                bank = None
                write_json(output / "calibration.json", report)
            clear(work, storage=storage)
            report.update(status="complete", total_seconds=time.monotonic() - started, peak_temporary_bytes=storage.peak, retained_temporary_bytes=storage.current, memory={"peak_gpu_bytes": torch.cuda.max_memory_allocated(execution.device), "peak_host_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, "host_peak_scope": "process-lifetime"})
            write_json(output / "calibration.json", report)
        except BaseException as error:
            if bank is not None:
                bank.close()
            report.update(status="failed", error=str(error), remaining_workspace=str(work))
            write_json(output / "calibration.json", report)
            raise
    return report


def prepare_fuller_reference(data_root, task):
    """Prepare a larger training bank once, preserving monitoring evaluation data."""
    data_root = resolve_path(data_root)
    reference = deepcopy(task)
    reference["id"] += "-fuller-bank-calibration-v1"
    reference["release_status"] = "calibration-only"
    reference["bank"].pop("train_per_class" if task["source"] == "imagenet" else "train")
    reference["reference_bank"] = {"base_task": task["id"], "base_task_digest": task["digest"], "training": "all-official-training-excluding-monitor-development"}
    reference["digest"] = digest({key: value for key, value in reference.items() if key != "digest"})
    with lock(data_root / ".bench" / "locks" / f"{task['source']}.lock"):
        original, original_training = load_prepared(data_root, task, "train")
        load_prepared(data_root, task, "development")
        directory = data_root / ".bench" / "prepared" / reference["id"]
        if not (directory / "manifest.json").exists():
            inventory_path = data_root / ".bench" / "sources" / f"{task['source']}.json"
            inventory = read_json(inventory_path)
            build_manifest(data_root, Path(original["source_root"]), reference, original["source"], inventory["files"])
            write_json(inventory_path, inventory)
        prepared, training = load_prepared(data_root, reference, "train")
        load_prepared(data_root, reference, "development")
        if prepared["labels"] != original["labels"] or any(prepared["splits"][split] != original["splits"][split] for split in ("development", "final")):
            raise ValueError("The fuller reference must preserve the monitoring labels and evaluation partitions")
        original_rows = {row["id"]: row for row in original_training}
        expanded_rows = {row["id"]: row for row in training}
        if len(training) <= len(original_training) or any(expanded_rows.get(key) != row for key, row in original_rows.items()):
            raise ValueError("The fuller reference requires a strict superset of the monitoring training bank")
    return reference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True, help="Hydra constructor YAML for one fixed adapter")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--names", nargs="+", choices=["imagenet-100", "pascal", "imagenet-1k", "ade20k", "clevr-count", "clevr-distance"], default=["imagenet-100", "pascal"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--stage", choices=["budget-scale", "variance-floor", "learning-rate", "convergence", "fuller-reference", "fuller-convergence", "fuller-selection", "search-boundary", "search-extension", "search-weight-decay", "search-grid"], default="budget-scale")
    args = parser.parse_args()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = instantiate(OmegaConf.load(args.model), _convert_="all")
    model.cuda()
    measure(model, names=args.names, data_root=args.data_root, output=args.output, execution=bench.Execution(device="cuda", batch_size=args.batch_size, numerics="fp32-v1"), stage=args.stage)


if __name__ == "__main__":
    main()
