"""The common Python/CLI runner and durable partial-result commits."""

from contextlib import ExitStack
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import resource
import shutil
import time

from omegaconf import OmegaConf
import torch

from .api import Execution, RunResult, VisionModel
from .features import build_bank, feature_diagnostics, feature_identity, model_identity, numerical_mode, storage_estimate, validate_execution
from .prepare import load_prepared
from .probes import evaluate_selection, fit_probe, probe_identity, read_selection
from .protocols import evaluation_source_digest, resolve_names
from .storage import TemporaryStorage, atomic_write, bytes_used, contained_path, digest, file_digest, lock, read_json, remove, resolve_path, write_json
from .tasks.correspondence import distinct_images, evaluate_pairs
from .workspace import clear, initialize, workspace_lock


def read_result(output):
    result = read_json(output / "results.json")
    if result["schema_version"].split(".")[0] != "1":
        raise ValueError(f"Unsupported result schema: {result['schema_version']}")
    for task in result["tasks"].values():
        if task["status"] == "complete":
            statistics = contained_path(output, task["statistics"]["file"])
            if not statistics.exists() or file_digest(statistics) != task["statistics"]["sha256"]:
                raise ValueError(f"Missing or corrupt committed statistics: {statistics}")
    return result


def cost_summary(bank, rows):
    summaries = {}
    for row in rows:
        for name, value in bank[row["id"]]["costs"].items():
            value = float(value)
            summary = summaries.setdefault(name, {"count": 0, "sum": 0.0, "min": value, "max": value})
            summary["count"] += 1
            summary["sum"] += value
            summary["min"] = min(summary["min"], value)
            summary["max"] = max(summary["max"], value)
    for summary in summaries.values():
        summary["mean"] = summary["sum"] / summary["count"]
    return summaries or None


def run(model: VisionModel, *, names: list[str], evaluation="development", data_root: Path, output_dir: Path, execution: Execution | None = None) -> RunResult:
    started = time.monotonic()
    execution = execution or Execution()
    tasks = resolve_names(names)
    if evaluation not in ("development", "final"):
        raise ValueError("evaluation must be 'development' or 'final'")
    if not isinstance(model, VisionModel):
        raise TypeError("model must implement the bench.VisionModel adapter seam")
    unsupported = [task["name"] for task in tasks if task["method"] not in model.info.readouts]
    if unsupported:
        raise ValueError(f"Adapter lacks the requested output for {', '.join(unsupported)}; select compatible tasks explicitly")
    validate_execution(execution)
    root, output = resolve_path(data_root), resolve_path(output_dir)
    work = resolve_path(execution.work_dir) if execution.work_dir is not None else output / ".work"
    if work == output or output.is_relative_to(work) or root == work or root.is_relative_to(work):
        raise ValueError("work_dir must not contain the output directory or shared data root")
    with ExitStack() as stack:
        stack.enter_context(numerical_mode(execution))
        stack.enter_context(lock(output.parent / f".{output.name}.bench-output-lock"))
        stack.enter_context(workspace_lock(work))
        metadata = model_identity(model, execution)
        prepared = {}
        # Only training and development files are inspected before probe selection.
        # Final source data is opened after a completed selection exists.
        for task in tasks:
            if task["kind"] == "correspondence":
                manifest, rows = load_prepared(root, task, "development")
                _, training_pairs = load_prepared(root, task, "train")
                diagnostic_ids = set(manifest["diagnostic_sample_ids"])
                diagnostic_images = [row for row in distinct_images(training_pairs) if row["id"] in diagnostic_ids]
                prepared[task["id"]] = (manifest, diagnostic_images, rows)
            else:
                manifest, training = load_prepared(root, task, "train")
                _, development = load_prepared(root, task, "development")
                prepared[task["id"]] = (manifest, training, development)
        identity = {"model": metadata["digest"], "requested_names": names, "tasks": [{"id": task["id"], "digest": task["digest"], "preparation": prepared[task["id"]][0]["digest"], "evaluation_implementation": evaluation_source_digest(task)} for task in tasks], "evaluation": evaluation, "runner_implementation": file_digest(__file__)}
        identity_digest = digest(identity)
        owner_run_id = digest([identity_digest, str(output)])
        if (output / "manifest.json").exists():
            previous = read_json(output / "manifest.json")
            if previous["identity_digest"] != identity_digest:
                raise ValueError(f"Conflicting run identity at {output}; choose a distinct output_dir")
            result = read_result(output) if (output / "results.json").exists() else None
            if result and result["status"] == "complete":
                owner = read_json(work / "owner.json") if (work / "owner.json").exists() else None
                if owner and owner.get("run_id") == owner_run_id and (evaluation == "final" or execution.work_dir is None):
                    clear(work)
                return RunResult(result["schema_version"], "complete", result["metrics"], output)
        else:
            result = None
            workspace_lock_path = work.parent / f".{work.name}.bench-lock"
            if output.exists() and any(path != workspace_lock_path for path in output.iterdir()):
                raise ValueError(f"Output directory is nonempty and has no bench manifest: {output}")
        output.mkdir(parents=True, exist_ok=True)
        initialize(work, metadata["digest"], owner_run_id)
        resolved = {"names": names, "evaluation": evaluation, "data_root": str(root), "output_dir": str(output), "execution": {**asdict(execution), "work_dir": str(work)}, "tasks": tasks, "model": metadata}
        atomic_write(output / "resolved.yaml", OmegaConf.to_yaml(OmegaConf.create(resolved)).encode())
        write_json(output / "manifest.json", {"schema_version": "1.0.0", "identity": identity, "identity_digest": identity_digest, "model": metadata})
        if result is None:
            result = {"schema_version": "1.0.0", "status": "running", "requested_names": names, "resolved_tasks": [task["id"] for task in tasks], "evaluation": evaluation, "metrics": {}, "tasks": {}, "model": metadata, "exposure": {"pretraining": model.info.recipe.get("pretraining", "unknown"), "model_selection": model.info.recipe.get("model_selection", "unknown")}, "software": {"python": platform.python_version(), "torch": torch.__version__}, "hardware": {"device": execution.device, "gpu": torch.cuda.get_device_name(execution.device) if torch.device(execution.device).type == "cuda" else None}, "numerics": execution.numerics}
            try:
                package_version = version("bench")
            except PackageNotFoundError:
                package_version = "uninstalled-source-tree"
            implementation = {str(path.relative_to(Path(__file__).parent)): file_digest(path) for path in Path(__file__).parent.rglob("*.py")}
            result["software"].update(bench=package_version, numpy=metadata["numpy"], pillow=metadata["pillow"], safetensors=metadata["safetensors"], implementation_sha256=digest(implementation), source_files=implementation)
            result["hardware"].update(platform=platform.platform(), architecture=platform.machine(), logical_cpus=os.cpu_count(), host_memory_bytes=os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"), workspace_device_id=work.stat().st_dev)
            if torch.device(execution.device).type == "cuda":
                properties = torch.cuda.get_device_properties(execution.device)
                result["hardware"].update(gpu_memory_bytes=properties.total_memory, cuda_capability=[properties.major, properties.minor], cuda_runtime=torch.version.cuda)
        result["status"] = "running"
        result.pop("failure", None)
        write_json(output / "results.json", result)
        if torch.device(execution.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(execution.device)
        setup_seconds = time.monotonic() - started
        active_banks = {}
        storage = TemporaryStorage(work)
        first_result_seconds = 0.0 if result["metrics"] else None
        try:
            early_extractions = {}
            planned_banks, planned_probes = {}, 0
            for task in tasks:
                identifier = task["id"]
                if result["tasks"].get(identifier, {}).get("status") == "complete":
                    continue
                manifest, training, development = prepared[identifier]
                selection = read_selection(work / "probes" / probe_identity(task, manifest, metadata)) if task["kind"] != "correspondence" else None
                if task["kind"] == "correspondence":
                    _, rows = load_prepared(root, task, evaluation)
                    rows = sorted({row["id"]: row for row in distinct_images(rows) + training}.values(), key=lambda row: row["id"])
                elif selection is None:
                    rows = training + development
                else:
                    rows = development if evaluation == "development" else []
                result["tasks"][identifier] = {"status": "running", "evaluation": evaluation}
                if selection is None:
                    bank, usage = build_bank(model, model_metadata=metadata, task=task, root=Path(manifest["source_root"]), rows=rows, work_dir=work, execution=execution, preflight=True, storage=storage)
                    active_banks[bank.directory] = bank
                    shape_metadata = bank.manifest
                    if usage["extracted_examples"]:
                        early_extractions[bank.directory] = usage
                else:
                    shape_metadata = selection
                context = digest({key: value for key, value in feature_identity(metadata, task, []).items() if key != "samples"})
                plan = planned_banks.setdefault(context, {"ids": set(), "final_count": 0, "metadata": shape_metadata, "task": task})
                plan["ids"].update(row["id"] for row in rows)
                if evaluation == "final" and task["kind"] != "correspondence":
                    # Counts are available without opening held-out examples before selection.
                    plan["final_count"] += manifest["splits"]["final"]["count"]
                planned_probes += storage_estimate(shape_metadata, 0, task, execution.batch_size)["probe_bytes"]
            estimated_banks, atomic_overhead = 0, 0
            for plan in planned_banks.values():
                estimate = storage_estimate(plan["metadata"], len(plan["ids"]) + plan["final_count"], plan["task"], execution.batch_size)
                estimated_banks += estimate["features_bytes"]
                atomic_overhead = max(atomic_overhead, estimate["atomic_shard_bytes"])
            # Conservative bound: all requested banks and selected heads coexist.
            estimated_peak = estimated_banks + planned_probes + atomic_overhead
            additional = max(0, estimated_peak - storage.current)
            print(f"Run temporary storage upper estimate: {estimated_peak:,} bytes ({additional:,} additional)", flush=True)
            if shutil.disk_usage(work).free < additional:
                raise OSError(f"Insufficient temporary space for the requested task list; need about {additional:,} additional bytes at {work}")
            preflight_seconds = time.monotonic() - started - setup_seconds
            for task_index, task in enumerate(tasks):
                identifier = task["id"]
                if result["tasks"].get(identifier, {}).get("status") == "complete":
                    continue
                task_started = time.monotonic()
                manifest, training, development = prepared[identifier]
                source = Path(manifest["source_root"])
                task_result = {"status": "running", "evaluation": evaluation, "task_digest": task["digest"], "preparation_digest": manifest["digest"], "data_digest": manifest["splits"][evaluation]["digest"], "release_status": task["release_status"], "method": task["method"], "readout": model.info.readouts[task["method"]], "model_identity": metadata["digest"], "extraction_digests": [], "extraction_work": [], "extraction_seconds": 0.0}
                task_result["preparation"] = {"source": manifest["source"], "source_root": manifest["source_root"], "splits": {split: {key: value for key, value in entry.items() if key != "file"} for split, entry in manifest["splits"].items()}}
                result["tasks"][identifier] = task_result
                write_json(output / "results.json", result)

                def bank_for(rows):
                    bank, usage = build_bank(model, model_metadata=metadata, task=task, root=source, rows=rows, work_dir=work, execution=execution, storage=storage)
                    timing = usage.pop("timing")
                    early = early_extractions.pop(bank.directory, None)
                    if early:
                        usage["extracted_examples"] += early["extracted_examples"]
                        usage["reused_examples"] -= early["extracted_examples"]
                        usage["extracted_ids"].extend(early["extracted_ids"])
                        for key in ("setup_seconds", "teardown_seconds", "total_seconds"):
                            timing[key] += early["timing"][key]
                        for unit in ("seconds", "examples"):
                            timing["steady_state_" + unit] += timing["first_batch_" + unit] + early["timing"]["steady_state_" + unit]
                            timing["first_batch_" + unit] = early["timing"]["first_batch_" + unit]
                    timing["steady_state_images_per_second"] = timing["steady_state_examples"] / timing["steady_state_seconds"] if timing["steady_state_examples"] else None
                    timing["warmup_scope"] = "first-new-batch-including-loader-startup-extraction-and-shard-commit"
                    task_result["extraction_seconds"] += timing["total_seconds"]
                    extraction_digest = digest(feature_identity(metadata, task, rows))
                    task_result["extraction_digests"].append(extraction_digest)
                    extracted_ids = set(usage.pop("extracted_ids"))
                    task_result["extraction_work"].append({"digest": extraction_digest, "performed": cost_summary(bank, [row for row in rows if row["id"] in extracted_ids]), "reused": cost_summary(bank, [row for row in rows if row["id"] not in extracted_ids]), "timing": timing, **usage})
                    active_banks[bank.directory] = bank
                    return bank, usage

                if task["kind"] == "correspondence":
                    if evaluation == "final":
                        _, scoring_rows = load_prepared(root, task, "final")
                    else:
                        scoring_rows = development
                    images = sorted({row["id"]: row for row in distinct_images(scoring_rows) + training}.values(), key=lambda row: row["id"])
                    bank, usage = bank_for(images)
                    extraction_done = time.monotonic()
                    metrics, supporting, records = evaluate_pairs(bank, scoring_rows, source, task)
                    task_result.update(supporting_metrics=supporting, diagnostics=feature_diagnostics(bank, manifest["diagnostic_sample_ids"]), sample_count=len(scoring_rows), extraction_costs=cost_summary(bank, images), extraction_usage=usage, feature_shape=bank.manifest["shape"], feature_dtype=bank.manifest["dtype"], probe_status="not_applicable")
                else:
                    selection_digest = probe_identity(task, manifest, metadata)
                    selection_directory = work / "probes" / selection_digest
                    selection = read_selection(selection_directory)
                    usage = {"extracted_examples": 0, "reused_examples": 0}
                    if selection is None:
                        bank, usage = bank_for(training + development)
                        diagnostics = feature_diagnostics(bank, manifest["diagnostic_sample_ids"])
                        fit_started = time.monotonic()
                        selection = fit_probe(bank, training=training, development=development, root=source, task=task, directory=selection_directory, execution=execution, diagnostics=diagnostics)
                        task_result["probe_fitting_seconds"] = time.monotonic() - fit_started
                        task_result["probe_status"] = "regenerated" if evaluation == "final" else "fitted"
                    else:
                        bank = None
                        task_result["probe_status"] = "reused"
                        task_result["probe_fitting_seconds"] = 0.0
                    selection_finished = time.monotonic()
                    if evaluation == "final":
                        _, scoring_rows = load_prepared(root, task, "final")
                        bank, score_usage = bank_for(scoring_rows)
                        usage = {key: usage[key] + score_usage[key] for key in usage}
                    else:
                        scoring_rows = development
                        if bank is None:
                            bank, score_usage = bank_for(scoring_rows)
                            usage = {key: usage[key] + score_usage[key] for key in usage}
                    extraction_done = time.monotonic()
                    metrics, seeds, records = evaluate_selection(selection, selection_directory, bank, scoring_rows, source, task, execution)
                    task_result.update(seed_results=seeds, selection={"digest": selection_digest, "normalization_sha256": selection["normalization_sha256"], "seeds": [{key: value for key, value in seed.items() if key != "head_file"} for seed in selection["seeds"]], "candidates": [{key: value for key, value in candidate.items() if key != "head_file"} for candidate in selection["candidates"]]}, diagnostics=selection["diagnostics"], sample_count=len(scoring_rows), extraction_costs=cost_summary(bank, scoring_rows), extraction_usage=usage, feature_shape=selection["shape"], feature_dtype=selection["dtype"], selection_seconds=selection_finished - task_started)
                statistics_path = output / "statistics" / f"{identifier}.json"
                write_json(statistics_path, {"schema_version": "1.0.0", "task": identifier, "evaluation": evaluation, "records": records})
                task_result.update(status="complete", metrics={name: {"value": value, "unit": "percent", "direction": "descriptive" if name.endswith("_std") or "ceiling" in name else "higher"} for name, value in metrics.items()}, statistics={"file": str(statistics_path.relative_to(output)), "sha256": file_digest(statistics_path)}, timing={"through_extraction_seconds": extraction_done - task_started, "scoring_and_commit_seconds": time.monotonic() - extraction_done, "total_seconds": time.monotonic() - task_started})
                result["metrics"][identifier] = metrics
                # Commit this task before starting another; later failures preserve it.
                write_json(output / "results.json", result)
                if first_result_seconds is None:
                    first_result_seconds = time.monotonic() - started
                result.setdefault("timing", {})["time_to_first_task_seconds"] = first_result_seconds
                print(f"{identifier} ({evaluation}): {metrics}", flush=True)
                future = tasks[task_index + 1:]
                for directory, bank in list(active_banks.items()):
                    future_contexts = [{key: value for key, value in feature_identity(metadata, other, []).items() if key != "samples"} for other in future]
                    if bank.manifest["identity"] not in future_contexts:
                        bank.close()
                        remove(directory, storage=storage)
                        del active_banks[directory]
            cleanup_started = time.monotonic()
            for directory, bank in active_banks.items():
                bank.close()
                remove(directory, storage=storage)
            # Also finish cleanup from an earlier interrupted task-commit/cleanup boundary.
            if (work / "banks").exists():
                remove(work / "banks", storage=storage)
            if evaluation == "final" or execution.work_dir is None:
                clear(work, storage=storage)
            result.update(status="complete", timing={"setup_seconds": setup_seconds, "cold_preflight_seconds": preflight_seconds, "time_to_first_task_seconds": first_result_seconds, "cleanup_seconds": time.monotonic() - cleanup_started, "total_seconds": time.monotonic() - started}, storage={"estimated_peak_temporary_bytes": estimated_peak, "peak_temporary_bytes": storage.peak, "retained_temporary_bytes": storage.current, "retained_workspace_bytes": bytes_used(work), "measurement": "logical-file-bytes-in-banks-and-probes-including-atomic-copies"}, memory={"peak_host_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, "peak_gpu_bytes": torch.cuda.max_memory_allocated(execution.device) if torch.device(execution.device).type == "cuda" else None})
            result["memory"]["host_peak_scope"] = "process-lifetime"
            write_json(output / "results.json", result)
        except BaseException as error:
            result["status"] = "failed"
            result["failure"] = {"type": type(error).__name__, "message": str(error), "remaining_workspace": str(work)}
            if "identifier" in locals() and result["tasks"][identifier]["status"] != "complete":
                result["tasks"][identifier].update(status="failed", failure=result["failure"])
            write_json(output / "results.json", result)
            raise
        return RunResult("1.0.0", "complete", result["metrics"], output)
