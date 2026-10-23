"""Verify real-data result reuse and fresh extraction after standalone cleanup."""

import argparse
from pathlib import Path
import time
from unittest.mock import patch

from omegaconf import OmegaConf

import bench
from bench.adapters.torchvision import TorchvisionAdapter
from bench.protocols import resolve_names
from bench.runner import read_result
from bench.storage import file_digest, read_json, write_json
from bench.workspace import prune


def verify_regeneration(adapter, *, baseline, data_root, output, loading_seconds):
    """Check a caller-constructed CUDA adapter against its completed standalone run."""
    baseline, output = Path(baseline), Path(output)
    original = read_result(baseline)
    regenerated_output = output / "regenerated"
    if regenerated_output.exists():
        raise ValueError("Choose a fresh output directory to measure regeneration")
    if (baseline / ".work").exists():
        raise ValueError("Baseline must be a completed standalone run with temporary state removed")
    resolved = read_json(baseline / "manifest.json")
    names, evaluation = resolved["identity"]["requested_names"], resolved["identity"]["evaluation"]
    # Preserve the baseline extraction batch size while changing only the output path.
    runtime = OmegaConf.load(baseline / "resolved.yaml")["execution"]
    execution = bench.Execution(device="cuda", batch_size=runtime["batch_size"], workers=runtime["workers"], numerics=original["numerics"])
    started = time.monotonic()
    with patch.object(adapter, "encode_global", side_effect=AssertionError("Completed-result reuse called the encoder")) as global_calls, patch.object(adapter, "encode_spatial", side_effect=AssertionError("Completed-result reuse called the encoder")) as spatial_calls:
        repeated = bench.run(adapter, names=names, evaluation=evaluation, data_root=data_root, output_dir=baseline, execution=execution)
        reuse_calls = {"global": global_calls.call_count, "spatial": spatial_calls.call_count}
    reuse_seconds = time.monotonic() - started
    assert repeated.metrics == original["metrics"]
    prune(work_dir=baseline / ".work")
    regeneration_calls = {"global": 0, "spatial": 0}

    def observed(method):
        encode = getattr(adapter, "encode_" + method)

        def counted(batch):
            regeneration_calls[method] += 1
            return encode(batch)

        return counted

    started = time.monotonic()
    with patch.object(adapter, "encode_global", new=observed("global")), patch.object(adapter, "encode_spatial", new=observed("spatial")):
        bench.run(adapter, names=names, evaluation=evaluation, data_root=data_root, output_dir=regenerated_output, execution=execution)
    regeneration_seconds = time.monotonic() - started
    regenerated = read_result(regenerated_output)
    assert regenerated["model"]["digest"] == original["model"]["digest"]
    comparisons = {}
    for task in resolve_names(names):
        identifier = task["id"]
        before, after = original["tasks"][identifier], regenerated["tasks"][identifier]
        assert before["task_digest"] == after["task_digest"]
        assert before["preparation_digest"] == after["preparation_digest"]
        assert regeneration_calls[task["method"]] > 0
        differences = {metric: abs(after["metrics"][metric]["value"] - entry["value"]) for metric, entry in before["metrics"].items()}
        tolerance = task["numerical_tolerance"]["absolute_percentage_points"]
        assert all(difference <= tolerance for difference in differences.values())
        comparisons[identifier] = {"absolute_score_differences": differences, "tolerance_percentage_points": tolerance, "statistics_identical": before["statistics"]["sha256"] == after["statistics"]["sha256"]}
    assert not (regenerated_output / ".work").exists()
    report = {
        "script_sha256": file_digest(__file__), "baseline": str(baseline.resolve()),
        "regenerated": str(regenerated_output.resolve()), "loading_seconds": loading_seconds,
        "completed_result_reuse_seconds": reuse_seconds, "regeneration_seconds": regeneration_seconds,
        "timing_scope": "phase timers exclude separately reported model loading; baseline run retains its original cold timing",
        "encoder_calls_on_reuse": reuse_calls, "encoder_calls_on_regeneration": regeneration_calls,
        "comparisons": comparisons, "temporary_workspace_removed": True,
    }
    write_json(output / "regeneration.json", report)
    print(f"Result reuse and regeneration passed: {output / 'regeneration.json'}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-manifest", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    original = read_result(args.baseline)
    record = next(row for row in read_json(args.weights_manifest) if row["architecture"] == original["model"]["info"]["recipe"]["architecture"])
    if file_digest(record["checkpoint"]) != record["sha256"]:
        raise ValueError("Reference checkpoint no longer matches its declared SHA256")
    started = time.monotonic()
    adapter = TorchvisionAdapter.from_checkpoint(record["checkpoint"], architecture=record["architecture"], provenance=record).cuda()
    verify_regeneration(adapter, baseline=args.baseline, data_root=args.data_root, output=args.output, loading_seconds=time.monotonic() - started)


if __name__ == "__main__":
    main()
