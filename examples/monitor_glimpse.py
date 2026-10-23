"""Measure one immutable checkpoint evaluation, including loading and cleanup."""

import argparse
from pathlib import Path
import time

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

import bench
from bench.analysis import paired_difference
from bench.storage import file_digest, write_json
from glimpse_ssl_adapter import GlimpseSSLAdapter
from grid_glimpse_ssl_adapter import GridGlimpseSSLAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous", type=Path, help="Previous completed monitoring output for paired checkpoint changes")
    parser.add_argument("--names", nargs="+", default=["imagenet-100", "pascal"])
    parser.add_argument("--numerics", choices=["fp32-v1", "bf16-v1"], default="bf16-v1")
    parser.add_argument("--initialization", choices=["trained", "untrained"], default="trained")
    args = parser.parse_args()
    started = time.monotonic()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    configuration = payload["config"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = instantiate(OmegaConf.create(configuration["model"]), _convert_="all")
    if args.initialization == "trained":
        model.load_state_dict(payload["model"], strict=True)
    model.cuda()
    import glimpse_ssl.models
    import glimpse_ssl.vit5

    recipe = {"configuration": configuration["model"], "initialization": args.initialization}
    if args.initialization == "trained":
        recipe.update(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=file_digest(args.checkpoint), training_epoch=payload["epoch"], pretraining={"dataset": configuration["dataset"], "split_exposure": "consult original sample manifest"}, model_selection="unknown; caller-supplied checkpoint; benchmark final does not imply untouched data")
    else:
        recipe.update(initialization_seed=0, pretraining="none", model_selection="none", configuration_source={"checkpoint": str(args.checkpoint.resolve()), "sha256": file_digest(args.checkpoint)})
    arguments = dict(model=model, condition="Ck", budget=2, observation_seed=0, source_revision=file_digest(glimpse_ssl.models.__file__) + ":" + file_digest(glimpse_ssl.vit5.__file__), model_recipe=recipe)
    if configuration["model"]["_target_"] == "glimpse_ssl.models.GridGlimpseSSL":
        recipe["training_source_revision"] = "unknown" if args.initialization == "trained" else "none"
        adapter = GridGlimpseSSLAdapter(**arguments)
    else:
        adapter = GlimpseSSLAdapter(**arguments, crop_size=configuration["glimpse"]["sampler"]["output_size"], patch_size=model.encoder.patch_size[0], training_budgets=[configuration["model"]["num_context_views"]] if args.initialization == "trained" else [])
    result = bench.run(adapter, names=args.names, data_root=args.data_root, output_dir=args.output, execution=bench.Execution(device="cuda", numerics=args.numerics))
    evaluated = time.monotonic()
    changes = {task: paired_difference(args.previous, args.output, task) for task in result.metrics} if args.previous else {}
    elapsed = time.monotonic() - started
    write_json(args.output / "harness.json", {"wall_seconds_including_model_loading": elapsed, "evaluation_seconds_including_model_loading": evaluated - started, "paired_analysis_seconds": time.monotonic() - evaluated, "metrics": result.metrics, "previous_output": str(args.previous.resolve()) if args.previous else None, "paired_changes": changes, "initialization": args.initialization, "script_sha256": file_digest(__file__), "calibration_gate": "single checkpoint measurement, not release calibration"})
    print(f"Harness wall seconds: {elapsed:.3f}")
    for task, change in changes.items():
        print(f"{task}: {change['difference_percentage_points']:+.4f} points; paired 95% interval {change['paired_95_interval']}")


if __name__ == "__main__":
    main()
