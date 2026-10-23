"""Run the matched condition matrix for a GlimpseSSL or GridGlimpseSSL checkpoint."""

import argparse
from dataclasses import asdict
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

import bench
from bench.analysis import paired_contrast, paired_difference
from bench.protocols import resolve_names
from bench.storage import file_digest, write_json
from bench.workspace import prune
from glimpse_ssl_adapter import GlimpseSSLAdapter
from grid_glimpse_ssl_adapter import GridGlimpseSSLAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("~/datasets"))
    parser.add_argument("--names", nargs="+", default=["imagenet-100"])
    parser.add_argument("--evaluation", choices=["development", "final"], default="development")
    parser.add_argument("--observation-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--numerics", choices=["bf16-v1", "fp32-v1"], default="bf16-v1")
    args = parser.parse_args()
    tasks = resolve_names(args.names)
    if any(task["kind"] != "classification" for task in tasks):
        raise ValueError("This classification condition matrix requires global classification tasks")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    configuration = checkpoint["config"]
    model = instantiate(OmegaConf.create(configuration["model"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device)
    import glimpse_ssl.models
    import glimpse_ssl.vit5

    arguments = dict(model=model, budget=args.budget, source_revision=file_digest(glimpse_ssl.models.__file__) + ":" + file_digest(glimpse_ssl.vit5.__file__), model_recipe={"configuration": configuration["model"], "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_digest(args.checkpoint), "training_epoch": checkpoint["epoch"], "pretraining": {"dataset": configuration["dataset"], "split_exposure": "consult original sample manifest"}, "model_selection": "unknown; caller-supplied checkpoint; benchmark final does not imply untouched data"})
    if configuration["model"]["_target_"] == "glimpse_ssl.models.GridGlimpseSSL":
        adapter_type = GridGlimpseSSLAdapter
        arguments["model_recipe"]["training_source_revision"] = "unknown"
    else:
        adapter_type = GlimpseSSLAdapter
        arguments.update(crop_size=configuration["glimpse"]["sampler"]["output_size"], patch_size=model.encoder.patch_size[0], training_budgets=[configuration["model"]["num_context_views"]])
    workspace = args.output / ".work"
    execution = bench.Execution(device=args.device, numerics=args.numerics, work_dir=workspace)
    matrix = {"checkpoint": str(args.checkpoint.resolve()), "observation_seeds": args.observation_seeds, "budget": args.budget, "evaluation": args.evaluation, "names": args.names, "conditions": {}, "paired_comparisons": {}, "observation_and_probe_seeds_are_distinct": True}
    for seed in args.observation_seeds:
        paths = {}
        for condition in ("M1", "C1", "Mk", "Ck", "Mk-repeat", "Ck-repeat"):
            adapter = adapter_type(condition=condition, observation_seed=seed, **arguments)
            output = args.output / f"observation-seed-{seed}" / condition / args.evaluation
            key = f"{seed}/{condition}"
            matrix["conditions"][key] = {"status": "running", "recipe": asdict(adapter.info), "output": str(output.resolve())}
            write_json(args.output / "matrix.json", matrix)
            result = bench.run(adapter, names=args.names, evaluation=args.evaluation, data_root=args.data_root, output_dir=output, execution=execution)
            matrix["conditions"][key].update(status=result.status, metrics=result.metrics)
            paths[condition] = output
            write_json(args.output / "matrix.json", matrix)
        matrix["paired_comparisons"][str(seed)] = {}
        for task in tasks:
            matrix["paired_comparisons"][str(seed)][task["id"]] = {
                "additional_observations_Mk_minus_M1": paired_difference(paths["M1"], paths["Mk"], task["id"]),
                "learned_readout_Ck_minus_Mk": paired_difference(paths["Mk"], paths["Ck"], task["id"]),
                "distinct_evidence_Ck_minus_Ck_repeat": paired_difference(paths["Ck-repeat"], paths["Ck"], task["id"]),
                "multi_observation_readout_interaction": paired_contrast([paths[name] for name in ("Ck", "Mk", "C1", "M1")], [1, -1, -1, 1], task["id"]),
            }
        write_json(args.output / "matrix.json", matrix)
    prune(work_dir=workspace)
    print(f"Condition matrix and paired comparisons: {args.output / 'matrix.json'}")


if __name__ == "__main__":
    main()
