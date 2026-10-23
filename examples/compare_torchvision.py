"""Compare published and untrained reference encoders on prepared development data."""

import argparse
import gc
from pathlib import Path
import time

import torch
from torchvision.models import get_model

import bench
from bench.adapters.torchvision import TorchvisionAdapter
from bench.storage import file_digest, read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-manifest", type=Path, required=True)
    parser.add_argument("--architecture", choices=["resnet18", "vit_b_16"], required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--numerics", choices=["bf16-v1", "fp32-v1"], default="bf16-v1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--names", nargs="+", default=["imagenet-100", "pascal"])
    args = parser.parse_args()
    record = next(row for row in read_json(args.weights_manifest) if row["architecture"] == args.architecture)
    checkpoint = Path(record["checkpoint"])
    if file_digest(checkpoint) != record["sha256"]:
        raise ValueError("Reference checkpoint no longer matches its declared SHA256")
    report = {
        "architecture": args.architecture, "weights": record, "names": args.names,
        "script_sha256": file_digest(__file__), "controls": {},
        "scope": "development controls; not trajectory calibration or an untouched final test",
        "timing_scope": "each control includes construction, transfer, evaluation and cleanup in one shared process",
    }
    execution = bench.Execution(device="cuda", numerics=args.numerics, batch_size=args.batch_size)
    for initialization in ("pretrained", "untrained"):
        output = args.output / initialization
        existing = (output / "manifest.json").exists()
        started = time.monotonic()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            if initialization == "pretrained":
                adapter = TorchvisionAdapter.from_checkpoint(checkpoint, architecture=args.architecture, provenance=record)
            else:
                adapter = TorchvisionAdapter(
                    get_model(args.architecture, weights=None), architecture=args.architecture,
                    provenance={"initialization_seed": 0, "pretraining": "none", "model_selection": "none"},
                )
        adapter.cuda()
        result = bench.run(adapter, names=args.names, data_root=args.data_root, output_dir=output, execution=execution)
        torch.cuda.synchronize()
        report["controls"][initialization] = {
            "status": result.status, "metrics": result.metrics,
            "wall_seconds_including_model_loading": time.monotonic() - started,
            "existing_output_on_entry": existing, "output": str(output.resolve()),
            "temporary_workspace_removed": not (output / ".work").exists(),
        }
        write_json(args.output / "controls.json", report)
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
    print(f"Control results: {args.output / 'controls.json'}")


if __name__ == "__main__":
    main()
