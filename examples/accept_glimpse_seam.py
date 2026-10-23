"""Exercise the real GlimpseSSL checkpoint in its owning model environment.

Run with the model repository's Python, with bench/src and examples on PYTHONPATH.
The checkpoint contains the model's resolved Hydra configuration.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from bench import Execution, ImageBatch
from bench.features import extract, model_identity
from bench.storage import file_digest, write_json
from glimpse_ssl_adapter import GlimpseSSLAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
    configuration = checkpoint["config"]
    model = instantiate(OmegaConf.create(configuration["model"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(args.device)
    import glimpse_ssl.models

    source_revision = file_digest(glimpse_ssl.models.__file__)
    arguments = dict(model=model, budget=2, observation_seed=0, crop_size=configuration["glimpse"]["sampler"]["output_size"], patch_size=model.encoder.patch_size[0], training_budgets=[configuration["model"]["num_context_views"]], source_revision=source_revision, model_recipe={"configuration": configuration["model"], "checkpoint_sha256": file_digest(args.checkpoint), "training_epoch": checkpoint["epoch"], "model_selection": "unknown; caller-supplied checkpoint; original split exposure must be reviewed"})
    adapters = {condition: GlimpseSSLAdapter(condition=condition, **arguments) for condition in ("M1", "C1", "Mk", "Ck", "Mk-repeat", "Ck-repeat")}
    generator = torch.Generator().manual_seed(42)
    pixels = torch.rand(2, 3, 224, 224, generator=generator).to(args.device)
    batch = ImageBatch(pixels, torch.ones(2, 1, 224, 224, dtype=torch.bool, device=args.device), torch.eye(3, device=args.device).repeat(2, 1, 1), ("acceptance-opaque-1", "acceptance-opaque-2"))
    report = {"checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": file_digest(args.checkpoint), "source_revision": source_revision, "input_kind": "artificial-contract-fixture-not-a-scientific-score", "profiles": {}}
    for profile in ("fp32-v1", "bf16-v1"):
        execution = Execution(device=args.device, numerics=profile)
        outputs, conditions = {}, {}
        for name, adapter in adapters.items():
            before = model_identity(adapter, execution)
            rng = torch.cuda.get_rng_state().clone() if args.device.startswith("cuda") else torch.random.get_rng_state().clone()
            captured = []

            def capture(module, inputs):
                captured.append(inputs[0].detach().cpu().clone())

            hook = adapter.encoder.register_forward_pre_hook(capture)
            try:
                result = extract(adapter, batch, "global", execution)
            finally:
                hook.remove()
            boxes = adapter.observation_boxes(batch)
            changed = pixels.clone()
            for index, sequence in enumerate(boxes):
                observed = torch.zeros(224, 224, dtype=torch.bool, device=args.device)
                for left, top, right, bottom in sequence:
                    observed[top:bottom, left:right] = True
                changed[index, :, ~observed] = 0.01
            modified = ImageBatch(changed, batch.valid_pixels, batch.image_to_original, batch.sample_ids)
            altered = extract(adapter, modified, "global", execution)
            torch.testing.assert_close(result["values"], altered["values"], rtol=0, atol=0)
            current_rng = torch.cuda.get_rng_state() if args.device.startswith("cuda") else torch.random.get_rng_state()
            assert torch.equal(rng, current_rng)
            assert before["state_digest"] == model_identity(adapter, execution)["state_digest"]
            outputs[name] = result["values"]
            conditions[name] = {"model_info": asdict(adapter.info), "boxes": boxes, "actual_encoder_input_sha256": __import__("hashlib").sha256(captured[0].numpy().tobytes()).hexdigest(), "shape": list(result["values"].shape), "costs": {key: value.tolist() for key, value in result["costs"].items()}, "unchanged_model_rng_and_unobserved_pixel_invariance": True}
        assert conditions["Mk"]["actual_encoder_input_sha256"] == conditions["Ck"]["actual_encoder_input_sha256"]
        assert conditions["Mk-repeat"]["actual_encoder_input_sha256"] == conditions["Ck-repeat"]["actual_encoder_input_sha256"]
        torch.testing.assert_close(outputs["M1"], outputs["Mk-repeat"], rtol=0.02 if profile == "bf16-v1" else 1e-5, atol=0.02 if profile == "bf16-v1" else 1e-5)
        spatial = extract(adapters["Ck"], batch, "spatial", execution)
        report["profiles"][profile] = {"conditions": conditions, "spatial_shape": list(spatial["values"].shape), "repeat_pooling_matches_single": True}
    write_json(args.output, report)
    print(f"Real checkpoint seam acceptance: {args.output}")


if __name__ == "__main__":
    main()
