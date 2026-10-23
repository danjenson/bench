"""Check a caller-loaded GridGlimpseSSL on artificial inputs, without scientific scores."""

from dataclasses import asdict
import hashlib
from pathlib import Path
import random
import time

import numpy as np
import torch

from bench import Execution, ImageBatch
from bench.features import adapter_mode, extract, model_identity, numerical_mode, supplied_extent, validate_execution
from bench.storage import file_digest, write_json
from grid_glimpse_ssl_adapter import GridGlimpseSSLAdapter


def verify(model, adapter_arguments, output, *, device="cuda", profiles=("fp32-v1", "bf16-v1")):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model.to(device)
    adapters = {name: GridGlimpseSSLAdapter(model, condition=name, **adapter_arguments) for name in ("M1", "C1", "Mk", "Ck", "Mk-repeat", "Ck-repeat")}
    model.train()
    model.encoder.norm.eval()
    model.decoder.blocks[0].eval()
    parameter = model.decoder.query_token
    parameter.grad = torch.full_like(parameter, 0.125)
    gradient = parameter.grad.clone()
    flags = [module.training for module in model.modules()]
    rng = (random.getstate(), np.random.get_state(), torch.random.get_rng_state().clone(), [state.clone() for state in torch.cuda.get_rng_state_all()] if device.startswith("cuda") else [])
    pixels = torch.rand(2, 3, 224, 224, generator=torch.Generator().manual_seed(42)).to(device)
    valid = torch.ones(2, 1, 224, 224, dtype=torch.bool, device=device)
    valid[1, :, :48] = False
    valid[1, :, 176:] = False
    transforms = torch.tensor([[2., 0., -0.5], [0., 3., 0.25], [0., 0., 1.]], device=device).repeat(2, 1, 1)
    batch = ImageBatch(pixels, valid, transforms, ("grid-contract-a", "grid-contract-b"))
    identity = model_identity(adapters["Ck"], Execution(device=device, numerics="fp32-v1"))
    report = {"status": "running", "model": identity, "profiles": {}, "harness_sha256": file_digest(__file__), "scope": "Artificial adapter contract checks; no scientific task scores, registered score-parity acceptance, trajectory sensitivity or training overhead."}
    all_outputs = {}
    for profile in profiles:
        execution = Execution(device=device, numerics=profile)
        validate_execution(execution)
        outputs, inputs, conditions = {}, {}, {}
        tolerance = 0.02 if profile == "bf16-v1" else 1e-5
        with numerical_mode(execution):
            # The native forward is the oracle. Its held-out crops are artificial zeros.
            crops, coordinates, query, _ = adapters["Ck"].observations(batch)
            native_crops = torch.cat((crops, torch.zeros_like(crops[:, :1]).expand(-1, 3, -1, -1, -1)), dim=1)
            native_coordinates = torch.cat((coordinates, query.expand(-1, 3, -1)), dim=1)
            with adapter_mode(model), torch.inference_mode(), torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=profile == "bf16-v1"):
                training_output = model(native_crops, native_coordinates)
                training_scene = training_output.scene_prediction.float().cpu()
                repeated = model.decoder(training_output.features, coordinates, query.expand(-1, 3, -1))[:, -1].float().cpu()
                native = model.decoder(training_output.features, coordinates, query)[:, 0].float().cpu()
            torch.testing.assert_close(repeated, training_scene, rtol=1e-5, atol=1e-5)
            # Native queries form independent batch entries. BF16 kernels can round
            # differently when their count changes; strict adapter parity uses one.
            torch.testing.assert_close(native, training_scene, rtol=tolerance, atol=tolerance)
            native_query_difference = native - training_scene

            def reject_target(*args):
                raise AssertionError("Adapter invoked the target encoder or training forward")

            target_hook = model.target_encoder.register_forward_pre_hook(reject_target)
            training_hook = model.register_forward_pre_hook(reject_target)
            try:
                for name, adapter in adapters.items():
                    captured, decoder_calls = [], []
                    encoder_hook = adapter.encoder.register_forward_pre_hook(lambda module, args: captured.append(args[0].detach().cpu().clone()))
                    decoder_hook = adapter.decoder.register_forward_pre_hook(lambda module, args: decoder_calls.append(1))
                    try:
                        result = extract(adapter, batch, "global", execution)
                    finally:
                        encoder_hook.remove()
                        decoder_hook.remove()
                    assert len(captured) == 1 and len(decoder_calls) == int(adapter.composed)
                    assert result["values"].shape == (2, model.encoder.representation_dim)
                    assert result["values"].dtype == torch.float32
                    assert result["costs"]["encoder_passes_count"].tolist() == [adapter.count] * 2
                    assert result["costs"]["encoder_pixels_count"].tolist() == [adapter.count * 96 ** 2] * 2
                    assert result["costs"]["decoder_passes_count"].tolist() == [int(adapter.composed)] * 2
                    outputs[name], inputs[name] = result["values"], captured[0]
                    changed = pixels.clone()
                    boxes = adapter.observation_boxes(batch)
                    for index, sequence in enumerate(boxes):
                        observed = torch.zeros(224, 224, dtype=torch.bool, device=device)
                        for left, top, right, bottom in sequence:
                            observed[top:bottom, left:right] = True
                        changed[index, :, ~observed] = 0.123
                    altered = ImageBatch(changed, valid, transforms, batch.sample_ids)
                    torch.testing.assert_close(extract(adapter, altered, "global", execution)["values"], outputs[name], rtol=0, atol=0)
                    single = ImageBatch(pixels[1:], valid[1:], transforms[1:], batch.sample_ids[1:])
                    torch.testing.assert_close(extract(adapter, single, "global", execution)["values"], outputs[name][1:], rtol=tolerance, atol=tolerance)
                    if adapter.composed:
                        spatial = extract(adapter, batch, "spatial", execution)
                        assert spatial["values"].shape == (2, model.encoder.representation_dim, 3, 3)
                        assert spatial["values"].dtype == torch.float32
                        torch.testing.assert_close(spatial["values"].mean((2, 3)), outputs[name], rtol=1e-5, atol=1e-5)
                        torch.testing.assert_close(spatial["grid_to_original"], transforms.cpu(), rtol=0, atol=0)
                        torch.testing.assert_close(spatial["extent_in_original"], supplied_extent(batch).cpu(), rtol=0, atol=0)
                        torch.testing.assert_close(extract(adapter, altered, "spatial", execution)["values"], spatial["values"], rtol=0, atol=0)
                        if name == "Ck":
                            torch.testing.assert_close(spatial["values"].flatten(2).transpose(1, 2), native, rtol=1e-5, atol=1e-5)
                    conditions[name] = {"model_info": asdict(adapter.info), "boxes": boxes, "input_sha256": hashlib.sha256(captured[0].numpy().tobytes()).hexdigest(), "global_shape": list(result["values"].shape), "global_sha256": hashlib.sha256(result["values"].numpy().tobytes()).hexdigest(), "costs": {key: value.tolist() for key, value in result["costs"].items()}, "unobserved_pixel_and_batching_checks": True}
                assert torch.equal(inputs["Mk"], inputs["Ck"])
                assert torch.equal(inputs["Mk-repeat"], inputs["Ck-repeat"])
                assert torch.equal(inputs["M1"], inputs["Mk"][:, :1])
                torch.testing.assert_close(outputs["M1"], outputs["Mk-repeat"], rtol=tolerance, atol=tolerance)
                torch.testing.assert_close(outputs["Ck"], native.mean(1), rtol=1e-5, atol=1e-5)

                def fail_after_decoder(*args):
                    raise RuntimeError("injected decoder failure")

                failure_hook = model.decoder.register_forward_hook(fail_after_decoder)
                try:
                    for method in ("global", "spatial"):
                        try:
                            extract(adapters["Ck"], batch, method, execution)
                        except RuntimeError as error:
                            assert str(error) == "injected decoder failure"
                        else:
                            raise AssertionError("The failure injection did not execute")
                        assert flags == [module.training for module in model.modules()]
                finally:
                    failure_hook.remove()
            finally:
                target_hook.remove()
                training_hook.remove()
        assert flags == [module.training for module in model.modules()]
        assert torch.equal(parameter.grad, gradient)
        assert all(value.grad is None for value in model.parameters() if value is not parameter)
        assert model_identity(adapters["Ck"], execution)["state_digest"] == identity["state_digest"]
        assert random.getstate() == rng[0]
        current = np.random.get_state()
        assert current[0] == rng[1][0] and np.array_equal(current[1], rng[1][1]) and current[2:] == rng[1][2:]
        assert torch.equal(torch.random.get_rng_state(), rng[2])
        assert all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all() if device.startswith("cuda") else [], rng[3], strict=True))
        report["profiles"][profile] = {"conditions": conditions, "native_global_and_spatial_parity": True, "native_parity_reference": "one native decoder query using training-forward context features", "native_query_batching": {"single_query_count": 1, "training_query_count": 3, "training_forward_matches_native_three_queries": True, "maximum_absolute_difference": float(native_query_difference.abs().max()), "rms_difference": float(native_query_difference.square().mean().sqrt()), "rtol": tolerance, "atol": tolerance}, "state_rng_gradients_and_flags_preserved": True, "injected_failures": 2, "batching_and_repeat_feature_tolerance": {"rtol": tolerance, "atol": tolerance}}
        all_outputs[profile] = outputs
    if set(profiles) == {"fp32-v1", "bf16-v1"}:
        report["feature_profile_differences"] = {name: {"maximum_absolute": float((all_outputs["fp32-v1"][name] - all_outputs["bf16-v1"][name]).abs().max()), "rms": float((all_outputs["fp32-v1"][name] - all_outputs["bf16-v1"][name]).square().mean().sqrt())} for name in adapters}
    report.update(status="passed", seconds=time.perf_counter() - started, device=device, gpu=torch.cuda.get_device_name(device) if device.startswith("cuda") else None)
    write_json(output / "acceptance.json", report)
    print(f"Grid adapter contract checks passed: {output}", flush=True)
    return report
