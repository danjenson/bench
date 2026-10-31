import numpy as np
from PIL import Image
import pytest
import torch

from bench import Execution, Extraction, ImageBatch, SpatialFeatures, VisionModel
from bench.features import extract, supplied_extent
from bench.geometry import logits_on_canvas
from bench.images import make_canvas
from bench.protocols import resolve_names


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration job")
def test_bf16_preserves_fp32_spatial_geometry_and_rng():
    canvas = make_canvas(Image.new("RGB", (381, 173)), resolve_names(["pascal"])[0])
    batch = ImageBatch(torch.from_numpy(np.array(canvas.image)).permute(2, 0, 1)[None].float().cuda() / 255, torch.from_numpy(canvas.valid[None, None]).cuda(), torch.from_numpy(canvas.to_original[None]).cuda(), ("opaque",))

    class Spatial(VisionModel):
        def encode_spatial(self, batch):
            values = torch.ones(1, 3, 2, 2, device="cuda", dtype=torch.bfloat16)
            return Extraction(SpatialFeatures(values, batch.image_to_original, supplied_extent(batch)))

    model = Spatial()
    before = torch.cuda.get_rng_state().clone()
    outputs = [extract(model, batch, "spatial", Execution(device="cuda", numerics=profile)) for profile in ("fp32-v1", "bf16-v1")]
    torch.testing.assert_close(outputs[0]["extent_in_original"], outputs[1]["extent_in_original"], rtol=0, atol=0)
    torch.testing.assert_close(outputs[1]["extent_in_original"], torch.tensor([[0., 0., 1., 1.]]), atol=1e-6, rtol=0)
    assert outputs[1]["extent_in_original"].dtype == torch.float32
    assert torch.equal(before, torch.cuda.get_rng_state())


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration job")
@pytest.mark.parametrize("grid_device,canvas_device", [("cpu", "cpu"), ("cuda", "cuda"), ("cpu", "cuda")])
def test_dense_resampling_cuda_values_and_gradients(grid_device, canvas_device):
    generator = torch.Generator().manual_seed(0)
    values = torch.randn(21, 4, 4, generator=generator)
    reference_values = values.double().requires_grad_()
    actual_values = values.cuda().requires_grad_()
    affine = torch.tensor([[1., 0., 0.], [0., 512 / 351, -80 / 351], [0., 0., 1.]])
    y, x = torch.meshgrid((torch.arange(512, dtype=torch.float64) + .5) / 512, (torch.arange(512, dtype=torch.float64) + .5) / 512, indexing="ij")
    reference = torch.nn.functional.grid_sample(reference_values[None], torch.stack([x, y], -1)[None] * 2 - 1, mode="bilinear", padding_mode="border", align_corners=False)[0]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=grid_device == canvas_device):
        actual = logits_on_canvas(actual_values, affine.to(grid_device), affine.to(canvas_device), (512, 512))
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu().double(), reference, rtol=1e-5, atol=3e-6)
    target = torch.randint(21, (1, 512, 512), generator=generator)
    target[:, :80] = 255
    target[:, -81:] = 255
    torch.nn.functional.cross_entropy(reference[None], target, ignore_index=255).backward()
    torch.nn.functional.cross_entropy(actual[None], target.cuda(), ignore_index=255).backward()
    torch.testing.assert_close(actual_values.grad.cpu().double(), reference_values.grad, rtol=1e-4, atol=1e-6)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA integration job")
@pytest.mark.parametrize("name", ["pascal", "ade20k"])
def test_packed_targets_preserve_cuda_probe_training(name, tmp_path, monkeypatch):
    from bench.features import numerical_mode
    from bench.probes import PreparedTargets, normalization, train_candidate

    task = resolve_names([name])[0]
    task["probe"].update(updates=4, batch_size=2, selection="final-update")
    generator = torch.Generator().manual_seed(41)
    rows = [{"id": str(i), "image": f"{i}.jpg", "mask": f"{i}.png"} for i in range(2)]
    features = {row["id"]: {"values": torch.randn(3, 2, 2, generator=generator),
        "grid_to_original": torch.eye(3), "extent_in_original": torch.tensor([0., 0., 1., 1.])} for row in rows}
    targets = {row["id"]: (torch.arange(20).reshape(4, 5) + i) % task["classes"] for i, row in enumerate(rows)}
    targets['0'][0] = 255
    targets['1'][0, :2] = 255
    monkeypatch.setattr('bench.probes.label_canvas', lambda root, row, task: (targets[row['id']], torch.eye(3)))

    class Bank:
        manifest = {"shape": [3, 2, 2]}
        storage = None

        def __getitem__(self, identifier):
            return features[identifier]

    bank = Bank()
    statistics = normalization(bank, rows, task)
    arguments = (bank, rows, rows, tmp_path, task, statistics, 'cuda', 0, 0.003, 0.01)
    with numerical_mode(Execution(device='cuda', numerics='fp32-v1')):
        ordinary = train_candidate(*arguments, tmp_path / 'ordinary.pt')
        with PreparedTargets(tmp_path, task) as cache:
            for row in rows:
                torch.testing.assert_close(cache.read(row)[0].cuda().long(), targets[row['id']].cuda(), rtol=0, atol=0)
            packed = train_candidate(*arguments, tmp_path / 'packed.pt', targets=cache)
        assert not cache.entries and cache.current_bytes == 0
    for key in ['seed', 'epoch', 'updates', 'learning_rate', 'weight_decay', 'probe_parameters', 'development_score']:
        assert ordinary[key] == packed[key]
    for key in ['training_loss', 'development_loss']:
        assert packed[key] == pytest.approx(ordinary[key], rel=1e-6, abs=1e-6)
    first = torch.load(tmp_path / 'ordinary.pt', weights_only=True)
    second = torch.load(tmp_path / 'packed.pt', weights_only=True)
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=1e-5, atol=1e-6)
