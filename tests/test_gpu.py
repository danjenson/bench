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
