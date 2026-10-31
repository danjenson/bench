from copy import deepcopy
import importlib

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F

import bench
from bench.geometry import grid_centers, logits_on_canvas
from bench.probes import FitResampling
from bench.storage import digest, read_json


def spatial_operations(aligned):
    values = torch.linspace(-1, 1, 36, dtype=torch.float32).reshape(3, 3, 4)
    affine = torch.eye(3, dtype=torch.float32)
    canvas = torch.eye(3, dtype=torch.float32)
    if not aligned:
        affine[0, 0], affine[1, 1], affine[0, 2] = .9, 1.1, .03
    centers = grid_centers({'values': values, 'grid_to_original': affine})
    target = (torch.arange(80, dtype=torch.long).reshape(8, 10) % 3)
    target[0] = 255
    results = [centers]
    for cached in (False, True):
        logits = values.clone().requires_grad_()
        with FitResampling({'kind': 'segmentation'}, 'cpu') as cache:
            output = cache('sample', logits, affine, canvas, target.shape) if cached else logits_on_canvas(logits, affine, canvas, target.shape)
            loss = F.cross_entropy(output[None], target[None], ignore_index=255)
            loss.backward()
            results.extend([output.detach(), loss.detach(), logits.grad.clone()])
        assert cache.current_bytes == 0 and not cache.entries
    return results


@pytest.mark.parametrize('dtype', [torch.float16, torch.float64])
@pytest.mark.parametrize('aligned', [False, True])
def test_spatial_coordinates_values_and_gradients_preserve_fp32_with_other_defaults(dtype, aligned):
    before = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        baseline = spatial_operations(aligned)
        torch.set_default_dtype(dtype)
        actual = spatial_operations(aligned)
        assert torch.get_default_dtype() == dtype
        for expected, measured in zip(baseline, actual):
            assert expected.dtype == measured.dtype == torch.float32
            torch.testing.assert_close(measured, expected, rtol=0, atol=0)
    finally:
        torch.set_default_dtype(before)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float64])
def test_actual_spatial_runner_preserves_caller_dtype_model_rng_and_records(tmp_path, monkeypatch, dtype):
    from bench import protocols
    from bench.acquisition import source_recipes
    from bench.prepare import prepare

    root = tmp_path / 'data'
    source = root / 'VOC2012'
    (source / 'ImageSets/Segmentation').mkdir(parents=True)
    (source / 'JPEGImages').mkdir()
    (source / 'SegmentationClass').mkdir()
    for split, indices in [('train', range(4)), ('val', range(4, 6))]:
        (source / 'ImageSets/Segmentation' / f'{split}.txt').write_text('\n'.join(map(str, indices)))
        for i in indices:
            Image.new('RGB', (16, 8), (20 * i, 50, 100)).save(source / 'JPEGImages' / f'{i}.jpg')
            mask = np.zeros((8, 16), dtype=np.uint8)
            mask[:, i + 1:] = 1
            Image.fromarray(mask).save(source / 'SegmentationClass' / f'{i}.png')
    declared = deepcopy(protocols.recipes())
    task = declared['pascal']
    task.update(canvas=16, bank={'train': 2, 'development': 2, 'final': 2})
    task['probe'].update(updates=2, batch_size=2)
    task['digest'] = digest({k: v for k, v in task.items() if k != 'digest'})
    sources = source_recipes()
    sources['voc2012']['counts'] = {'train': 4, 'final': 2}
    monkeypatch.setattr(protocols, 'recipes', lambda: declared)
    monkeypatch.setattr(importlib.import_module('bench.prepare'), 'source_recipes', lambda: sources)
    assert prepare(names=['pascal'], data_root=root)['ready']

    class Spatial(bench.VisionModel):
        def __init__(self):
            super().__init__()
            self.info = bench.ModelInfo('spatial-dtype-fixture', '1', {'spatial': 'rgb-grid'}, {})

        def encode_spatial(self, batch):
            affine = batch.image_to_original.clone()
            affine[:, 0, 2] -= .02
            extent = batch.images.new_tensor([0, 0, 1, 1])[None].expand(len(batch.sample_ids), -1)
            return bench.Extraction(bench.SpatialFeatures(F.avg_pool2d(batch.images, 4), affine, extent))

    model = Spatial()
    original = torch.get_default_dtype()
    reports = []
    try:
        for name, current in [('baseline', torch.float32), ('changed', dtype)]:
            torch.set_default_dtype(current)
            rng = torch.random.get_rng_state().clone()
            output = tmp_path / name
            result = bench.run(model, names=['pascal'], data_root=root, output_dir=output,
                execution=bench.Execution(device='cpu', workers=0))
            assert result.status == 'complete' and torch.get_default_dtype() == current
            assert torch.equal(rng, torch.random.get_rng_state())
            assert not (output / '.work').exists()
            reports.append(read_json(output / 'results.json'))
        first, second = reports
        assert first['model'] == second['model']
        for key in ['metrics', 'selection', 'statistics']:
            assert first['tasks'][task['id']][key] == second['tasks'][task['id']][key]
    finally:
        torch.set_default_dtype(original)
