"""Compare bounded resampling with the uncached canonical geometry sampler."""
import importlib.util
import os
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from bench import probes


# Reload this package's fitter so the comparison can replace its sampler without
# mutating the implementation under test or importing historical acceptance files.
spec = importlib.util.spec_from_file_location('bench._uncached_resampling_reference', Path(probes.__file__))
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)


class UncachedResampling:
    def __init__(self, task, device):
        pass

    def __call__(self, sample, logits, grid_to_original, canvas_to_original, size):
        return original.logits_on_canvas(logits, grid_to_original, canvas_to_original, size)

    def __enter__(self):
        return self

    def __exit__(self, *error):
        pass


original.FitResampling = UncachedResampling
torch.set_num_threads(1)
DEVICE = os.environ.get("BENCH_RESAMPLING_TEST_DEVICE", "cpu")
assert DEVICE in ["cpu", "cuda"]
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False


def transforms(kind):
    canvas = torch.tensor([[327.3, 0., 12.], [0., 221.7, -9.], [0., 0., 1.]])
    warp = torch.eye(3)
    if kind == 'shift':
        warp[:2, 2] = torch.tensor([-.096875, -.096875])
    elif kind == 'border':
        warp[0, 0], warp[1, 1] = .71, .86
        warp[:2, 2] = torch.tensor([-.2, .3])
    elif kind == 'flip':
        warp[0, 0] = -1
        warp[:2, 2] = torch.tensor([1., .1])
    elif kind == 'shear':
        warp[0, 1] = .12
    return canvas @ warp, canvas


@pytest.mark.parametrize('shape', [(1, 1), (1, 5), (2, 3), (5, 5), (3, 8), (9, 5)])
@pytest.mark.parametrize('kind', ['identity', 'shift', 'border', 'flip', 'shear'])
def test_forward_gradient_and_dispatch_match_original(shape, kind):
    generator = torch.Generator().manual_seed(71)
    values = torch.randn(4, *shape, generator=generator)
    target = torch.randint(4, (31, 47), generator=generator).to(DEVICE)
    matrix, canvas = transforms(kind)
    outputs, gradients = [], []
    for module in [original, probes]:
        logits = values.clone().to(DEVICE).requires_grad_()
        with module.FitResampling({'kind': 'segmentation'}, DEVICE) as sampler:
            result = sampler('sample', logits, matrix, canvas, target.shape)
            if module is probes:
                payloads = [v[0] for v in sampler.entries.values()]
                assert ('small-weights' in payloads) == (max(shape) <= 8 and kind in ['shift', 'border', 'flip'])
            F.cross_entropy(result[None], target[None]).backward()
            outputs.append(result.detach())
            gradients.append(logits.grad.detach())
    torch.testing.assert_close(outputs[1], outputs[0], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(gradients[1], gradients[0], atol=2e-6, rtol=2e-4)
    if max(shape) > 8 or kind in ['identity', 'shear']:
        assert torch.equal(outputs[0], outputs[1])
        if DEVICE == 'cpu':
            assert torch.equal(gradients[0], gradients[1])
    assert torch.cuda.is_initialized() == (DEVICE == "cuda")


def test_actual_five_by_five_and_160_canvas_forward_and_gradient():
    generator = torch.Generator().manual_seed(17)
    raw = torch.randn(21, 5, 5, generator=generator)
    target = torch.randint(21, (160, 160), generator=generator).to(DEVICE)
    target[:11, :13] = 255
    matrix, canvas = transforms('shift')
    runs = []
    for module in [original, probes]:
        logits = raw.clone().to(DEVICE).requires_grad_()
        with module.FitResampling({'kind': 'segmentation'}, DEVICE) as sampler:
            values = sampler('actual-shape', logits, matrix, canvas, target.shape)
            F.cross_entropy(values[None], target[None], ignore_index=255).backward()
            runs.append((values.detach(), logits.grad.detach()))
    torch.testing.assert_close(runs[0][0], runs[1][0], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(runs[0][1], runs[1][1], atol=2e-6, rtol=2e-4)


def test_mixed_shared_and_small_weight_cache_is_accounted_and_pruned():
    generator = torch.Generator().manual_seed(9)
    logits = torch.randn(3, 5, 5, generator=generator).to(DEVICE)
    with probes.FitResampling({'kind': 'segmentation'}, DEVICE) as sampler:
        for name, kind in [('aligned', 'identity'), ('shifted', 'shift'), ('flipped', 'flip')]:
            matrix, canvas = transforms(kind)
            sampler(name, logits, matrix, canvas, (31, 47))
        tensors = {}
        for kind, values in sampler.entries.values():
            for tensor in (values if kind in ['weights', 'small-weights'] else [values]):
                tensors[id(tensor)] = tensor
        assert sampler.current_bytes == sum(t.numel() * t.element_size() for t in tensors.values())
        assert sampler.current_bytes <= sampler.limit
        assert len({v[0] for v in sampler.entries.values()}) == 2
    assert sampler.current_bytes == 0 and sampler.entries == {}


def test_actual_adamw_probe_fit_preserves_head_selection_records_and_rng(tmp_path, monkeypatch):
    generator = torch.Generator().manual_seed(314)
    matrix, canvas = transforms('shift')
    rows = [{'id': str(i)} for i in range(5)]
    raw = {r['id']: {'values': torch.randn(4, 5, 5, generator=generator),
                     'grid_to_original': matrix} for r in rows}
    labels = {r['id']: torch.randint(3, (16, 16), generator=generator) for r in rows}

    class Bank:
        manifest = {'shape': [4, 5, 5]}
        storage = None

        def __getitem__(self, key):
            return raw[key]

    task = {'kind': 'segmentation', 'classes': 3,
            'probe': {'batch_size': 2, 'betas': [.9, .999], 'epsilon': 1e-8,
                      'amsgrad': False, 'foreach': False, 'fused': False,
                      'updates': 7, 'warmup_fraction': .1, 'final_lr_fraction': .001,
                      'selection': 'final-update'}}
    statistics = {'mean': torch.zeros(4), 'std': torch.ones(4)}
    runs = []
    rng = torch.random.get_rng_state().clone()
    for name, module in [('original', original), ('candidate', probes)]:
        monkeypatch.setattr(module, 'label_canvas', lambda root, row, task: (labels[row['id']], canvas))
        path = tmp_path / (name + '.pt')
        selected = module.train_candidate(Bank(), rows[:3], rows[3:], tmp_path, task,
                                          statistics, DEVICE, 7, .003, .01, path)
        state = torch.load(path, map_location='cpu', weights_only=True)
        probe = module.LinearProbe(4, 3, 7, DEVICE)
        probe.load_state_dict(state)
        score, records = module.score(probe, Bank(), rows[3:], tmp_path, task, statistics, DEVICE, 7)
        runs.append((selected, state, score, records))
    for key in runs[0][1]:
        torch.testing.assert_close(runs[0][1][key], runs[1][1][key], atol=1e-5, rtol=0)
    for key in ['seed', 'epoch', 'updates', 'learning_rate', 'weight_decay', 'development_score']:
        assert runs[0][0][key] == runs[1][0][key]
    assert runs[0][3] == runs[1][3]
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert torch.cuda.is_initialized() == (DEVICE == "cuda")
