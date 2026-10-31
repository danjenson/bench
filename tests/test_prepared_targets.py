from copy import deepcopy

import pytest
import torch

from bench.probes import PreparedTargets


@pytest.fixture
def targets(monkeypatch):
    calls = []

    def load(root, row, task):
        calls.append(row['id'])
        value = 2 if row.get('files', {}).get('mask') == 'changed' else 1
        return torch.tensor([[0, value], [255, value]]), torch.eye(3)

    monkeypatch.setattr('bench.probes.label_canvas', load)
    monkeypatch.setattr(PreparedTargets, 'byte_limit', 80)
    rows = [{'id': name, 'image': name + '.jpg', 'mask': name + '.png', 'files': {'mask': 'initial'}}
            for name in 'ABC']
    return PreparedTargets('.', {}), rows, calls


def test_lossless_targets_reuse_sources_obey_byte_bound_and_release_on_exit(targets):
    cache, (a, b, c), calls = targets
    with cache:
        packed, affine = cache.read(a)
        assert packed.dtype == torch.uint8
        assert torch.equal(packed.long(), torch.tensor([[0, 1], [255, 1]]))
        assert torch.equal(affine, torch.eye(3))
        cache.read(b)
        cache.read(a)
        cache.read(c)
        cache.read(b)
        assert calls == ['A', 'B', 'C', 'B']
        assert cache.current_bytes == 80
    assert not cache.entries and cache.current_bytes == 0


def test_changed_annotation_digest_invalidates_the_cached_target(targets):
    cache, (a, _, _), calls = targets
    first = cache.read(a)[0]
    changed = deepcopy(a)
    changed['files']['mask'] = 'changed'
    second = cache.read(changed)[0]
    assert calls == ['A', 'A']
    assert int(first[0, 1]) == 1 and int(second[0, 1]) == 2
    cache.close()


def test_oversized_target_does_not_evict_useful_entries(targets, monkeypatch):
    cache, (a, b, _), _ = targets
    cache.read(a)
    monkeypatch.setattr('bench.probes.label_canvas', lambda *args: (torch.zeros(20, 20, dtype=torch.long), torch.eye(3)))
    assert cache.read(b)[0].shape == (20, 20)
    assert cache.current_bytes == 40 and len(cache.entries) == 1
    cache.close()


def test_out_of_range_labels_are_preserved_for_normal_loss_validation(targets, monkeypatch):
    cache, (a, _, _), _ = targets
    original = torch.tensor([[-1, 256], [255, 0]])
    monkeypatch.setattr('bench.probes.label_canvas', lambda *args: (original, torch.eye(3)))
    value, _ = cache.read(a)
    assert value.dtype == torch.long and torch.equal(value, original)
    cache.close()


def test_cache_is_released_on_failure(targets):
    cache, (a, _, _), _ = targets
    with pytest.raises(RuntimeError, match='fixture failure'):
        with cache:
            cache.read(a)
            raise RuntimeError('fixture failure')
    assert not cache.entries and cache.current_bytes == 0
