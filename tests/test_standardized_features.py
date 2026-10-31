from copy import deepcopy

import pytest
import torch

from bench.probes import StandardizedFeatures, global_values, spatial_values, standardize


class Bank:
    def __init__(self, spatial=False):
        self.manifest = {'shape': [2, 2, 2] if spatial else [2]}
        self.entries = {name: {'values': torch.arange(8 if spatial else 2, dtype=torch.bfloat16).reshape(self.manifest['shape']) + i,
                              'grid_to_original': torch.eye(3), 'extent_in_original': torch.ones(4)}
                        for i, name in enumerate(['train', 'dev'])}
        self.reads = 0

    def __getitem__(self, name):
        self.reads += 1
        return self.entries[name]

    def read_values(self, ids):
        self.reads += 1
        return torch.stack([self.entries[name]['values'] for name in ids])


@pytest.mark.parametrize('spatial', [False, True])
def test_fit_local_cache_reuses_normalized_features_preserves_bank_and_cleans_up(spatial):
    bank = Bank(spatial)
    original = deepcopy(bank.entries)
    stats = {'mean': torch.tensor([1.25, 3.5]), 'std': torch.tensor([0.25, 2.0])}
    cache = StandardizedFeatures(bank, [{'id': 'train'}], [{'id': 'dev'}], stats, 'cpu')
    with cache:
        assert cache.enabled
        count = bank.reads
        for name in ['dev', 'train', 'dev']:
            if spatial:
                feature, value = spatial_values(bank, name, stats, 'cpu', cache)
                assert torch.equal(value, standardize(original[name]['values'], stats))
                feature['values'].fill_(0)
                feature['grid_to_original'].fill_(0)
            else:
                value = global_values(bank, [name], stats, 'cpu', cache, scoring=name == 'dev')
                assert torch.equal(value[0], standardize(original[name]['values'], stats))
                value.fill_(0)
        assert bank.reads == count
    assert not cache.enabled and cache.indices == {} and cache.geometry == []
    for name in original:
        for key in original[name]:
            assert torch.equal(original[name][key], bank.entries[name][key])


def test_over_limit_fallback_reads_original_bank_without_retaining_features(monkeypatch):
    bank = Bank()
    stats = {'mean': torch.zeros(2), 'std': torch.ones(2)}
    monkeypatch.setattr(StandardizedFeatures, 'cpu_byte_limit', 1)
    with StandardizedFeatures(bank, [{'id': 'train'}], [{'id': 'dev'}], stats, 'cpu') as cache:
        assert not cache.enabled and bank.reads == 0
        for _ in range(2):
            assert torch.equal(global_values(bank, ['train'], stats, 'cpu', cache)[0], bank.entries['train']['values'].float())
        assert bank.reads == 2
    assert cache.values is None


def test_cache_initialization_and_body_failures_release_owned_tensors(monkeypatch):
    stats = {'mean': torch.zeros(2), 'std': torch.ones(2)}
    bank = Bank()
    cache = StandardizedFeatures(bank, [{'id': 'train'}], [{'id': 'dev'}], stats, 'cpu')
    with pytest.raises(RuntimeError, match='fit interrupted'):
        with cache:
            assert cache.enabled
            raise RuntimeError('fit interrupted')
    assert cache.values is None and not cache.indices
    monkeypatch.setattr(bank, 'read_values', lambda ids: (_ for _ in ()).throw(RuntimeError('bank interrupted')))
    with pytest.raises(RuntimeError, match='bank interrupted'):
        with cache:
            pass
    assert cache.values is None and not cache.indices


def test_cache_allocation_failure_falls_back_to_streaming(monkeypatch):
    bank = Bank()
    stats = {'mean': torch.zeros(2), 'std': torch.ones(2)}
    monkeypatch.setattr(torch, 'empty', lambda *a, **k: (_ for _ in ()).throw(torch.cuda.OutOfMemoryError('allocation')))
    with StandardizedFeatures(bank, [{'id': 'train'}], [{'id': 'dev'}], stats, 'cpu') as cache:
        assert not cache.enabled and not cache.indices and cache.info['reason'] == 'streaming-fallback-allocation'
        assert torch.equal(global_values(bank, ['dev'], stats, 'cpu', cache)[0], bank.entries['dev']['values'].float())
