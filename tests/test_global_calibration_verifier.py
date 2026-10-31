from copy import deepcopy
from pathlib import Path
import runpy

import pytest


@pytest.fixture
def comparison():
    module = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/verify_global_calibration.py'))
    candidate = {'seed': 0, 'epoch': 41, 'updates': 2542, 'learning_rate': 0.001,
                 'weight_decay': 0.0, 'development_score': 55.5, 'probe_parameters': 6152,
                 'training_loss': 0.98, 'development_loss': 1.02, 'head_sha256': 'a' * 64}
    return module['compare_candidates'], candidate, ({'gpu': 'r2'}, {'gpu': 'original'})


def test_explicit_gpu_transition_keeps_exact_scores_but_preserves_changed_hashes(comparison):
    compare, old, hardware = comparison
    new = {**old, 'head_sha256': 'b' * 64, 'development_loss': old['development_loss'] + 6e-8}
    result = compare([new], [old], hardware_pair=hardware)
    assert result['scores_and_selection_metadata_identical']
    assert result['hardware_transition'] and not result['head_hashes_identical']
    assert 0 < result['largest_absolute_loss_difference'] < 1e-7
    with pytest.raises(AssertionError, match='Same-hardware'):
        compare([new], [old])
    with pytest.raises(AssertionError, match='GPU transition'):
        compare([new], [old], hardware_pair=({'gpu': 'same'}, {'gpu': 'same'}))


@pytest.mark.parametrize('field,value', [('seed', 1), ('epoch', 42), ('updates', 2604),
    ('learning_rate', 0.003), ('weight_decay', 0.1), ('development_score', 55.500000001), ('probe_parameters', 6153)])
def test_score_and_selection_drift_cannot_be_hidden_as_a_hardware_change(comparison, field, value):
    compare, old, hardware = comparison
    with pytest.raises(AssertionError, match='score or selection'):
        compare([{**old, field: value}], [old], hardware_pair=hardware)


@pytest.mark.parametrize('field,value', [('training_loss', float('nan')),
    ('development_loss', float('inf')), ('training_loss', 0.99), ('head_sha256', 'invalid')])
def test_invalid_hashes_or_large_loss_changes_are_rejected(comparison, field, value):
    compare, old, hardware = comparison
    with pytest.raises(AssertionError):
        compare([{**old, field: value}], [old], hardware_pair=hardware)


def test_missing_candidates_and_unrecognized_record_fields_are_rejected(comparison):
    compare, old, hardware = comparison
    for actual in ([], [old, old], [{**old, 'new_field': 1}]):
        with pytest.raises(AssertionError):
            compare(actual, [old], hardware_pair=hardware)
    assert compare([deepcopy(old)], [old])['head_hashes_identical']
