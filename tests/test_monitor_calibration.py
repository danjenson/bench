from copy import deepcopy
from pathlib import Path
import runpy

import pytest
import torch


pytestmark = pytest.mark.usefixtures('historical_monitor_recipes')


@pytest.fixture
def dense():
    module = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/monitor_calibration.py'))
    records = [{'eligible_pixels': 3, 'intersection': [1, 1], 'union': [2, 2], 'prediction_sha256': 'a' * 64}]
    return module['dense_score'], records, [torch.tensor([[0, 0], [1, 255]])]


def test_dense_reconstruction_pools_source_counts_and_ignores_void(dense):
    score, records, targets = dense
    assert score(records, targets, 2) == 50
    second = {'eligible_pixels': 1, 'intersection': [0, 0], 'union': [1, 1], 'prediction_sha256': 'b' * 64}
    assert score(records + [second], targets + [torch.tensor([[0, 255]])], 2) == pytest.approx(100 / 3)


@pytest.mark.parametrize('field,value', [('eligible_pixels', 4), ('intersection', [3, 0]),
    ('union', [0, 2]), ('union', [3, 2]), ('intersection', [1.0, 1]), ('prediction_sha256', 'invalid')])
def test_impossible_dense_statistics_are_rejected_against_source_labels(dense, field, value):
    score, records, targets = dense
    changed = deepcopy(records)
    changed[0][field] = value
    with pytest.raises((AssertionError, ValueError)):
        score(changed, targets, 2)


def test_candidate_budget_keeps_completed_variant_identities_for_both_tasks():
    from bench.protocols import resolve_names

    declare = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/monitor_calibration.py'))['declared_variants']
    tasks = resolve_names(['imagenet-100', 'pascal'])
    selected = declare(tasks, 'candidate-budget')
    assert len(selected) == 4
    for task in tasks:
        current = [v for v in selected if v['calibration']['base_task'] == task['id']]
        expected_updates = {400, 800} if task['kind'] == 'classification' else {1600, 3200}
        assert {v['probe']['updates'] for v in current} == expected_updates
        previous = declare([task], 'convergence' if task['kind'] == 'classification' else 'long-convergence')
        for variant in current:
            assert variant == next(v for v in previous if v['id'] == variant['id'])


def test_longer_candidate_keeps_overlap_identity_and_changes_only_update_budget():
    from bench.protocols import resolve_names

    declare = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/monitor_calibration.py'))['declared_variants']
    tasks = resolve_names(['imagenet-100', 'pascal'])
    previous = declare(tasks, 'candidate-budget')
    for task in tasks:
        cases = declare([task], 'candidate-long-budget')
        expected_updates = [800, 1600] if task['kind'] == 'classification' else [3200, 6400]
        assert [v['probe']['updates'] for v in cases] == expected_updates
        assert cases[0] == next(v for v in previous if v['id'] == cases[0]['id'])
        normalized = []
        for case in cases:
            case = deepcopy(case)
            for key in ['id', 'name', 'digest']:
                case.pop(key)
            case['probe'].pop('updates')
            case['calibration'].pop('budget_multiplier')
            normalized.append(case)
        assert normalized[0] == normalized[1]


def test_regularized_dense_candidate_declares_weight_decay_without_changing_base_recipe():
    from bench.protocols import resolve_names

    declare = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/monitor_calibration.py'))['declared_variants']
    task = resolve_names(['pascal'])[0]
    original = deepcopy(task)
    variants = declare([task], 'candidate-regularized')
    assert task == original and [v['probe']['updates'] for v in variants] == [1600, 3200]
    for v in variants:
        assert v['probe']['weight_decays'] == [1.0] and v['probe']['learning_rates'] == [0.01]
        assert v['probe']['selection'] == task['probe']['selection'] == 'final-update'
        assert v['bank'] == task['bank'] and v['canvas'] == task['canvas']
        assert v['calibration']['weight_decay_reference'] == task['probe']['weight_decays']
        assert v['calibration']['weight_decay_candidate'] == 1.0 and v['id'].endswith('-wd1')
    classifier = declare(resolve_names(['imagenet-100']), 'candidate-regularized')
    assert [v['probe']['updates'] for v in classifier] == [400, 800]
    assert all(v['probe']['weight_decays'] == [1.0] and v['probe']['learning_rates'] == [0.03] for v in classifier)
    assert all(v['probe']['selection'] == 'final-update' for v in classifier)


def test_regularized_tail_fits_only_missing_budget_and_retains_completed_overlap():
    from bench.protocols import resolve_names

    declare = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/monitor_calibration.py'))['declared_variants']
    task = resolve_names(['pascal'])[0]
    original = declare([task], 'candidate-regularized')
    longer = declare([task], 'candidate-regularized-long')
    tail = declare([task], 'candidate-regularized-tail')
    assert [v['probe']['updates'] for v in longer] == [3200, 6400]
    assert longer[0] == original[1] and tail == [longer[1]]
    classifier = resolve_names(['imagenet-100'])
    assert declare(classifier, 'candidate-regularized-long') == declare(classifier, 'candidate-regularized')
    with pytest.raises(AssertionError, match='segmentation only'):
        declare(classifier, 'candidate-regularized-tail')


def test_selected_monitor_preserves_existing_cases_and_declares_new_rates():
    from bench.protocols import resolve_names

    declare = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/monitor_calibration.py'))['declared_variants']
    tasks = resolve_names(['imagenet-100', 'pascal'])
    settings = {task['id']: {'base_updates': 400 if task['kind'] == 'classification' else 1600,
                             'learning_rate_multipliers': [10], 'weight_decay': 1.0} for task in tasks}
    assert declare(tasks, 'selected-monitor', monitor_recipe=settings) == declare(tasks, 'candidate-regularized')
    settings[tasks[1]['id']]['learning_rate_multipliers'] = [30, 100]
    cases = declare(tasks, 'selected-monitor', monitor_recipe=settings)
    assert len(cases) == 6 and [c['probe']['learning_rates'][0] for c in cases[2:]] == [0.03, 0.03, 0.1, 0.1]
    assert all(c['bank'] == tasks[1]['bank'] and c['probe']['selection'] == 'final-update' for c in cases[2:])


@pytest.mark.parametrize('setting', [
    {'base_updates': 1599, 'learning_rate_multipliers': [30], 'weight_decay': 1.0},
    {'base_updates': 1600.0, 'learning_rate_multipliers': [30], 'weight_decay': 1.0},
    {'base_updates': 1600, 'learning_rate_multipliers': [True], 'weight_decay': 1.0},
    {'base_updates': 1600, 'learning_rate_multipliers': [30, 30], 'weight_decay': 1.0},
    {'base_updates': 1600, 'learning_rate_multipliers': [30], 'weight_decay': -1.0},
    {'base_updates': 1600, 'learning_rate_multipliers': [30], 'weight_decay': 1.0, 'canvas': 256},
])
def test_selected_monitor_rejects_hidden_or_invalid_settings(setting):
    from bench.protocols import resolve_names

    declare = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/monitor_calibration.py'))['declared_variants']
    tasks = resolve_names(['pascal'])
    with pytest.raises(AssertionError):
        declare(tasks, 'selected-monitor', monitor_recipe={tasks[0]['id']: setting})


def test_selected_fuller_recipe_reuses_canonical_identities_without_repeating_the_rate_grid():
    from bench.protocols import resolve_names

    declare = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/monitor_calibration.py'))['declared_variants']
    tasks = resolve_names(['imagenet-100', 'pascal'])
    counts = dict(zip([task['id'] for task in tasks], [124189, 1208]))
    recipe = {task['id']: {'base_epochs': 50 if task['kind'] == 'classification' else 80,
                           'learning_rate_multiplier': 10 if task['kind'] == 'classification' else 3} for task in tasks}
    originals = deepcopy(tasks)
    selected = declare(tasks, 'selected-fuller', counts, recipe)
    grid = declare(tasks, 'fuller-selection', counts)
    assert len(selected) == 4 and tasks == originals
    for task in tasks:
        cases = [v for v in selected if v['calibration']['base_task'] == task['id']]
        assert [v['calibration']['reference_epochs'] for v in cases] == ([50, 100] if task['kind'] == 'classification' else [80, 160])
        assert all(v['probe']['seeds'] == [0] and v['probe']['scoring'] == 'each-epoch' for v in cases)
        assert cases[0] == next(v for v in grid if v['id'] == cases[0]['id'])
        assert all(v['calibration']['learning_rate_multiplier'] == recipe[task['id']]['learning_rate_multiplier'] for v in cases)


@pytest.mark.parametrize('setting', [
    {'base_epochs': 25, 'learning_rate_multiplier': 10},
    {'base_epochs': 50.0, 'learning_rate_multiplier': 10},
    {'base_epochs': 50, 'learning_rate_multiplier': 2},
    {'base_epochs': 50, 'learning_rate_multiplier': True},
    {'base_epochs': 50, 'learning_rate_multiplier': 10, 'canvas': 16},
])
def test_selected_fuller_rejects_hidden_or_undeclared_changes(setting):
    from bench.protocols import resolve_names

    declare = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/monitor_calibration.py'))['declared_variants']
    task = resolve_names(['imagenet-100'])[0]
    with pytest.raises(AssertionError):
        declare([task], 'selected-fuller', {task['id']: 124189}, {task['id']: setting})
