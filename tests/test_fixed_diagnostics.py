"""Explicit diagnostics must not inherit the full automatic probe controller."""
from copy import deepcopy
from pathlib import Path
import runpy

import torch

import bench
from bench.prepare import prepare
from bench.protocols import recipes
from bench.storage import digest, read_json
from test_features import GlobalAdapter


def test_missing_fit_variant_drops_the_base_controller_without_mutation():
    task = recipes()['clevr-count']
    assert 'search_plan' in task['probe'] and 'budget_validation' in task['probe']
    before = deepcopy(task)
    variant = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/extend_global_calibration.py'))['variant']
    explicit = variant(task, 50, [.3], [0.], 'one declared boundary fit')
    assert task == before
    assert explicit['calibration']['base_task_digest'] == task['digest']
    assert 'search_plan' not in explicit['probe'] and 'budget_validation' not in explicit['probe']
    assert explicit['probe']['epochs'] == 50 and explicit['probe']['learning_rates'] == [.3]
    assert explicit['probe']['weight_decays'] == [0.] and explicit['probe']['seeds'] == [0]


def test_actual_diagnostics_fit_only_declared_grids_with_a_controlled_base(clevr_data, tmp_path, monkeypatch):
    from bench import protocols

    root, _ = clevr_data
    task = protocols.recipes()['clevr-count']
    task['probe'].update(search_plan={'method': 'coordinate', 'learning_rate_ladder': [.01, .03],
        'weight_decay_ladder': [0.]}, budget_validation={'multipliers': [1, 2], 'absolute_percentage_points': .1})
    task['digest'] = digest({k: v for k, v in task.items() if k != 'digest'})
    before = deepcopy(task)
    assert prepare(names=['clevr-count'], data_root=root)['ready']
    for name, value in [('get_device_name', lambda device: 'CPU fixture'),
        ('reset_peak_memory_stats', lambda device: None), ('max_memory_allocated', lambda device: 0)]:
        monkeypatch.setattr(torch.cuda, name, value)
    measure = runpy.run_path(str(Path(__file__).parents[1] / 'examples/calibrate_monitor_probes.py'))['measure']
    output = tmp_path / 'diagnostic'
    result = measure(GlobalAdapter(), names=['clevr-count'], data_root=root, output=output,
        execution=bench.Execution(device='cpu', workers=0, batch_size=4), stage='search-boundary')
    plan = read_json(output / 'plan.json')
    assert len(plan['variants']) == len(result['results']) == 6
    assert protocols.recipes()['clevr-count'] == before
    for variant in plan['variants']:
        assert 'search_plan' not in variant['probe'] and 'budget_validation' not in variant['probe']
        measured = result['results'][variant['id']]
        assert len(measured['candidate_scores']) == 1
        selected, = measured['candidate_scores']
        assert selected['learning_rate'] == variant['probe']['learning_rates'][0]
        assert selected['weight_decay'] == variant['probe']['weight_decays'][0]
    assert result['retained_temporary_bytes'] == 0 and not (output / '.work').exists()
