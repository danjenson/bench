from copy import deepcopy

import pytest
import torch

import bench
from bench.prepare import prepare
from bench import protocols
from bench.storage import digest, file_digest, read_json
from test_features import GlobalAdapter


def configure():
    task = protocols.recipes()['clevr-count']
    task['probe'].update(seeds=[0, 1, 2],
        search_plan={'method': 'coordinate', 'learning_rate_ladder': [.003, .01, .03, .1],
                     'weight_decay_ladder': [0.]},
        budget_validation={'multipliers': [1, 2, 4], 'absolute_percentage_points': .1})
    task['digest'] = digest({key: value for key, value in task.items() if key != 'digest'})
    return task


def execution(work):
    return bench.Execution(device='cpu', numerics='fp32-v1', workers=0, batch_size=4, work_dir=work)


def model():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        return GlobalAdapter()


def test_full_search_matches_independent_fixed_setting_fits_and_prunes_losers(clevr_data, tmp_path):
    root, _ = clevr_data
    task = configure()
    prepare(names=['clevr-count'], data_root=root)
    adapter = model()
    workspace = tmp_path / 'controlled-work'
    output = tmp_path / 'controlled'
    controlled = bench.run(adapter, names=['clevr-count'], data_root=root, output_dir=output,
        execution=execution(workspace))
    selected_path, = (workspace / 'probes').glob('*/selection.json')
    selected = read_json(selected_path)
    exposed = read_json(output / 'results.json')
    summary = exposed['tasks'][task['id']]['selection']
    assert summary['budget_validation'] == selected['budget_validation']
    assert summary['search'] == selected['search']
    assert selected['validation_fits']
    assert {p.name for p in selected_path.parent.glob('*.pt')} == {
        'normalization.pt', *[seed['head_file'] for seed in selected['seeds']]}
    assert exposed['storage']['peak_temporary_bytes'] <= exposed['storage']['estimated_peak_temporary_bytes']
    # Refit the selected setting independently with the original fixed fitter.
    # Exact head tensors and per-image observations must agree, beyond score parity.
    original = deepcopy(task)
    task['probe'].pop('search_plan')
    task['probe'].pop('budget_validation')
    task['probe'].update(epochs=summary['budget_validation']['reporting_epochs'],
        learning_rates=[selected['seeds'][0]['learning_rate']], weight_decays=[selected['seeds'][0]['weight_decay']])
    task['digest'] = digest({key: value for key, value in task.items() if key != 'digest'})
    prepare(names=['clevr-count'], data_root=root)
    legacy_work = tmp_path / 'fixed-work'
    fixed_output = tmp_path / 'fixed'
    fixed = bench.run(adapter, names=['clevr-count'], data_root=root, output_dir=fixed_output,
        execution=execution(legacy_work))
    fixed_path, = (legacy_work / 'probes').glob('*/selection.json')
    fixed_selection = read_json(fixed_path)
    assert fixed.metrics == controlled.metrics
    assert file_digest(output / 'statistics/clevr-count-v1.json') == file_digest(fixed_output / 'statistics/clevr-count-v1.json')
    for first, second in zip(selected['seeds'], fixed_selection['seeds']):
        a = torch.load(selected_path.parent / first['head_file'], weights_only=True)
        b = torch.load(fixed_path.parent / second['head_file'], weights_only=True)
        assert a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)
    task.clear()
    task.update(original)


@pytest.mark.parametrize('corrupt_retained', [False, True])
def test_full_search_resumes_committed_fits_and_final_reuses_reporting_heads(clevr_data, tmp_path, monkeypatch, corrupt_retained):
    from bench import probes

    root, _ = clevr_data
    task = configure()
    prepare(names=['clevr-count'], data_root=root)
    adapter = model()
    workspace, output = tmp_path / 'work', tmp_path / 'result'
    original = probes.train_candidate
    completed, attempts = [], []

    def interrupt(*args, **kwargs):
        key = args[7], args[4]['probe']['epochs'], args[8], args[9]
        attempts.append(key)
        if len(attempts) == 3:
            raise RuntimeError('interrupted third fit')
        result = original(*args, **kwargs)
        completed.append(key)
        return result

    monkeypatch.setattr(probes, 'train_candidate', interrupt)
    with pytest.raises(RuntimeError, match='interrupted third fit'):
        bench.run(adapter, names=['clevr-count'], data_root=root, output_dir=output, execution=execution(workspace))
    search_path, = (workspace / 'probes').glob('*/search.json')
    journal = read_json(search_path)['record']
    assert len(journal['fits']) == 2 and len(journal['retained']) == 1
    if corrupt_retained:
        (search_path.parent / journal['retained'][0]).write_bytes(b'corrupt head')
    prior = list(completed)
    result = bench.run(adapter, names=['clevr-count'], data_root=root, output_dir=output, execution=execution(workspace))
    selection = read_json(search_path.parent / 'selection.json')
    assert ('search_restarted' in selection) is corrupt_retained
    if not corrupt_retained:
        assert len(completed) == len(set(completed)) and completed[:2] == prior
    else:
        assert completed[2:4] == prior
    assert result.status == 'complete'
    def refuse_fit(*args, **kwargs):
        pytest.fail('Final reporting refitted a verified selection')
    monkeypatch.setattr(probes, 'train_candidate', refuse_fit)
    final_output = tmp_path / 'final'
    bench.run(adapter, names=['clevr-count'], evaluation='final', data_root=root,
        output_dir=final_output, execution=execution(workspace))
    final = read_json(final_output / 'results.json')
    assert final['tasks'][task['id']]['probe_status'] == 'reused'
    assert final['storage']['retained_temporary_bytes'] == final['storage']['retained_workspace_bytes'] == 0


def test_full_spatial_search_preserves_ignore_geometry_and_selected_heads(tmp_path, monkeypatch):
    from bench import probes

    task = protocols.recipes()['pascal']
    task['probe'].pop('updates')
    task['probe'].update(epochs=1, batch_size=2, seeds=[0, 1], learning_rates=[.01],
        weight_decays=[.01], scoring='each-epoch',
        selection='best-development-score-then-earlier-epoch-then-candidate-order',
        search_plan={'method': 'coordinate', 'learning_rate_ladder': [.01], 'weight_decay_ladder': [.01]},
        budget_validation={'multipliers': [1, 2], 'absolute_percentage_points': .25})
    task['digest'] = digest({key: value for key, value in task.items() if key != 'digest'})
    rows = [{'id': str(i), 'image': f'{i}.jpg', 'mask': f'{i}.png'} for i in range(4)]
    generator = torch.Generator().manual_seed(73)
    affine = torch.tensor([[.7, 0., .1], [0., .8, .05], [0., 0., 1.]])
    values = {row['id']: {'values': torch.randn(3, 2, 3, generator=generator),
        'grid_to_original': affine, 'extent_in_original': torch.tensor([0., 0., 1., 1.])} for row in rows}
    labels = {row['id']: (torch.arange(48).reshape(6, 8) + i) % 3 for i, row in enumerate(rows)}
    for target in labels.values():
        target[0] = 255
    monkeypatch.setattr(probes, 'label_canvas', lambda root, row, task: (labels[row['id']], torch.eye(3)))

    class Bank:
        manifest = {'shape': [3, 2, 3], 'dtype': 'float32', 'bytes_per_sample': 72}
        storage = None

        def __getitem__(self, identifier):
            return values[identifier]

    bank = Bank()
    controlled_path, fixed_path = tmp_path / 'controlled', tmp_path / 'fixed'
    kwargs = dict(training=rows[:2], development=rows[2:], root=tmp_path,
        execution=bench.Execution(device='cpu', numerics='fp32-v1'), diagnostics={})
    selected = probes.fit_probe(bank, task=task, directory=controlled_path, **kwargs)
    reference_task = deepcopy(task)
    reference_task['probe'].pop('search_plan')
    reference_task['probe'].pop('budget_validation')
    reference_task['probe']['epochs'] = selected['budget_validation']['reporting_epochs']
    reference_task['digest'] = digest({key: value for key, value in reference_task.items() if key != 'digest'})
    reference = probes.fit_probe(bank, task=reference_task, directory=fixed_path, **kwargs)
    assert probes.evaluate_selection(selected, controlled_path, bank, rows[2:], tmp_path, task, kwargs['execution']) == probes.evaluate_selection(
        reference, fixed_path, bank, rows[2:], tmp_path, reference_task, kwargs['execution'])
    for a, b in zip(selected['seeds'], reference['seeds']):
        first = torch.load(controlled_path / a['head_file'], weights_only=True)
        second = torch.load(fixed_path / b['head_file'], weights_only=True)
        assert all(torch.equal(first[key], second[key]) for key in first)
    assert {p.name for p in controlled_path.glob('*.pt')} == {'normalization.pt', *[s['head_file'] for s in selected['seeds']]}
