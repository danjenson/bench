from copy import deepcopy
from pathlib import Path
import runpy

import pytest

import bench
from bench.protocols import resolve_names
from bench.storage import write_json


@pytest.fixture
def extension():
    return runpy.run_path(str(Path(__file__).parents[1] / 'scripts/extend_global_calibration.py'))


def request_file(tmp_path, extension, cases):
    task = resolve_names(['clevr-count'])[0]
    previous = tmp_path / 'previous'
    write_json(previous / 'results.json', {'tasks': {task['id']: {'selection': {
        'candidates': [{'learning_rate': 0.1, 'weight_decay': 0.0,
                        'development_score': 0.0, 'epoch': 1}]}}}})
    path = tmp_path / 'request.json'
    write_json(path, {'pins': {}, 'package_root': str(Path(bench.__file__).parent),
        'names': ['clevr-count'], 'priors': {task['id']: {'kind': 'run', 'output': str(previous)}},
        'cases': cases})
    return path


def test_extension_preserves_base_recipe_and_accepts_only_missing_candidates(tmp_path, extension):
    task = resolve_names(['clevr-count'])[0]
    original = deepcopy(task)
    case = extension['variant'](task, task['probe']['epochs'], [0.3, 1.0], [0.0, 0.1, 1.0], 'learning-rate boundary')
    request, resolved = extension['checked_request'](request_file(tmp_path, extension, [case]))
    assert task == original and resolved == [task]
    assert case['probe']['seeds'] == [0] and case['release_status'] == 'calibration-only'
    assert case['calibration']['base_task_digest'] == task['digest']
    assert request['cases'] == [case]


def test_completed_candidates_cannot_be_refitted(tmp_path, extension):
    task = resolve_names(['clevr-count'])[0]
    case = extension['variant'](task, task['probe']['epochs'], [0.1], [0.0], 'boundary')
    with pytest.raises(AssertionError, match='must not be refitted'):
        extension['checked_request'](request_file(tmp_path, extension, [case]))


@pytest.mark.parametrize('conflicting', [False, True])
def test_combined_priors_count_identical_evidence_once_and_reject_conflicts(tmp_path, extension, conflicting):
    task = resolve_names(['clevr-count'])[0]
    candidate = {'seed': 0, 'learning_rate': .01, 'weight_decay': 0.,
        'development_score': 60., 'training_loss': .5, 'epoch': 2}
    parts = []
    for index in range(2):
        directory = tmp_path / str(index)
        value = {**candidate, 'training_loss': .6} if conflicting and index else candidate
        write_json(directory / 'results.json', {'tasks': {task['id']: {'selection': {'candidates': [value]}}}})
        parts.append({'kind': 'run', 'output': str(directory)})
    prior = {'kind': 'combined', 'parts': parts}
    if conflicting:
        with pytest.raises(ValueError, match='Conflicting previous candidate evidence'):
            extension['previous_candidates'](prior, task)
    else:
        assert extension['previous_candidates'](prior, task) == [(task['probe']['epochs'], candidate)]


def test_different_case_names_cannot_duplicate_the_same_fit(tmp_path, extension):
    task = resolve_names(['clevr-count'])[0]
    cases = [extension['variant'](task, task['probe']['epochs'], [0.3], [0.0], purpose)
             for purpose in ['boundary', 'renamed boundary']]
    assert cases[0]['id'] != cases[1]['id']
    with pytest.raises(AssertionError, match='Duplicate new candidate'):
        extension['checked_request'](request_file(tmp_path, extension, cases))


def test_budget_extension_can_reuse_hyperparameters_without_refitting_old_budget(tmp_path, extension):
    task = resolve_names(['clevr-count'])[0]
    case = extension['variant'](task, 2 * task['probe']['epochs'], [0.1], [0.0], 'doubled budget')
    extension['checked_request'](request_file(tmp_path, extension, [case]))


def test_changed_predeclared_case_is_rejected(tmp_path, extension):
    task = resolve_names(['clevr-count'])[0]
    case = extension['variant'](task, task['probe']['epochs'], [0.3], [0.0], 'boundary')
    case['probe']['seeds'] = [1]
    with pytest.raises(AssertionError):
        extension['checked_request'](request_file(tmp_path, extension, [case]))


def test_budget_doubling_requires_the_selected_setting_and_an_unresolved_change(extension):
    case = {'probe': {'learning_rates': [0.001], 'weight_decays': [1.0]}}
    condition = {'kind': 'winner-needs-budget', 'reference_epochs': 50,
                 'compare_epochs': [50, 100], 'tolerance': 0.1}
    winner = {'learning_rate': 0.001, 'weight_decay': 1.0, 'development_score': 94.0, 'epoch': 49}
    assert not extension['needed'](case, condition, [(50, winner)])
    assert not extension['needed'](case, condition, [(50, winner), (100, {**winner, 'development_score': 94.05})])
    known = [(50, winner), (100, {**winner, 'development_score': 94.2})]
    assert extension['needed'](case, condition, known)
    assert not extension['needed']({'probe': {'learning_rates': [0.003], 'weight_decays': [1.0]}}, condition, known)


def test_conditional_coordinate_search_uses_the_current_joint_winner(extension):
    winner = {'learning_rate': 0.0003, 'weight_decay': 1.0, 'development_score': 94.1, 'epoch': 49}
    known = [(50, winner), (50, {**winner, 'learning_rate': 0.001, 'development_score': 94.0})]
    condition = {'kind': 'winner-parameters', 'reference_epochs': 50,
                 'parameters': {'learning_rate': 0.0003, 'weight_decay': 1.0}}
    assert extension['needed']({}, condition, known)
    assert not extension['needed']({}, {**condition, 'parameters': {'learning_rate': 0.0001}}, known)
    known.append((50, {**winner, 'weight_decay': 10.0, 'development_score': 94.2}))
    assert not extension['needed']({}, condition, known)
    assert extension['needed']({}, {**condition, 'parameters': {'learning_rate': 0.0003, 'weight_decay': 10.0}}, known)


@pytest.mark.parametrize('parameters', [{}, {'epochs': 50}, {'learning_rate': 0}, {'weight_decay': float('nan')}])
def test_conditional_coordinate_policy_rejects_invalid_axes(tmp_path, extension, parameters):
    from bench.storage import read_json

    task = resolve_names(['clevr-count'])[0]
    case = extension['variant'](task, 50, [0.3], [0.0], 'conditional coordinate')
    path = request_file(tmp_path, extension, [case])
    request = read_json(path)
    request['conditions'] = {case['id']: {'kind': 'winner-parameters', 'reference_epochs': 50, 'parameters': parameters}}
    if any(isinstance(v, float) and v != v for v in parameters.values()):
        # The JSON storage boundary already rejects non-finite declarations.
        with pytest.raises(ValueError):
            write_json(path, request)
    else:
        write_json(path, request)
        with pytest.raises(AssertionError):
            extension['checked_request'](path)


@pytest.mark.parametrize('conditional', [False, True])
def test_missing_candidate_extraction_fit_reconstruction_and_cleanup(clevr_data, tmp_path, extension, monkeypatch, conditional):
    import torch
    from bench import Execution
    from bench.features import model_identity
    from bench.prepare import prepare
    from bench.storage import read_json
    from test_features import GlobalAdapter

    # This synthetic fit shares pytest's process. Production verification runs
    # in a fresh process with CUDA_VISIBLE_DEVICES='' in the frozen caller.
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(torch.cuda, 'is_initialized', lambda: False)
    root, _ = clevr_data
    prepare(names=['clevr-count'], data_root=root)
    task = resolve_names(['clevr-count'])[0]
    case = extension['variant'](task, task['probe']['epochs'], [0.3], [0.0], 'boundary')
    path = request_file(tmp_path, extension, [case])
    model = GlobalAdapter()
    yaml = tmp_path / 'model.yaml'
    yaml.write_text('{}')
    request = read_json(path)
    if conditional:
        second = extension['variant'](task, 2 * task['probe']['epochs'], [0.3], [0.0], 'conditional budget')
        third = extension['variant'](task, task['probe']['epochs'], [1.0], [0.0], 'unneeded tail')
        request['cases'] += [second, third]
        request['conditions'] = {
            second['id']: {'kind': 'winner-parameters', 'reference_epochs': task['probe']['epochs'], 'parameters': {'weight_decay': 0.0}},
            third['id']: {'kind': 'winner-parameters', 'reference_epochs': task['probe']['epochs'], 'parameters': {'learning_rate': 1.0}}}
    request.update(output=str(tmp_path / 'new-calibration'), gpu_uuid='fixture', cpu_threads=1,
        model_yaml=str(yaml), data_root=str(root), numerics='fp32-v1', workers=0, batch_size=4,
        feature_width=5, expected_model=model_identity(model, Execution(device='cpu', numerics='fp32-v1')))
    write_json(path, request)
    globals_ = extension['run'].__globals__
    monkeypatch.setitem(globals_, 'instantiate', lambda *args, **kwargs: model)
    monkeypatch.setattr(model, 'cuda', lambda: model)
    monkeypatch.setattr(bench, 'Execution', lambda **kwargs: Execution(**{**kwargs, 'device': 'cpu'}))
    monkeypatch.setattr(globals_['subprocess'], 'check_output',
        lambda command, **kwargs: 'fixture' if command[1] == '--query-gpu=uuid' else '')
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'CPU fixture')
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    previous_threads = torch.get_num_threads()
    try:
        extension['run'](path)
        proof = extension['verify'](path)
    finally:
        torch.set_num_threads(previous_threads)
    assert proof['status'] == 'verified' and proof['verified_records'] == (6 if conditional else 3)
    assert len(proof['skipped']) == int(conditional)
    assert model.calls == 3 and not (Path(request['output']) / '.work').exists()
    stats = Path(request['output']) / 'statistics' / (case['id'] + '.json')
    saved = read_json(stats)
    saved['records'][0]['prediction'] = -1
    write_json(stats, saved)
    with pytest.raises(AssertionError):
        extension['verify'](path)
