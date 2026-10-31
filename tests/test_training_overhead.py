from copy import deepcopy
from pathlib import Path
import random
import runpy
from types import SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def harness():
    return runpy.run_path(str(Path(__file__).parents[1] / 'scripts/measure_training_overhead.py'))


def test_training_fingerprints_cover_optimizer_rng_and_actual_encoder_updates(harness):
    def trial():
        random.seed(4)
        np.random.seed(4)
        torch.manual_seed(4)
        model, head = torch.nn.Linear(3, 4), torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD([*model.parameters(), *head.parameters()], lr=0.01, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
        generator = torch.Generator().manual_seed(7)
        initial = harness['training_state'](model, head, optimizer, scheduler, generator, cuda=False)
        inputs = torch.randn(5, 3)
        labels = torch.tensor([0, 1, 0, 1, 0])
        for _ in range(2):
            optimizer.zero_grad()
            torch.nn.functional.cross_entropy(head(model(inputs)), labels).backward()
            optimizer.step()
            scheduler.step()
        final = harness['training_state'](model, head, optimizer, scheduler, generator, cuda=False)
        return initial, final

    first, second = trial(), trial()
    assert first == second
    assert all(first[0][key] != first[1][key] for key in ['encoder', 'head', 'optimizer', 'scheduler', 'torch_rng'])
    digest = harness['fingerprint']
    assert digest(torch.ones(2)) != digest(torch.ones(2).double())
    assert digest(torch.ones(2)) != digest(torch.ones(1, 2))
    assert digest({1: 'a', '1': 'b'}) == digest({'1': 'b', 1: 'a'})
    assert digest(np.array([1, 2])) != digest(np.array([2, 1]))


@pytest.mark.parametrize('finish_after_steps,expected_wall,on_cadence', [(1, 7, True), (4, 12, False)])
def test_timing_counts_snapshot_collection_and_wait_and_preserves_cadence(harness, finish_after_steps, expected_wall, on_cadence):
    now = [0]
    steps = []
    def advance(seconds):
        now[0] += seconds

    class Process:
        returncode = None
        def poll(self):
            if len(steps) >= finish_after_steps:
                self.returncode = 0
            return self.returncode
        def wait(self):
            advance(5)
            self.returncode = 0
            return 0

    def step(index):
        steps.append(index)
        advance(1)
    report = harness['timed_interval'](steps=3, synchronize=lambda: None,
        snapshot=lambda: advance(2), launch=Process, train_step=step,
        collect=lambda: advance(2), clock=lambda: now[0])
    assert steps == [0, 1, 2]
    assert report['wall_seconds'] == expected_wall
    assert report['training_steps_wall_seconds'] == 5
    assert report['snapshot_seconds'] == 2
    assert report['result_before_next_step_interval'] is on_cadence


def trial_reports():
    common = dict(request_sha256='request', steps=10, warmup_steps=2,
        allocation={'gpu_uuid': 'gpu', 'cpu_threads': 2}, training={'optimizer': 'SGD'},
        initial_state={'encoder': 'first', 'encoder_parameters': 'first', 'head': 'first', 'optimizer': 'momentum', 'torch_rng': 'rng'},
        final_state={'encoder': 'last', 'encoder_parameters': 'last', 'head': 'last', 'optimizer': 'changed', 'torch_rng': 'rng2'},
        warmup_input_sha256='warmup', timed_input_sha256='inputs', snapshot_state_sha256='first')
    off = {**deepcopy(common), 'mode': 'off', 'timing': {'wall_seconds': 100, 'evaluation_completion_seconds': None}}
    on = {**deepcopy(common), 'mode': 'on',
          'timing': {'wall_seconds': 105, 'evaluation_completion_seconds': 80, 'result_before_next_step_interval': True},
          'evaluation': {'status': 'complete', 'retained_temporary_bytes': 0}}
    return off, on


def test_actual_overhead_gate_and_result_deadline_are_separate(harness):
    off, on = trial_reports()
    assert harness['paired_report'](off, on)['passed']
    on['timing']['wall_seconds'] = 105.01
    report = harness['paired_report'](off, on)
    assert not report['overhead_passed'] and report['result_before_next_step_interval']
    on['timing'].update(wall_seconds=103, result_before_next_step_interval=False)
    report = harness['paired_report'](off, on)
    assert report['overhead_passed'] and not report['passed']


@pytest.mark.parametrize('key', ['initial_state', 'final_state', 'allocation', 'training', 'timed_input_sha256', 'steps'])
def test_mismatched_training_comparisons_are_rejected(harness, key):
    off, on = trial_reports()
    on[key] = 'changed'
    with pytest.raises(AssertionError, match='Unmatched training'):
        harness['paired_report'](off, on)


def test_unchanged_encoder_does_not_establish_training_overhead(harness):
    off, on = trial_reports()
    for row in [off, on]:
        row['final_state']['encoder'] = row['initial_state']['encoder']
    with pytest.raises(AssertionError, match='No encoder update'):
        harness['paired_report'](off, on)
    off, on = trial_reports()
    for row in [off, on]:
        row['final_state']['encoder_parameters'] = row['initial_state']['encoder_parameters']
    with pytest.raises(AssertionError, match='No encoder parameter update'):
        harness['paired_report'](off, on)


@pytest.mark.parametrize('damage', [None, 'profile', 'index-hash', 'statistics', 'path', 'partial'])
def test_mixed_evaluation_checks_each_committed_task(harness, tmp_path, damage):
    from bench.storage import file_digest, read_json, write_json

    variants = [{'id': 'global', 'kind': 'classification'}, {'id': 'dense', 'kind': 'segmentation'}]
    profiles = {'global': 'fp32-v1', 'dense': 'bf16-v1'}
    request = {'variants': variants, 'task_numerics': profiles}
    index = {'kind': 'two-task-monitoring-caller', 'monitoring_caller_schema_version': '1.0.0',
        'status': 'complete', 'requested_names': ['global', 'dense'], 'task_numerics': profiles,
        'tasks': {}, 'metrics': {}, 'retained_temporary_bytes': 0}
    for name, profile in profiles.items():
        destination = tmp_path / 'tasks' / name
        statistics = destination / 'statistics.json'
        write_json(statistics, {'task': name, 'evaluation': 'development', 'records': []})
        index['metrics'][name] = {'score': 20.0}
        result = destination / 'results.json'
        write_json(result, {'schema_version': '1.0.0', 'status': 'complete', 'evaluation': 'development',
            'numerics': profile, 'metrics': {name: index['metrics'][name]},
            'tasks': {name: {'status': 'complete', 'statistics': {'file': statistics.name, 'sha256': file_digest(statistics)}}},
            'storage': {'retained_temporary_bytes': 0, 'retained_workspace_bytes': 0}})
        index['tasks'][name] = {'status': 'complete', 'numerics': profile,
            'results_file': str(result.relative_to(tmp_path)), 'results_sha256': file_digest(result)}
    if damage == 'profile':
        result = tmp_path / index['tasks']['global']['results_file']
        altered = read_json(result)
        altered['numerics'] = 'bf16-v1'
        write_json(result, altered)
        index['tasks']['global']['results_sha256'] = file_digest(result)
    elif damage == 'index-hash':
        index['tasks']['global']['results_sha256'] = 'changed'
    elif damage == 'statistics':
        write_json(tmp_path / 'tasks/dense/statistics.json', {'records': ['changed']})
    elif damage == 'path':
        index['tasks']['global']['results_file'] = '../outside/results.json'
    elif damage == 'partial':
        index['status'] = 'running'
    write_json(tmp_path / 'monitoring.json', index)
    if damage is None:
        artifact, results = harness['evaluation_results'](request, tmp_path)
        assert artifact == tmp_path / 'monitoring.json'
        assert {name: result['numerics'] for name, result in results.items()} == profiles
    else:
        with pytest.raises((AssertionError, ValueError)):
            harness['evaluation_results'](request, tmp_path)


def test_real_training_snapshot_evaluation_verification_and_pruned_replay(harness, clevr_data, tmp_path, monkeypatch):
    """CPU fixture runs real optimizer steps and bench calls; only CUDA/process boundaries are replaced."""
    import bench
    from bench.features import model_identity, numerical_mode
    from bench.prepare import load_prepared, prepare
    from bench.protocols import resolve_names
    from bench.storage import file_digest, read_json, write_json
    from test_features import GlobalAdapter

    root, _ = clevr_data
    prepare(names=['clevr-count', 'clevr-distance'], data_root=root)
    tasks = resolve_names(['clevr-count', 'clevr-distance'])
    prepared, rows = load_prepared(root, tasks[0], 'train')
    training_task = deepcopy(tasks[0])
    training_task['classes'] = 100
    def construct(_):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(0)
            return GlobalAdapter()

    model = construct(None)
    path = tmp_path / 'request.json'
    write_json(path, {})
    request = dict(output=str(tmp_path / 'output'), expected_model=model_identity(model, bench.Execution(device='cpu', numerics='fp32-v1')),
        steps=3, warmup_steps=2, cpu_threads=1, numerics='fp32-v1', seed=4, input_seed=5,
        feature_width=5, optimizer={'lr': 0.01, 'momentum': 0.9, 'foreach': False},
        training_batch_size=2, training_workers=0, training_data_root=str(root),
        training_preparation_digest=prepared['digest'], gpu_uuid='cpu-fixture', variants=tasks)
    namespace = harness['train'].__globals__
    monkeypatch.setitem(namespace, 'checked_request', lambda _: request)
    monkeypatch.setitem(namespace, 'construct', construct)
    monkeypatch.setitem(namespace, 'resolve_names', lambda _: [training_task])
    monkeypatch.setitem(namespace, 'load_prepared', lambda *args: (prepared, rows))
    monkeypatch.setitem(namespace, 'numerical_mode', lambda _: numerical_mode(bench.Execution(device='cpu', numerics='fp32-v1')))
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self, **kwargs: self)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self, **kwargs: self)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda: 'CPU fixture')
    original_tensor = torch.tensor
    monkeypatch.setattr(torch, 'tensor', lambda *args, **kwargs:
                        original_tensor(*args, **{**kwargs, 'device': 'cpu'}) if kwargs.get('device') == 'cuda' else original_tensor(*args, **kwargs))
    original_loader = namespace['DataLoader']
    monkeypatch.setitem(namespace, 'DataLoader', lambda *args, **kwargs: original_loader(*args, **{**kwargs, 'pin_memory': False}))

    class Evaluator:
        returncode = 0
        def __init__(self, *args, **kwargs):
            directory = Path(request['output']) / 'on'
            envelope = read_json(directory / 'evaluation-input.json')
            # A separate real process has separate RNG state. Reproduce that boundary here.
            python_state, numpy_state = random.getstate(), np.random.get_state()
            try:
                with torch.random.fork_rng(devices=[]):
                    model = construct(request)
                    model.load_state_dict(torch.load(directory / 'checkpoint.pt', weights_only=True))
                    model.info.recipe['runtime_training_checkpoint_sha256'] = envelope['checkpoint_sha256']
                    expected = model_identity(model, bench.Execution(device='cpu', numerics='fp32-v1'))
                    bench.run(model, names=['clevr-count', 'clevr-distance'], data_root=root,
                        output_dir=directory / 'evaluation', execution=bench.Execution(device='cpu', numerics='fp32-v1', workers=0))
                    write_json(directory / 'evaluation-caller.json', {'request_sha256': file_digest(path),
                        'envelope_sha256': file_digest(directory / 'evaluation-input.json'), 'model': expected, 'encoder_unchanged': True})
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)
        def poll(self):
            return 0
        def wait(self):
            return 0

    monkeypatch.setitem(namespace, 'subprocess', SimpleNamespace(Popen=Evaluator))
    harness['train'](path, 'off')
    harness['train'](path, 'on')
    report = harness['verify'](path)
    assert report['matched_initial_and_final_training_state']
    assert report['snapshot_checks']['off']['encoder_state_sha256'] == report['snapshot_checks']['on']['encoder_state_sha256']
    output = Path(request['output'])
    original_proof = (output / 'verification.json').read_bytes()
    for mode in ['off', 'on']:
        (output / mode / 'checkpoint.pt').unlink()
    write_json(output / 'cleanup.json', {'verification_sha256': file_digest(output / 'verification.json'),
        'removed': ['off/checkpoint.pt', 'on/checkpoint.pt'], 'retained_generated_checkpoint_bytes': 0})
    assert harness['verify'](path) == report
    assert (output / 'verification.json').read_bytes() == original_proof
    changed = read_json(output / 'on/training.json')
    changed['timed_input_sha256'] = 'altered'
    write_json(output / 'on/training.json', changed)
    with pytest.raises(AssertionError, match='Unmatched training'):
        harness['verify'](path)
