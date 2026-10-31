"""Matched short training intervals with a separate frozen evaluator on one GPU.

This acceptance caller is outside the bench runner. It fine-tunes a public
encoder plus a fresh classification head on prepared ImageNet-100 training
images. It is not a reference-model training recipe or an eligibility claim.
Only a pinned request can launch real training; this module performs no work
when imported. Choose the monitoring recipe before staging such a request.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys
import time
import types

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

import bench
from bench import Execution, ImageBatch, ModelInfo
from bench.features import ImageDataset, collate_images, model_identity, numerical_mode, save_tensors
from bench.prepare import load_prepared
from bench.probes import label_canvas
from bench.protocols import resolve_names
from bench.runner import read_result
from bench.storage import contained_path, digest, file_digest, read_json, write_json


def fingerprint(value):
    """Hash actual tensor contents and structured optimizer/RNG state."""
    state = hashlib.sha256()

    def update(item):
        if isinstance(item, torch.Tensor):
            state.update(digest(['tensor', str(item.dtype), list(item.shape)]).encode())
            state.update(item.detach().reshape(-1).contiguous().view(torch.uint8).cpu().numpy().tobytes())
        elif isinstance(item, np.ndarray):
            state.update(digest(['array', str(item.dtype), list(item.shape)]).encode())
            state.update(item.tobytes())
        elif isinstance(item, dict):
            state.update(b'dict')
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                update(key)
                update(item[key])
        elif isinstance(item, (tuple, list)):
            state.update(type(item).__name__.encode() + str(len(item)).encode())
            for child in item:
                update(child)
        else:
            state.update(digest([type(item).__name__, item]).encode())
    update(value)
    return state.hexdigest()


def training_state(model, head, optimizer, scheduler, generator, *, cuda):
    return {name: fingerprint(value) for name, value in {
        'encoder': model.state_dict(), 'encoder_parameters': dict(model.named_parameters()), 'head': head.state_dict(),
        'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
        'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
        'cuda_rng': torch.cuda.get_rng_state_all() if cuda else [],
        'input_generator': generator.get_state(),
    }.items()}


def timed_interval(*, steps, synchronize, snapshot, launch, train_step, collect, clock=time.monotonic):
    """Include snapshot, launch, fixed steps, collection and all evaluator waits."""
    assert type(steps) is int and steps > 0
    synchronize()
    started = clock()
    snapshot()
    snapshot_finished = clock()
    process = launch()
    launched = clock() if process is not None else None
    completed = None
    for index in range(steps):
        train_step(index)
        if process is not None and completed is None and process.poll() is not None:
            assert process.returncode == 0, 'Evaluator failed; preserve the original trial'
            completed = clock()
    synchronize()
    training_finished = clock()
    if process is not None and completed is None:
        assert process.wait() == 0, 'Evaluator failed; preserve the original trial'
        completed = clock()
    collect()
    finished = clock()
    return {'wall_seconds': finished - started,
            'training_steps_wall_seconds': training_finished - started,
            'snapshot_seconds': snapshot_finished - started,
            'evaluation_launch_seconds': None if launched is None else launched - started,
            'evaluation_completion_seconds': None if completed is None else completed - started,
            'result_before_next_step_interval': None if completed is None else completed <= training_finished,
            'scope': 'Matched fixed optimizer-step interval, including the same immutable checkpoint save in both trials; enabled time includes evaluator launch, overlap, collection and any remaining wait.'}


def paired_report(off, on):
    for key in ['request_sha256', 'steps', 'warmup_steps', 'allocation', 'training',
                'initial_state', 'final_state', 'warmup_input_sha256', 'timed_input_sha256',
                'snapshot_state_sha256']:
        assert off[key] == on[key], f'Unmatched training trials: {key}'
    assert off['mode'] == 'off' and on['mode'] == 'on'
    assert off['initial_state']['encoder'] != off['final_state']['encoder'], 'No encoder update was measured'
    assert off['initial_state']['encoder_parameters'] != off['final_state']['encoder_parameters'], 'No encoder parameter update was measured'
    assert off['initial_state']['head'] != off['final_state']['head'], 'No training head update was measured'
    assert off['timing']['evaluation_completion_seconds'] is None
    assert on['timing']['evaluation_completion_seconds'] is not None
    assert on['evaluation']['status'] == 'complete' and on['evaluation']['retained_temporary_bytes'] == 0
    a, b = [row['timing']['wall_seconds'] for row in [off, on]]
    assert all(math.isfinite(value) and value > 0 for value in [a, b])
    overhead = 100 * (b - a) / a
    cadence = on['timing']['result_before_next_step_interval'] is True
    return {'off_seconds': a, 'on_seconds': b, 'overhead_percent': overhead,
            'overhead_limit_percent': 5, 'overhead_passed': overhead <= 5,
            'result_before_next_step_interval': cadence, 'passed': overhead <= 5 and cadence,
            'steps': off['steps'], 'matched_initial_and_final_training_state': True,
            'scope': 'Measured training overhead for this exact fixed-step cadence, allocation, caller and evaluation recipe. No estimated-time denominator, other-GPU claim, reference-model eligibility, or universal cadence claim.'}


def checked_request(path):
    request = read_json(path)
    assert all(file_digest(p) == expected for p, expected in request['pins'].items()), 'Pinned input changed'
    assert request['pins'][str(Path(__file__).resolve())] == file_digest(__file__)
    assert Path(bench.__file__).resolve().parent == Path(request['package_root']).resolve()
    assert Path(sys.executable).resolve() == Path(request['python_binary']).resolve()
    assert type(request['steps']) is int and request['steps'] > 0
    assert type(request['warmup_steps']) is int and request['warmup_steps'] > 0
    assert request['numerics'] in ['bf16-v1', 'fp32-v1']
    assert request['names'] and len(request['names']) == 2
    assert {v['kind'] for v in request['variants']} == {'classification', 'segmentation'}
    assert resolve_names(request['names']) == request['variants']
    if 'task_numerics' in request:
        assert [v['kind'] for v in request['variants']] == ['classification', 'segmentation']
        assert request['task_numerics'] == {v['id']: 'fp32-v1' if v['kind'] == 'classification' else 'bf16-v1'
                                            for v in request['variants']}
        for key in ['profile_caller_script', 'catalogue_script', 'calibration_script']:
            assert request['pins'][request[key]] == file_digest(request[key])
    return request


def construct(request):
    sys.path.insert(0, request['adapter_directory'])
    if request.get('open_clip_source'):
        namespace = types.ModuleType('open_clip')
        namespace.__path__ = [str(Path(request['open_clip_source']) / 'src/open_clip')]
        sys.modules['open_clip'] = namespace
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        return instantiate(OmegaConf.load(request['model_yaml']), _convert_='all')


def next_batches(loader):
    while True:
        yield from loader


def evaluation_results(request, output):
    """Read committed ordinary results without treating a caller index as one."""
    output = Path(output)
    if 'task_numerics' not in request:
        result = read_result(output)
        assert result['status'] == 'complete'
        assert result['storage']['retained_temporary_bytes'] == result['storage']['retained_workspace_bytes'] == 0
        return output / 'results.json', {v['id']: result for v in request['variants']}
    index_path = output / 'monitoring.json'
    index = read_json(index_path)
    expected = {v['id'] for v in request['variants']}
    assert index['kind'] == 'two-task-monitoring-caller' and index['monitoring_caller_schema_version'] == '1.0.0'
    assert index['status'] == 'complete' and index['task_numerics'] == request['task_numerics']
    assert index['requested_names'] == [v['id'] for v in request['variants']]
    assert set(index['tasks']) == set(index['metrics']) == expected
    assert index['retained_temporary_bytes'] == 0
    results = {}
    for name, entry in index['tasks'].items():
        assert entry['status'] == 'complete' and entry['numerics'] == request['task_numerics'][name]
        path = contained_path(output, entry['results_file'])
        assert path.name == 'results.json'
        assert file_digest(path) == entry['results_sha256']
        result = read_result(path.parent)
        assert result['status'] == 'complete' and result['evaluation'] == 'development'
        assert result['numerics'] == request['task_numerics'][name] and set(result['tasks']) == {name}
        assert result['metrics'] == {name: index['metrics'][name]}
        assert result['storage']['retained_temporary_bytes'] == result['storage']['retained_workspace_bytes'] == 0
        assert not (path.parent / '.work').exists()
        results[name] = result
    return index_path, results


def train(path, mode):
    request = checked_request(path)
    directory = Path(request['output']) / mode
    assert not directory.exists(), 'Inspect any original training interval instead of repeating it'
    directory.mkdir(parents=True)
    torch.set_num_threads(request['cpu_threads'])
    random.seed(request['seed'])
    np.random.seed(request['seed'])
    torch.manual_seed(request['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    model = construct(request)
    assert model_identity(model, Execution(device='cpu', numerics=request['numerics'])) == request['expected_model']
    model.cuda().train()
    head = nn.Linear(request['feature_width'], 100).cuda().train()
    optimizer = torch.optim.SGD([*model.parameters(), *head.parameters()], **request['optimizer'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=request['steps'] + request['warmup_steps'])
    task = resolve_names(['imagenet-100'])[0]
    prepared, rows = load_prepared(Path(request['training_data_root']), task, 'train')
    assert prepared['digest'] == request['training_preparation_digest'] and task['classes'] == 100
    labels = {row['id']: row['target'] for row in rows}
    generator = torch.Generator().manual_seed(request['input_seed'])
    loader = DataLoader(ImageDataset(Path(prepared['source_root']), rows, task),
        batch_size=request['training_batch_size'], shuffle=True, generator=generator,
        num_workers=request['training_workers'], collate_fn=collate_images,
        persistent_workers=request['training_workers'] > 0, pin_memory=True)
    batches = next_batches(loader)
    last_loss = None
    warmup_inputs, timed_inputs = hashlib.sha256(), hashlib.sha256()

    def step(hasher):
        nonlocal last_loss
        images, valid, affine, ids = next(batches)
        hasher.update(digest(list(ids)).encode())
        batch = ImageBatch(images.cuda(non_blocking=True), valid.cuda(non_blocking=True),
                           affine.cuda(non_blocking=True), ids)
        targets = torch.tensor([labels[sample] for sample in ids], device='cuda')
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=request['numerics'] == 'bf16-v1'):
            values = model.encode_global(batch).features
            last_loss = torch.nn.functional.cross_entropy(head(values).float(), targets)
        last_loss.backward()
        optimizer.step()
        scheduler.step()

    checkpoint = directory / 'checkpoint.pt'
    envelope = directory / 'evaluation-input.json'
    snapshot_state = {}
    evaluation = {}
    with numerical_mode(Execution(device='cuda', numerics=request['numerics'])):
        for _ in range(request['warmup_steps']):
            step(warmup_inputs)
        torch.cuda.synchronize()
        initial = training_state(model, head, optimizer, scheduler, generator, cuda=True)

        def snapshot():
            state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            snapshot_state['sha256'] = fingerprint(state)
            assert snapshot_state['sha256'] == initial['encoder']
            save_tensors(checkpoint, state)
            write_json(envelope, {'request_sha256': file_digest(path), 'checkpoint_sha256': file_digest(checkpoint),
                                  'encoder_state_sha256': snapshot_state['sha256']})

        def launch():
            if mode == 'off':
                return None
            return subprocess.Popen([sys.executable, '-u', __file__, 'evaluate', str(path)],
                                    env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})

        def collect():
            if mode == 'on':
                artifact, _ = evaluation_results(request, directory / 'evaluation')
                evaluation.update(status='complete', retained_temporary_bytes=0,
                                  results_file=artifact.name, results_sha256=file_digest(artifact))

        timing = timed_interval(steps=request['steps'], synchronize=torch.cuda.synchronize,
            snapshot=snapshot, launch=launch, train_step=lambda _: step(timed_inputs), collect=collect)
        final = training_state(model, head, optimizer, scheduler, generator, cuda=True)
    assert math.isfinite(float(last_loss.detach()))
    report = {'mode': mode, 'request_sha256': file_digest(path), 'steps': request['steps'],
        'warmup_steps': request['warmup_steps'], 'initial_state': initial, 'final_state': final,
        'snapshot_state_sha256': snapshot_state['sha256'], 'checkpoint_sha256': file_digest(checkpoint),
        'warmup_input_sha256': warmup_inputs.hexdigest(), 'timed_input_sha256': timed_inputs.hexdigest(),
        'timing': timing, 'evaluation': evaluation, 'last_training_loss': float(last_loss.detach()),
        'allocation': {'gpu_uuid': request['gpu_uuid'], 'gpu': torch.cuda.get_device_name(),
                       'cpu_affinity': sorted(os.sched_getaffinity(0)), 'cpu_threads': torch.get_num_threads()},
        'training': {key: request[key] for key in ['training_batch_size', 'training_workers', 'optimizer',
            'numerics', 'seed', 'input_seed', 'training_preparation_digest']},
        'scope': 'Short supervised transfer-training interval; encoder and separate training head update. Fixed center crops of prepared training images; no final examples, new public reference or benchmark recipe selection.'}
    write_json(directory / 'training.json', report)
    print(mode, 'training interval seconds:', timing['wall_seconds'], flush=True)


def evaluate(path):
    request = checked_request(path)
    torch.set_num_threads(request['cpu_threads'])
    directory = Path(request['output']) / 'on'
    checkpoint = directory / 'checkpoint.pt'
    envelope = read_json(directory / 'evaluation-input.json')
    assert envelope['request_sha256'] == file_digest(path) and envelope['checkpoint_sha256'] == file_digest(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    assert fingerprint(state) == envelope['encoder_state_sha256']
    model = construct(request)
    model.load_state_dict(state, strict=True)
    del state
    info = model.info
    provenance = dict(info.recipe)
    provenance['initial_public_checkpoint_sha256'] = provenance.pop('checkpoint_sha256', provenance.pop('sha256', None))
    provenance['runtime_initial_checkpoint_metadata'] = {key: provenance.pop(key)
        for key in ['checkpoint_epoch', 'checkpoint_index', 'training_epochs_from_batchnorm_counters'] if key in provenance}
    model.info = ModelInfo(info.name, info.revision, info.readouts, {**provenance,
        'checkpoint_sha256': envelope['checkpoint_sha256'],
        'runtime_training_checkpoint_sha256': envelope['checkpoint_sha256'],
        'runtime_training_scope': 'Private short fine-tuning interval for measured overhead only; not a public reference checkpoint.'})
    model.cuda().eval()
    execution = Execution(device='cuda', numerics=request['numerics'],
                          batch_size=request['evaluation_batch_size'], workers=request['evaluation_workers'])
    before = model_identity(model, execution)
    profiles = {}
    if 'task_numerics' in request:
        profiles = {profile: model_identity(model, Execution(device='cuda', numerics=profile))
                    for profile in set(request['task_numerics'].values())}
        run_tasks = runpy.run_path(request['profile_caller_script'])['run_tasks']
        run_tasks(model, variants=request['variants'], task_numerics=request['task_numerics'],
            data_root=request['evaluation_data_root'], output=directory / 'evaluation',
            batch_size=request['evaluation_batch_size'], workers=request['evaluation_workers'], expected_models=profiles)
    else:
        bench.run(model, names=request['names'], evaluation='development', data_root=request['evaluation_data_root'],
                  output_dir=directory / 'evaluation', execution=execution)
    assert model_identity(model, execution) == before, 'Frozen evaluator modified the snapshot'
    write_json(directory / 'evaluation-caller.json', {'request_sha256': file_digest(path),
        'envelope_sha256': file_digest(directory / 'evaluation-input.json'), 'model': before,
        'encoder_unchanged': True, 'models_by_numerics': profiles,
        'scope': 'Separate process sharing the declared GPU with training; no GPU-idle guard inside the overlap. Optional fixed per-task profiles use two ordinary calls with one immutable model snapshot.'})


def verify(path):
    request = checked_request(path)
    output = Path(request['output'])
    off, on = [read_json(output / mode / 'training.json') for mode in ['off', 'on']]
    assert off['request_sha256'] == on['request_sha256'] == file_digest(path)
    assert off['steps'] == on['steps'] == request['steps']
    assert off['warmup_steps'] == on['warmup_steps'] == request['warmup_steps']
    for row in [off, on]:
        assert row['allocation']['gpu_uuid'] == request['gpu_uuid']
        assert row['allocation']['cpu_threads'] == request['cpu_threads']
        assert row['training'] == {key: request[key] for key in row['training']}
    evaluation_artifact, evaluated_tasks = evaluation_results(request, output / 'on/evaluation')
    assert on['evaluation']['results_sha256'] == file_digest(evaluation_artifact)
    assert on['evaluation'].get('results_file', 'results.json') == evaluation_artifact.name
    caller = read_json(output / 'on/evaluation-caller.json')
    assert caller['request_sha256'] == file_digest(path) and caller['encoder_unchanged']
    if 'task_numerics' not in request:
        result = next(iter(evaluated_tasks.values()))
        assert result['evaluation'] == 'development' and result['model'] == caller['model']
        assert result['numerics'] == request['numerics']
        assert set(result['tasks']) == {v['id'] for v in request['variants']}
        assert all(result['tasks'][v['id']]['task_digest'] == v['digest'] for v in request['variants'])
    else:
        models = caller['models_by_numerics']
        assert set(models) == set(request['task_numerics'].values())
        assert len({model['state_digest'] for model in models.values()}) == 1
        stripped = [{k: v for k, v in model.items() if k not in ['digest', 'numerics']} for model in models.values()]
        assert all(value == stripped[0] for value in stripped)
        assert caller['model'] == models[request['numerics']]
        reconstruct = runpy.run_path(request['catalogue_script'])['reconstruct']
        dense_score = runpy.run_path(request['calibration_script'])['dense_score']
        index = read_json(output / 'on/evaluation/monitoring.json')
        source = Path(request['package_root'])
        expected_sources = {str(p.relative_to(source)): file_digest(p) for p in source.rglob('*.py')}
        for task in request['variants']:
            name, profile = task['id'], request['task_numerics'][task['id']]
            result = evaluated_tasks[name]
            assert result['model'] == models[profile] and result['numerics'] == profile
            assert result['software']['source_files'] == expected_sources
            assert result['model']['info']['recipe']['runtime_training_checkpoint_sha256'] == on['checkpoint_sha256']
            measured = result['tasks'][name]
            prepared, rows = load_prepared(Path(request['evaluation_data_root']), task, 'development', verify=False)
            assert measured['task_digest'] == task['digest'] and measured['preparation_digest'] == prepared['digest']
            assert measured['data_digest'] == prepared['splits']['development']['digest']
            task_result = contained_path(output / 'on/evaluation', index['tasks'][name]['results_file'])
            statistics = contained_path(task_result.parent, measured['statistics']['file'])
            saved = read_json(statistics)
            assert saved['task'] == name and saved['evaluation'] == 'development'
            check = reconstruct(task, measured, saved['records'], rows)
            if task['kind'] == 'segmentation':
                targets = [label_canvas(Path(prepared['source_root']), row, task)[0] for row in rows]
                assert math.isclose(dense_score(saved['records'], targets, task['classes']), check['metrics'][task['metric']], rel_tol=0, abs_tol=1e-9)
            selected = measured['selection']['seeds']
            assert len(selected) == len(measured['selection']['candidates']) == 1
            assert selected[0]['seed'] == 0 and selected[0]['updates'] == task['probe']['updates']
            assert selected[0]['learning_rate'] == task['probe']['learning_rates'][0]
            assert selected[0]['weight_decay'] == task['probe']['weight_decays'][0]
    assert caller['envelope_sha256'] == file_digest(output / 'on/evaluation-input.json')
    envelope = read_json(output / 'on/evaluation-input.json')
    assert envelope == {'request_sha256': file_digest(path), 'checkpoint_sha256': on['checkpoint_sha256'],
                        'encoder_state_sha256': on['initial_state']['encoder']}
    if 'task_numerics' not in request:
        assert result['model']['info']['recipe']['runtime_training_checkpoint_sha256'] == on['checkpoint_sha256']
    cleanup = read_json(output / 'cleanup.json') if (output / 'cleanup.json').exists() else None
    previous = read_json(output / 'verification.json') if cleanup is not None else None
    if cleanup is not None:
        assert cleanup['verification_sha256'] == file_digest(output / 'verification.json')
        assert cleanup['removed'] == ['off/checkpoint.pt', 'on/checkpoint.pt']
        assert cleanup['retained_generated_checkpoint_bytes'] == 0
    snapshots = {}
    for mode, report in [('off', off), ('on', on)]:
        checkpoint = output / mode / 'checkpoint.pt'
        snapshot = {'checkpoint_sha256': report['checkpoint_sha256'], 'encoder_state_sha256': report['initial_state']['encoder']}
        if cleanup is None:
            assert report['checkpoint_sha256'] == file_digest(checkpoint)
            assert fingerprint(torch.load(checkpoint, map_location='cpu', weights_only=True)) == report['initial_state']['encoder']
        else:
            assert not checkpoint.exists() and previous['snapshot_checks'][mode] == snapshot
        snapshots[mode] = snapshot
    report = paired_report(off, on)
    report.update(status='verified-measurement', at_utc=datetime.now(timezone.utc).isoformat(),
        request_sha256=file_digest(path), worker_sha256=file_digest(__file__),
        training_receipts={mode: file_digest(output / mode / 'training.json') for mode in ['off', 'on']},
        snapshot_checks=snapshots,
        evaluation_caller_sha256=file_digest(output / 'on/evaluation-caller.json'),
        evaluation_envelope_sha256=file_digest(output / 'on/evaluation-input.json'),
        evaluation_results_sha256=file_digest(evaluation_artifact), evaluation_results_file=evaluation_artifact.name,
        evaluation_task_numerics=request.get('task_numerics'))
    if cleanup is not None:
        assert all(previous[key] == value for key, value in report.items() if key != 'at_utc')
        return previous
    write_json(output / 'verification.json', report)
    return report


def run(path):
    request = checked_request(path)
    assert file_digest(path) == os.environ['BENCH_EXTENSION_REQUEST_SHA256']
    output = Path(request['output'])
    assert not output.exists(), 'Preserve any original measurement'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() == request['gpu_uuid']
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU occupied'
    output.mkdir(parents=True)
    environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': str(request['seed']),
                   'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                   'PYTHONPATH': str(Path(request['package_root']).parent)}
    for mode in ['off', 'on']:
        subprocess.run([sys.executable, '-u', __file__, 'train', str(path), '--mode', mode], env=environment, check=True)
    report = verify(path)
    # Compact evidence survives; the generated adapter snapshots are temporary.
    for mode in ['off', 'on']:
        (output / mode / 'checkpoint.pt').unlink()
    write_json(output / 'cleanup.json', {'verification_sha256': file_digest(output / 'verification.json'),
        'removed': ['off/checkpoint.pt', 'on/checkpoint.pt'], 'retained_generated_checkpoint_bytes': 0,
        'scope': 'Original public checkpoints, prepared data, metrics and provenance are preserved.'})
    print('Measured training overhead:', report['overhead_percent'], 'passed:', report['passed'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'train', 'evaluate', 'verify'])
    parser.add_argument('request', type=Path)
    parser.add_argument('--mode', choices=['off', 'on'])
    args = parser.parse_args()
    if args.action == 'train':
        assert args.mode is not None
        train(args.request, args.mode)
    else:
        globals()[args.action](args.request)
