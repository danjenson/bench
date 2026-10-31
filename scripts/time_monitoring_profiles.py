"""Time ordinary per-task calls with one fixed numerical profile per task.

Classification uses FP32 for every checkpoint; segmentation uses BF16. This
caller keeps their ordinary bench artifacts separate and records their combined
latency without introducing a new numerical profile or aggregate model score.
"""
from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path
import resource
import runpy
import sys
import types

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import bench
from bench.features import model_identity
from bench.prepare import load_prepared
from bench.probes import label_canvas
from bench.protocols import resolve_names
from bench.runner import read_result
from bench.storage import contained_path, file_digest, read_json, write_json


def run_tasks(model, *, variants, task_numerics, data_root, output, batch_size, workers,
              expected_models, run_task=bench.run):
    """Keep completed ordinary results readable when a later task fails."""
    output = Path(output)
    assert not output.exists(), 'Preserve any original pair measurement'
    assert len(variants) == 2 and [v['kind'] for v in variants] == ['classification', 'segmentation']
    expected_profiles = {v['id']: 'fp32-v1' if v['kind'] == 'classification' else 'bf16-v1' for v in variants}
    assert task_numerics == expected_profiles
    index = {'monitoring_caller_schema_version': '1.0.0', 'kind': 'two-task-monitoring-caller',
             'status': 'running', 'requested_names': [v['id'] for v in variants],
             'task_numerics': task_numerics, 'metrics': {}, 'tasks': {}}
    path = output / 'monitoring.json'
    write_json(path, index)
    for variant in variants:
        name = variant['id']
        profile = task_numerics[name]
        destination = output / 'tasks' / name
        try:
            result = run_task(model, names=[name], evaluation='development', data_root=data_root,
                output_dir=destination, execution=bench.Execution(device='cuda', numerics=profile,
                                                                 batch_size=batch_size, workers=workers))
            assert result.status == 'complete' and set(result.metrics) == {name}
            saved = read_result(destination)
            assert saved['model'] == expected_models[profile] and saved['numerics'] == profile
            assert saved['evaluation'] == 'development' and set(saved['tasks']) == {name}
            assert saved['storage']['retained_temporary_bytes'] == saved['storage']['retained_workspace_bytes'] == 0
            assert not (destination / '.work').exists()
            index['metrics'][name] = result.metrics[name]
            index['tasks'][name] = {'status': 'complete', 'numerics': profile,
                'results_file': str((destination / 'results.json').relative_to(output)),
                'results_sha256': file_digest(destination / 'results.json')}
            write_json(path, index)
        except BaseException as error:
            index['status'] = 'failed'
            index['tasks'][name] = {'status': 'failed', 'numerics': profile, 'error': repr(error)}
            write_json(path, index)
            raise
    index['status'] = 'complete'
    index['retained_temporary_bytes'] = 0
    write_json(path, index)
    return index


def checked_request(path):
    request = read_json(path)
    assert all(file_digest(p) == h for p, h in request['pins'].items())
    assert Path(bench.__file__).resolve().parent == Path(request['package_root']).resolve()
    tasks = resolve_names(request['names'])
    assert tasks == request['variants'] and len(tasks) == 2
    assert [t['kind'] for t in tasks] == ['classification', 'segmentation']
    assert request['partial_result_file'] == 'monitoring.json'
    assert request['batch_size'] == 32 and request['cpu_threads'] == 2
    assert request['task_numerics'] == {t['id']: 'fp32-v1' if t['kind'] == 'classification' else 'bf16-v1' for t in tasks}
    assert set(request['expected_models']) == {'fp32-v1', 'bf16-v1'}
    stripped = [{k: v for k, v in request['expected_models'][n].items() if k not in ['numerics', 'digest']}
                for n in ['fp32-v1', 'bf16-v1']]
    assert stripped[0] == stripped[1]
    return request, tasks


def run(path):
    request, tasks = checked_request(path)
    torch.set_num_threads(request['cpu_threads'])
    if request.get('open_clip_source'):
        namespace = types.ModuleType('open_clip')
        namespace.__path__ = [str(Path(request['open_clip_source']) / 'src/open_clip')]
        sys.modules['open_clip'] = namespace
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = instantiate(OmegaConf.load(request['model_yaml']), _convert_='all')
    for profile, expected in request['expected_models'].items():
        assert model_identity(model, bench.Execution(device='cpu', numerics=profile)) == expected
    model.cuda()
    torch.cuda.reset_peak_memory_stats()
    run_tasks(model, variants=tasks, task_numerics=request['task_numerics'], data_root=request['data_root'],
        output=request['output'], batch_size=request['batch_size'], workers=request['workers'],
        expected_models=request['expected_models'])
    torch.cuda.synchronize()
    for profile, expected in request['expected_models'].items():
        assert model_identity(model, bench.Execution(device='cuda', numerics=profile)) == expected
    write_json(Path(request['output']) / 'caller.json', {'request_sha256': file_digest(path),
        'encoder_state_unchanged': True, 'gpu_uuid': request['gpu_uuid'], 'cpu_threads': torch.get_num_threads(),
        'peak_gpu_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_host_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        'scope': 'One fresh model load and two ordinary bench.run calls, FP32 classification then BF16 segmentation, including both cleanups. Per-task results remain ordinary separate benchmark artifacts; the caller index has its own schema and no aggregate score.'})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def verify(path):
    request, tasks = checked_request(path)
    output = Path(request['output'])
    index, caller, timing = [read_json(output / name) for name in ['monitoring.json', 'caller.json', 'timing.json']]
    assert index['monitoring_caller_schema_version'] == '1.0.0' and index['kind'] == 'two-task-monitoring-caller'
    assert index['status'] == 'complete' and index['requested_names'] == request['names']
    assert index['task_numerics'] == request['task_numerics'] and index['retained_temporary_bytes'] == 0
    assert set(index['tasks']) == set(index['metrics']) == set(request['names'])
    assert caller['request_sha256'] == timing['request_sha256'] == file_digest(path)
    assert caller['encoder_state_unchanged'] and caller['gpu_uuid'] == request['gpu_uuid']
    assert caller['cpu_threads'] == request['cpu_threads']
    reconstruct = runpy.run_path(request['catalogue_script'])['reconstruct']
    dense_score = runpy.run_path(request['calibration_script'])['dense_score']
    load_case = runpy.run_path(request['comparison_script'])['load_case']
    aliases = read_json(request['alias_receipt'])['tasks']
    proof = {'status': 'verified', 'request_sha256': file_digest(path), 'cases': {}, 'score_parity': {},
        'results_sha256': file_digest(output / 'monitoring.json'), 'results_file': 'monitoring.json',
        'timing_sha256': file_digest(output / 'timing.json'), 'verifier_sha256': file_digest(__file__),
        'task_numerics': request['task_numerics']}
    for task in tasks:
        name, profile = task['id'], request['task_numerics'][task['id']]
        entry = index['tasks'][name]
        assert entry['status'] == 'complete' and entry['numerics'] == profile
        result_path = contained_path(output, entry['results_file'])
        assert file_digest(result_path) == entry['results_sha256']
        destination = result_path.parent
        result = read_result(destination)
        assert result['model'] == request['expected_models'][profile] and result['numerics'] == profile
        assert result['status'] == 'complete' and result['evaluation'] == 'development'
        assert result['metrics'] == {name: index['metrics'][name]}
        assert set(result['tasks']) == {name}
        assert result['storage']['retained_temporary_bytes'] == result['storage']['retained_workspace_bytes'] == 0
        assert not (destination / '.work').exists()
        prepared, training = load_prepared(Path(request['data_root']), task, 'train', verify=False)
        _, rows = load_prepared(Path(request['data_root']), task, 'development', verify=False)
        measured = result['tasks'][name]
        assert measured['task_digest'] == task['digest'] and measured['preparation_digest'] == prepared['digest']
        assert prepared['digest'] == aliases[name]['alias_preparation_digest']
        assert measured['data_digest'] == prepared['splits']['development']['digest']
        statistics = contained_path(destination, measured['statistics']['file'])
        saved = read_json(statistics)
        assert saved['task'] == name and saved['evaluation'] == 'development'
        check = reconstruct(task, measured, saved['records'], rows)
        if task['kind'] == 'segmentation':
            targets = [label_canvas(Path(prepared['source_root']), row, task)[0] for row in rows]
            assert math.isclose(dense_score(saved['records'], targets, task['classes']), check['metrics'][task['metric']], abs_tol=1e-9)
        selected = measured['selection']['seeds']
        assert len(selected) == len(measured['selection']['candidates']) == 1
        assert selected[0]['updates'] == task['probe']['updates'] and selected[0]['seed'] == 0
        assert selected[0]['learning_rate'] == task['probe']['learning_rates'][0]
        assert selected[0]['weight_decay'] == task['probe']['weight_decays'][0]
        prior = request['parity_priors'][name]
        old = load_case(prior['request'], name)
        assert file_digest(prior['request']) == prior['request_sha256']
        assert old['variant'] == task and old['model'] == result['model']
        assert old['software'] == result['software']['source_files']
        assert old['execution'] == {'device': 'cuda', 'batch_size': request['batch_size'], 'workers': request['workers']}
        assert old['sample_digests'] == prepared['splits']
        assert old['preparation_digest'] == aliases[name]['base_preparation_digest']
        difference = measured['metrics'][task['metric']]['value'] - old['score']
        tolerance = task['numerical_tolerance']['absolute_percentage_points']
        proof['cases'][name] = {**check, 'statistics_sha256': file_digest(statistics), 'results_sha256': file_digest(result_path)}
        proof['score_parity'][name] = {'difference': difference, 'tolerance': tolerance,
            'passed': abs(difference) <= tolerance + 1e-12, 'prior': old['provenance'], 'numerics': profile,
            'scope': 'Fresh ordinary single-task call versus the identical calibration profile, source, recipe, bank and extraction batch32.'}
    assert 0 < timing['wall_seconds'] and 0 <= timing['time_to_first_result_seconds'] <= timing['wall_seconds']
    assert timing['latency_limit_seconds'] == request['latency_limit_seconds'] == 120
    assert timing['passed'] == (timing['wall_seconds'] <= 120)
    proof['verified_records'] = sum(c['records'] for c in proof['cases'].values())
    proof['scope'] = 'Combined caller latency and independently reconstructed ordinary per-task artifacts, FP32 classification / BF16 segmentation for every checkpoint. Same-profile repeatability is separate from registered FP32/BF16 qualification. No reserved acceptance, training-overhead or v1 readiness claim.'
    write_json(output / 'verification.json', proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['run', 'verify'])
    parser.add_argument('request', type=Path)
    args = parser.parse_args()
    if args.action == 'run':
        run(args.request)
    else:
        assert not torch.cuda.is_initialized()
        verify(args.request)
        assert not torch.cuda.is_initialized()
