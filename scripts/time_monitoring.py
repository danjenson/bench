"""Prepare and measure fixed monitoring candidates through the ordinary runner."""

import argparse
from copy import deepcopy
import gc
import math
import os
from pathlib import Path
import runpy
import time
import types
import sys

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import bench
from bench.features import model_identity
from bench.prepare import load_prepared
from bench.probes import label_canvas
from bench.protocols import resolve_names
from bench.runner import read_result
from bench.storage import digest, file_digest, read_json, write_json


def prepare_aliases(base_root, destination, variants):
    """Derive task metadata without reading final rows or copying source data."""
    base_root, destination = Path(base_root).resolve(), Path(destination).resolve()
    assert not destination.exists(), 'Use a fresh preparation destination'
    receipt = {'scope': 'Calibration task aliases; identical source data and split files. Final manifest file is hardlinked without reading it; final images/annotations are not accessed.', 'tasks': {}}
    for variant in variants:
        base = resolve_names([variant['calibration']['base_task']])[0]
        expected = deepcopy(base)
        expected.update(id=variant['id'], name=variant['id'], release_status='calibration-only', calibration=variant['calibration'])
        expected['probe'].update(updates=variant['probe']['updates'], learning_rates=variant['probe']['learning_rates'])
        if 'weight_decay_candidate' in variant['calibration']:
            decay = variant['calibration']['weight_decay_candidate']
            assert type(decay) in (int, float) and math.isfinite(decay) and decay >= 0, 'Invalid declared weight decay'
            assert variant['calibration']['weight_decay_reference'] == base['probe']['weight_decays'], 'Declared weight-decay reference differs from the base recipe'
            expected['probe']['weight_decays'] = [decay]
        expected['digest'] = digest({k: v for k, v in expected.items() if k != 'digest'})
        assert expected == variant and variant['calibration']['base_task_digest'] == base['digest'], 'Alias changed more than the declared probe settings/identity'
        assert variant['id'] != base['id'] and variant['probe']['seeds'] == [0]
        prepared, training = load_prepared(base_root, base, 'train', verify=False)
        _, development = load_prepared(base_root, base, 'development', verify=False)
        source = base_root / '.bench/prepared' / base['id']
        target = destination / '.bench/prepared' / variant['id']
        target.mkdir(parents=True)
        for entry in prepared['splits'].values():
            os.link(source / entry['file'], target / entry['file'])
        inventory = destination / '.bench/sources' / f"{base['source']}.json"
        inventory.parent.mkdir(parents=True, exist_ok=True)
        if not inventory.exists():
            os.link(base_root / '.bench/sources' / inventory.name, inventory)
        alias = deepcopy(prepared)
        alias['task'] = variant
        alias['digest'] = digest({k: v for k, v in alias.items() if k not in ['digest', 'source_root']})
        write_json(target / 'manifest.json', alias)
        assert load_prepared(destination, variant, 'train', verify=False)[1] == training
        assert load_prepared(destination, variant, 'development', verify=False)[1] == development
        receipt['tasks'][variant['id']] = {'base_task': base['id'], 'base_manifest': str(source / 'manifest.json'),
            'base_manifest_sha256': file_digest(source / 'manifest.json'), 'base_preparation_digest': prepared['digest'],
            'alias_manifest_sha256': file_digest(target / 'manifest.json'), 'alias_preparation_digest': alias['digest'],
            'splits': alias['splits'], 'source_root': alias['source_root']}
    return receipt


def checked_request(path):
    request = read_json(path)
    assert all(file_digest(p) == h for p, h in request['pins'].items()), 'Pinned timing input changed'
    assert Path(bench.__file__).resolve().parent == Path(request['package_root']).resolve()
    tasks = resolve_names(request['names'])
    assert tasks == request['variants'] and len(tasks) == 2
    assert [task['kind'] for task in tasks] == ['classification', 'segmentation']
    return request, tasks


def run(path):
    request, tasks = checked_request(path)
    assert not Path(request['output']).exists(), 'Inspect an existing trial instead of refitting'
    torch.set_num_threads(request['cpu_threads'])
    if request.get('open_clip_source'):
        namespace = types.ModuleType('open_clip')
        namespace.__path__ = [str(Path(request['open_clip_source']) / 'src/open_clip')]
        sys.modules['open_clip'] = namespace
    started = time.monotonic()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = instantiate(OmegaConf.load(request['model_yaml']), _convert_='all')
    assert model_identity(model, bench.Execution(device='cpu', numerics=request['numerics'])) == request['expected_model']
    model.cuda()
    execution = bench.Execution(device='cuda', numerics=request['numerics'], batch_size=request['batch_size'], workers=request['workers'])
    assert model_identity(model, execution) == request['expected_model']
    loading = time.monotonic() - started
    print('Fresh fixed-recipe timing:', request['label'], request['numerics'], flush=True)
    result = bench.run(model, names=request['names'], evaluation='development', data_root=request['data_root'],
                       output_dir=request['output'], execution=execution)
    torch.cuda.synchronize()
    assert result.status == 'complete' and model_identity(model, execution) == request['expected_model']
    assert not (Path(request['output']) / '.work').exists()
    write_json(Path(request['output']) / 'caller.json', {'request_sha256': file_digest(path),
        'model_loading_and_identity_seconds': loading, 'encoder_state_unchanged': True,
        'gpu_uuid': request['gpu_uuid'], 'cpu_threads': torch.get_num_threads(),
        'scope': 'Fresh standalone bench.run with fixed calibration aliases. Parent wall clock includes child startup, loading/hash checks, runner and process cleanup. No feature/probe reuse or final examples.'})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def verify(path):
    request, tasks = checked_request(path)
    output = Path(request['output'])
    result, caller, timing = read_result(output), read_json(output / 'caller.json'), read_json(output / 'timing.json')
    assert result['status'] == 'complete' and result['evaluation'] == 'development'
    assert result['model'] == request['expected_model'] and result['numerics'] == request['numerics']
    assert caller['request_sha256'] == timing['request_sha256'] == file_digest(path)
    assert caller['encoder_state_unchanged'] and caller['gpu_uuid'] == request['gpu_uuid']
    assert result['storage']['retained_temporary_bytes'] == result['storage']['retained_workspace_bytes'] == 0
    assert not (output / '.work').exists()
    assert set(result['tasks']) == {task['id'] for task in tasks}
    reconstruct = runpy.run_path(request['catalogue_script'])['reconstruct']
    dense_score = runpy.run_path(request['calibration_script'])['dense_score']
    load_case = runpy.run_path(request['comparison_script'])['load_case']
    aliases = read_json(request['alias_receipt'])['tasks']
    proof = {'status': 'verified', 'request_sha256': file_digest(path), 'cases': {}, 'score_parity': {},
             'results_sha256': file_digest(output / 'results.json'), 'timing_sha256': file_digest(output / 'timing.json'),
             'verifier_sha256': file_digest(__file__)}
    for task in tasks:
        prepared, training = load_prepared(Path(request['data_root']), task, 'train', verify=False)
        _, rows = load_prepared(Path(request['data_root']), task, 'development', verify=False)
        measured = result['tasks'][task['id']]
        assert measured['task_digest'] == task['digest'] and measured['preparation_digest'] == prepared['digest']
        assert prepared['digest'] == aliases[task['id']]['alias_preparation_digest']
        assert measured['data_digest'] == prepared['splits']['development']['digest']
        statistics = output / measured['statistics']['file']
        saved = read_json(statistics)
        assert saved['task'] == task['id'] and saved['evaluation'] == 'development'
        check = reconstruct(task, measured, saved['records'], rows)
        if task['kind'] == 'segmentation':
            targets = [label_canvas(Path(prepared['source_root']), row, task)[0] for row in rows]
            actual = dense_score(saved['records'], targets, task['classes'])
            assert math.isclose(actual, check['metrics'][task['metric']], rel_tol=0, abs_tol=1e-9)
            del targets
        selected = measured['selection']['seeds']
        assert len(selected) == len(measured['selection']['candidates']) == 1
        assert selected[0]['updates'] == task['probe']['updates']
        assert selected[0]['learning_rate'] == task['probe']['learning_rates'][0]
        assert selected[0]['weight_decay'] == task['probe']['weight_decays'][0]
        assert selected[0]['epoch'] == math.ceil(selected[0]['updates'] / math.ceil(len(training) / task['probe']['batch_size']))
        proof['cases'][task['id']] = {**check, 'statistics_sha256': file_digest(statistics)}
        if task['id'] in request.get('parity_priors', {}):
            prior = request['parity_priors'][task['id']]
            old = load_case(prior['request'], task['id'])
            assert old['variant'] == task and old['model'] == result['model']
            assert old['software'] == result['software']['source_files']
            assert old['sample_digests'] == prepared['splits']
            assert old['preparation_digest'] == aliases[task['id']]['base_preparation_digest']
            difference = measured['metrics'][task['metric']]['value'] - old['score']
            tolerance = task['numerical_tolerance']['absolute_percentage_points']
            proof['score_parity'][task['id']] = {'difference': difference, 'tolerance': tolerance,
                'passed': abs(difference) <= tolerance, 'prior': old['provenance'],
                'scope': f"Same profile/recipe and exact source splits; extraction batch {old['execution']['batch_size']} versus {request['batch_size']}, and the new preparation digest records metadata aliases."}
    assert 0 < timing['wall_seconds'] and 0 <= timing['time_to_first_result_seconds'] <= timing['wall_seconds']
    assert timing['latency_limit_seconds'] == request['latency_limit_seconds'] == 120
    assert timing['passed'] == (timing['wall_seconds'] <= request['latency_limit_seconds'])
    proof['verified_records'] = sum(case['records'] for case in proof['cases'].values())
    proof['scope'] = 'Verified fixed-candidate development timing/statistics. Latency/parity failures remain findings. Numerical profile comparison, reserved acceptance, precision/agreement and measured training overhead remain separate.'
    write_json(output / 'verification.json', proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['run', 'verify'])
    parser.add_argument('request', type=Path)
    args = parser.parse_args()
    if args.action == 'run':
        run(args.request)
    else:
        assert not torch.cuda.is_initialized()
        verify(args.request)
        assert not torch.cuda.is_initialized()
