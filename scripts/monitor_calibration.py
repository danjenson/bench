"""Pinned monitoring calibration with independent compact-record reconstruction."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time
import types

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import bench
from bench.features import model_identity
from bench.prepare import load_prepared
from bench.probes import label_canvas
from bench.protocols import resolve_names
from bench.storage import contained_path, digest, file_digest, read_json, write_json


def fuller_task(task):
    reference = deepcopy(task)
    reference['id'] += '-fuller-bank-calibration-v1'
    reference['release_status'] = 'calibration-only'
    reference['bank'].pop('train_per_class' if task['source'] == 'imagenet' else 'train')
    reference['reference_bank'] = {'base_task': task['id'], 'base_task_digest': task['digest'],
                                 'training': 'all-official-training-excluding-monitor-development'}
    reference['digest'] = digest({key: value for key, value in reference.items() if key != 'digest'})
    return reference


def candidate_budgets(kind, stage):
    budgets = (4, 8) if kind == 'classification' else (16, 32)
    if stage == 'candidate-long-budget' or (kind == 'segmentation' and stage in ('candidate-regularized-long', 'candidate-regularized-tail')):
        budgets = tuple(2 * value for value in budgets)
    if stage == 'candidate-regularized-tail':
        assert kind == 'segmentation', 'Missing-budget tails support segmentation only'
        budgets = (budgets[-1],)
    return budgets


def selected_monitor_cases(tasks, recipe):
    assert isinstance(recipe, dict) and set(recipe) == {task['id'] for task in tasks}, 'Declare each requested monitoring task'
    result = {}
    for task in tasks:
        setting = recipe[task['id']]
        assert set(setting) == {'base_updates', 'learning_rate_multipliers', 'weight_decay'}, 'Unexpected monitoring settings'
        updates = setting['base_updates']
        assert type(updates) is int and updates >= task['probe']['updates'] and updates % task['probe']['updates'] == 0
        rates = setting['learning_rate_multipliers']
        assert isinstance(rates, list) and rates and len(set(rates)) == len(rates)
        assert all(type(rate) is int and rate in (1, 3, 10, 30, 100, 300) for rate in rates)
        decay = setting['weight_decay']
        assert type(decay) in (int, float) and math.isfinite(decay) and decay >= 0
        factor = updates // task['probe']['updates']
        result[task['id']] = [(budget, 1.0, rate) for rate in rates for budget in (factor, 2 * factor)]
    return result


def declared_variants(tasks, stage, reference_counts=None, reference_recipe=None, monitor_recipe=None):
    if stage not in ('convergence', 'fast-convergence', 'long-convergence', 'candidate-budget', 'candidate-long-budget', 'candidate-selection', 'candidate-regularized', 'candidate-regularized-long', 'candidate-regularized-tail', 'selected-monitor', 'fuller-selection', 'selected-fuller', 'budget-scale', 'variance-floor'):
        raise ValueError('Declare a supported convergence, budget-scale or variance-floor stage')
    monitor_cases = selected_monitor_cases(tasks, monitor_recipe) if stage == 'selected-monitor' else {}
    if stage != 'selected-monitor':
        assert monitor_recipe is None
    if stage == 'candidate-selection':
        assert all(task['kind'] == 'classification' for task in tasks)
    reference_stage = stage in ('fuller-selection', 'selected-fuller')
    rate_check = stage in ('convergence', 'fast-convergence', 'long-convergence', 'candidate-budget', 'candidate-long-budget', 'candidate-selection', 'candidate-regularized', 'candidate-regularized-long', 'candidate-regularized-tail', 'selected-monitor', 'fuller-selection', 'selected-fuller')
    if stage == 'selected-fuller':
        assert isinstance(reference_recipe, dict) and set(reference_recipe) == {task['id'] for task in tasks}
    else:
        assert reference_recipe is None
    rates = (3, 10, 30, 100) if stage == 'long-convergence' else (100, 300) if stage == 'fast-convergence' else (1, 3, 10, 30)
    budgets = (1, 2) if reference_stage else (8, 16, 32) if stage == 'long-convergence' else (4, 8)
    cases = [(budget, 1.0, rate) for rate in rates for budget in budgets] if rate_check else [
        (1, 1.0, 1), (2, 1.0, 1), (1, 0.1, 1), (1, 10.0, 1)]
    variants = []
    for task in tasks:
        if stage == 'selected-fuller':
            setting = reference_recipe[task['id']]
            unit = 50 if task['kind'] == 'classification' else 40
            assert set(setting) == {'base_epochs', 'learning_rate_multiplier'}
            assert type(setting['base_epochs']) is int and setting['base_epochs'] >= unit and setting['base_epochs'] % unit == 0
            assert type(setting['learning_rate_multiplier']) is int and setting['learning_rate_multiplier'] in (1, 3, 10, 30)
            factor = setting['base_epochs'] // unit
            cases = [(factor, 1.0, setting['learning_rate_multiplier']), (2 * factor, 1.0, setting['learning_rate_multiplier'])]
        if stage in ('candidate-budget', 'candidate-long-budget', 'candidate-selection', 'candidate-regularized', 'candidate-regularized-long', 'candidate-regularized-tail'):
            budgets = candidate_budgets(task['kind'], stage)
            cases = [(budget, 1.0, 10) for budget in budgets]
        if monitor_cases:
            cases = monitor_cases[task['id']]
        for budget, scale, rate in cases:
            variant = fuller_task(task) if reference_stage else deepcopy(task)
            reference_epochs = 50 if task['kind'] == 'classification' else 40
            if reference_stage:
                variant['probe']['updates'] = reference_epochs * math.ceil(reference_counts[task['id']] / task['probe']['batch_size'])
            variant['probe']['updates'] *= budget
            variant['probe']['learning_rates'] = [value * rate for value in task['probe']['learning_rates']]
            variant['probe']['seeds'] = [0]
            variant['id'] = f"{variant['id']}-calibration-u{variant['probe']['updates']}-s{scale:g}"
            if rate_check:
                variant['id'] += f'-lrm{rate}'
            if stage == 'variance-floor':
                variant['standardization']['variance_floor'] = 1e-8
                variant['id'] += '-floor1e-08'
            if reference_stage:
                variant['probe'].update(selection='best-development-score-then-earlier-epoch-then-candidate-order', scoring='each-epoch')
                variant['id'] += '-bestdev'
            if stage == 'candidate-selection':
                variant['probe'].update(selection='best-development-score-then-earlier-epoch-then-candidate-order', scoring='each-epoch')
                variant['id'] += '-bestdev'
            if stage in ('candidate-regularized', 'candidate-regularized-long', 'candidate-regularized-tail', 'selected-monitor'):
                decay = monitor_recipe[task['id']]['weight_decay'] if stage == 'selected-monitor' else 1.0
                variant['probe']['weight_decays'] = [decay]
                variant['id'] += f'-wd{decay:g}'
            variant.update(name=variant['id'], release_status='calibration-only')
            variant['calibration'] = {'base_task': task['id'], 'base_task_digest': task['digest'],
                'feature_scale_before_standardization': scale, 'scale_arithmetic': 'float32',
                'budget_multiplier': budget, 'learning_rate_multiplier': rate,
                'learning_rate_reference': task['probe']['learning_rates']}
            if stage == 'variance-floor':
                variant['calibration']['variance_floor_reference'] = task['standardization']['variance_floor']
            if reference_stage:
                variant['calibration'].update(reference_epochs=reference_epochs * budget,
                                              selection_reference=task['probe']['selection'])
            if stage == 'candidate-selection':
                variant['calibration']['selection_reference'] = task['probe']['selection']
            if stage in ('candidate-regularized', 'candidate-regularized-long', 'candidate-regularized-tail', 'selected-monitor'):
                variant['calibration'].update(weight_decay_reference=task['probe']['weight_decays'], weight_decay_candidate=decay)
            variant['digest'] = digest({key: value for key, value in variant.items() if key != 'digest'})
            variants.append(variant)
    return variants


def checked_request(path):
    request = read_json(path)
    assert all(file_digest(p) == expected for p, expected in request['pins'].items()), 'Pinned input changed'
    assert Path(bench.__file__).resolve().parent == Path(request['package_root']).resolve()
    assert request['names'] and set(request['names']) <= {'imagenet-100', 'pascal'}
    tasks = resolve_names(request['names'])
    counts = {}
    if request['stage'] in ('fuller-selection', 'selected-fuller'):
        for task in tasks:
            reference = fuller_task(task)
            original, base = load_prepared(request['data_root'], task, 'train', verify=False)
            prepared, rows = load_prepared(request['data_root'], reference, 'train', verify=False)
            counts[task['id']] = len(rows)
            assert len(rows) > len(base) and {row['id'] for row in base} <= {row['id'] for row in rows}
            assert original['labels'] == prepared['labels']
            assert all(original['splits'][split] == prepared['splits'][split] for split in ['development', 'final'])
        assert counts == request['reference_counts']
    assert request['variants'] == declared_variants(tasks, request['stage'], counts, request.get('reference_recipe'), request.get('monitor_recipe'))
    return request, tasks


def run(path):
    request, tasks = checked_request(path)
    assert not Path(request['output']).exists(), 'Inspect the existing run instead of refitting'
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    assert gpu == request['gpu_uuid']
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU occupied'
    torch.set_num_threads(request['cpu_threads'])
    if request.get('open_clip_source'):
        namespace = types.ModuleType('open_clip')
        namespace.__path__ = [str(Path(request['open_clip_source']) / 'src/open_clip')]
        sys.modules['open_clip'] = namespace
    started = time.monotonic()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = instantiate(OmegaConf.load(request['model_yaml']), _convert_='all')
    cpu_identity = model_identity(model, bench.Execution(device='cpu', numerics=request['numerics']))
    assert cpu_identity == request['expected_model']
    model.cuda()
    execution = bench.Execution(device='cuda', numerics=request['numerics'],
        workers=request['workers'], batch_size=request['batch_size'])
    assert model_identity(model, execution) == cpu_identity
    loading_seconds = time.monotonic() - started
    measure = runpy.run_path(request['measure_script'])['measure']
    print('Starting', request['label'], request['stage'], request['numerics'], flush=True)
    extra = {'reference_recipe': request['reference_recipe']} if request['stage'] == 'selected-fuller' else {}
    if request['stage'] == 'selected-monitor':
        extra['monitor_recipe'] = request['monitor_recipe']
    report = measure(model, names=request['names'], data_root=request['data_root'],
        output=request['output'], execution=execution, stage=request['stage'], **extra)
    assert model_identity(model, execution) == cpu_identity, 'Encoder state changed'
    plan = read_json(Path(request['output']) / 'plan.json')
    assert plan['variants'] == request['variants'] and plan['model'] == cpu_identity
    assert report['status'] == 'complete' and report['retained_temporary_bytes'] == 0
    write_json(Path(request['output']) / 'caller.json', {'request_sha256': file_digest(path),
        'model_loading_and_identity_seconds': loading_seconds, 'total_seconds': time.monotonic() - started,
        'encoder_state_unchanged': True, 'gpu_uuid': gpu, 'cpu_threads': torch.get_num_threads(),
        'scope': 'Exploratory multiple-fit calibration; not a single-checkpoint latency trial or training-overhead measurement.'})
    checked_request(path)


def dense_score(records, targets, classes):
    intersections, unions = [0] * classes, [0] * classes
    assert len(records) == len(targets) > 0
    for record, truth in zip(records, targets):
        ground = torch.bincount(truth[truth != 255].long().flatten(), minlength=classes).tolist()
        eligible = sum(ground)
        assert record['eligible_pixels'] == eligible
        assert len(record['intersection']) == len(record['union']) == classes
        predicted = []
        for index, (intersection, union, count) in enumerate(zip(record['intersection'], record['union'], ground)):
            assert type(intersection) is int and type(union) is int
            assert 0 <= intersection <= count <= union <= eligible
            prediction_count = union - count + intersection
            assert intersection <= prediction_count <= eligible
            predicted.append(prediction_count)
            intersections[index] += intersection
            unions[index] += union
        assert sum(predicted) == eligible
        prediction_hash = record['prediction_sha256']
        assert len(prediction_hash) == 64 and int(prediction_hash, 16) >= 0
    present = [i / u for i, u in zip(intersections, unions) if u]
    assert present
    return 100 * sum(present) / len(present)


def verify(path):
    request, tasks = checked_request(path)
    output = Path(request['output'])
    plan, report, caller = [read_json(output / name) for name in ['plan.json', 'calibration.json', 'caller.json']]
    assert caller['request_sha256'] == file_digest(path) and caller['encoder_state_unchanged'] is True
    assert caller['gpu_uuid'] == request['gpu_uuid'] and caller['cpu_threads'] == request['cpu_threads']
    assert plan['model'] == request['expected_model'] and plan['numerics'] == request['numerics']
    assert plan['variants'] == request['variants'] and plan['script_sha256'] == file_digest(request['measure_script'])
    if request['stage'] == 'selected-fuller':
        assert plan['reference_recipe'] == request['reference_recipe']
    if request['stage'] == 'selected-monitor':
        assert plan['monitor_recipe'] == request['monitor_recipe']
    assert report['status'] == 'complete' and report['plan_sha256'] == file_digest(output / 'plan.json')
    assert report['retained_temporary_bytes'] == 0 and not os.path.lexists(output / '.work')
    assert set(report['results']) == {variant['id'] for variant in request['variants']}
    proof = {'status': 'verified', 'request_sha256': file_digest(path),
        'plan_sha256': file_digest(output / 'plan.json'), 'calibration_sha256': file_digest(output / 'calibration.json'),
        'caller_sha256': file_digest(output / 'caller.json'), 'verifier_sha256': file_digest(__file__),
        'cases': {}, 'budget_checks': {}, 'budget_series': {}, 'optimization_parity': {}}
    for task in tasks:
        preparation_task = fuller_task(task) if request['stage'] in ('fuller-selection', 'selected-fuller') else task
        prepared, training = load_prepared(Path(request['data_root']), preparation_task, 'train', verify=False)
        _, rows = load_prepared(Path(request['data_root']), preparation_task, 'development', verify=False)
        if request['stage'] in ('fuller-selection', 'selected-fuller'):
            assert plan['reference_preparation'][task['id']] == preparation_task
        targets = [label_canvas(Path(prepared['source_root']), row, task)[0] for row in rows] if task['kind'] == 'segmentation' else None
        values = {}
        for variant in [v for v in plan['variants'] if v['calibration']['base_task'] == task['id']]:
            result = report['results'][variant['id']]
            assert result['preparation_digest'] == prepared['digest'] and result['sample_digests'] == prepared['splits']
            statistics = contained_path(output, result['statistics']['file'])
            assert file_digest(statistics) == result['statistics']['sha256']
            saved = read_json(statistics)
            assert saved['task'] == variant['id'] and saved['evaluation'] == 'development'
            records = saved['records']
            assert [(r['id'], r['seed']) for r in records] == [(r['id'], 0) for r in rows]
            if task['kind'] == 'classification':
                assert [r['label'] for r in records] == [r['target'] for r in rows]
                assert all(type(r['prediction']) is int and 0 <= r['prediction'] < task['classes'] for r in records)
                score = 100 * sum(r['label'] == r['prediction'] for r in records) / len(records)
            else:
                score = dense_score(records, targets, task['classes'])
            assert len(result['candidate_scores']) == len(result['selection']) == len(result['seed_results']) == 1
            selected = result['selection'][0]
            assert selected == result['candidate_scores'][0] and selected['seed'] == 0
            measured = result['metrics'][task['metric']]
            coherence = 1e-10 if targets is not None else 0.0
            assert math.isclose(score, measured, rel_tol=0, abs_tol=coherence)
            assert measured == selected['development_score'] == result['seed_results'][0]['score']
            assert selected['learning_rate'] == variant['probe']['learning_rates'][0]
            assert selected['weight_decay'] == variant['probe']['weight_decays'][0]
            if request['stage'] in ('fuller-selection', 'selected-fuller'):
                assert 1 <= selected['epoch'] <= variant['calibration']['reference_epochs']
                assert selected['updates'] == selected['epoch'] * math.ceil(len(training) / variant['probe']['batch_size'])
            elif request['stage'] == 'candidate-selection':
                steps = math.ceil(len(training) / variant['probe']['batch_size'])
                assert 1 <= selected['epoch'] <= math.ceil(variant['probe']['updates'] / steps)
                assert selected['updates'] == min(selected['epoch'] * steps, variant['probe']['updates'])
            else:
                assert selected['updates'] == variant['probe']['updates']
                assert selected['epoch'] == math.ceil(selected['updates'] / math.ceil(len(training) / variant['probe']['batch_size']))
            assert selected['probe_parameters'] == task['classes'] * (request['feature_widths'][task['method']] + 1)
            assert all(math.isfinite(selected[key]) for key in ['development_score', 'development_loss', 'training_loss'])
            assert len(selected['head_sha256']) == 64 and int(selected['head_sha256'], 16) >= 0
            calibration = variant['calibration']
            values[calibration['budget_multiplier'], calibration['feature_scale_before_standardization'], calibration['learning_rate_multiplier']] = measured
            proof['cases'][variant['id']] = {'score': measured, 'reconstructed_score': score,
                'metric_coherence_tolerance': coherence, 'records': len(records), 'statistics_sha256': file_digest(statistics)}
        rate_check = request['stage'] in ('convergence', 'fast-convergence', 'long-convergence', 'candidate-budget', 'candidate-long-budget', 'candidate-selection', 'candidate-regularized', 'candidate-regularized-long', 'candidate-regularized-tail', 'selected-monitor', 'fuller-selection', 'selected-fuller')
        budgets = (1, 2) if request['stage'] == 'fuller-selection' else (8, 16, 32) if request['stage'] == 'long-convergence' else (4, 8) if rate_check else (1, 2)
        if request['stage'] in ('candidate-budget', 'candidate-long-budget', 'candidate-selection', 'candidate-regularized', 'candidate-regularized-long', 'candidate-regularized-tail'):
            budgets = candidate_budgets(task['kind'], request['stage'])
        if request['stage'] == 'selected-fuller':
            factor = request['reference_recipe'][task['id']]['base_epochs'] // (50 if task['kind'] == 'classification' else 40)
            budgets = (factor, 2 * factor)
        if request['stage'] == 'selected-monitor':
            factor = request['monitor_recipe'][task['id']]['base_updates'] // task['probe']['updates']
            budgets = (factor, 2 * factor)
        if len(budgets) == 1:
            assert report['comparisons'][task['id']]['learning_rate_checks'] == {}
            proof['budget_checks'][task['id']] = {}
            continue
        tolerance = {'imagenet-100': 0.5, 'pascal': 0.25}[task['name']]
        checks = {}
        for rate in sorted({key[2] for key in values}):
            base, doubled = values[budgets[0], 1.0, rate], values[budgets[1], 1.0, rate]
            checks[str(rate)] = {'base_score': base, 'doubled_budget_score': doubled,
                'difference': doubled - base, 'tolerance': tolerance, 'passed': abs(doubled - base) <= tolerance}
            supplied = report['comparisons'][task['id']]['learning_rate_checks'][str(rate)] if rate_check else report['comparisons'][task['id']]
            assert (supplied['base_score'], supplied['doubled_budget_score'], supplied['budget_passed']) == (base, doubled, checks[str(rate)]['passed'])
        proof['budget_checks'][task['id']] = checks
        if request['stage'] == 'long-convergence':
            series = {str(rate): [{'base_updates': task['probe']['updates'] * first,
                'doubled_updates': task['probe']['updates'] * second,
                'base_score': values[first, 1.0, rate], 'doubled_budget_score': values[second, 1.0, rate],
                'difference': values[second, 1.0, rate] - values[first, 1.0, rate],
                'passed': abs(values[second, 1.0, rate] - values[first, 1.0, rate]) <= tolerance}
                for first, second in zip(budgets, budgets[1:])] for rate in sorted({key[2] for key in values})}
            assert series == report['comparisons'][task['id']]['budget_series']
            proof['budget_series'][task['id']] = series
        if not rate_check:
            differences = {str(scale): values[1, scale, 1] - values[1, 1.0, 1] for scale in (0.1, 10.0)}
            assert differences == report['comparisons'][task['id']]['scale_score_differences']
            assert report['comparisons'][task['id']]['scale_passed'] == all(abs(d) <= task['numerical_tolerance']['absolute_percentage_points'] for d in differences.values())
    for directory in request.get('parity_priors', []):
        directory = Path(directory)
        old_plan, old_report, old_proof = [read_json(directory / name) for name in ['plan.json', 'calibration.json', 'verification.json']]
        assert old_proof['status'] == 'verified' and old_proof['calibration_sha256'] == file_digest(directory / 'calibration.json')
        assert old_proof['plan_sha256'] == file_digest(directory / 'plan.json') and old_plan['model'] == plan['model']
        old_variants = {v['id']: v for v in old_plan['variants']}
        for variant in plan['variants']:
            if variant['id'] not in old_variants:
                continue
            assert variant == old_variants[variant['id']]
            before, after = old_report['results'][variant['id']], report['results'][variant['id']]
            assert before['preparation_digest'] == after['preparation_digest'] and before['sample_digests'] == after['sample_digests']
            first, second = before['selection'][0], after['selection'][0]
            assert all(first[k] == second[k] for k in ['seed', 'epoch', 'updates', 'learning_rate', 'weight_decay', 'probe_parameters'])
            difference = after['metrics'][variant['metric']] - before['metrics'][variant['metric']]
            allowed = variant['numerical_tolerance']['absolute_percentage_points']
            assert abs(difference) <= allowed, 'Target-reuse optimization exceeded registered score tolerance'
            proof['optimization_parity'][variant['id']] = {'prior': str(directory), 'difference': difference,
                'tolerance': allowed, 'passed': True, 'head_hashes_identical': first['head_sha256'] == second['head_sha256']}
    if request.get('parity_priors'):
        assert len(proof['optimization_parity']) == request['expected_parity_cases']
    proof['verified_records'] = sum(row['records'] for row in proof['cases'].values())
    proof['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
    proof['scope'] = 'Independent compact-record/selection/source-count reconstruction. Budget failures remain findings; verification does not establish monitoring agreement, precision, reserved acceptance, latency, overhead or v1 readiness.'
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
