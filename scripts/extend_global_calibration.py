"""Fit only predeclared missing global-probe cases and verify compact records.

The request pins its package, checkpoint, prior evidence and prepared manifests.
Existing fits are read as evidence; they are never silently fitted again.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import bench
from bench.features import build_bank, feature_diagnostics, feature_identity, model_identity, numerical_mode
from bench.prepare import load_prepared
from bench.probes import evaluate_selection, fit_probe, probe_identity
from bench.protocols import resolve_names
from bench.storage import TemporaryStorage, contained_path, digest, file_digest, lock, read_json, remove, write_json
from bench.workspace import clear, initialize, workspace_lock


def variant(base, epochs, rates, decays, purpose):
    if base['kind'] != 'classification' or base['method'] != 'global':
        raise ValueError('This extension accepts global classification probes only')
    if type(epochs) is not int or epochs < 1 or not rates or not decays:
        raise ValueError('Declare positive epoch budgets and nonempty search axes')
    if any(not math.isfinite(x) or x <= 0 for x in rates) or any(not math.isfinite(x) or x < 0 for x in decays):
        raise ValueError('Invalid search axis')
    if len(set(rates)) != len(rates) or len(set(decays)) != len(decays):
        raise ValueError('Duplicate search candidates')
    result = deepcopy(base)
    # The request declares missing fixed fits, independent of the base controller.
    result['probe'].pop('search_plan', None)
    result['probe'].pop('budget_validation', None)
    result['probe'].update(epochs=epochs, learning_rates=rates, weight_decays=decays, seeds=[0])
    declaration = {'base_task': base['id'], 'base_task_digest': base['digest'],
                   'epochs': epochs, 'learning_rates': rates, 'weight_decays': decays, 'purpose': purpose}
    result['id'] = base['id'] + '-calibration-tail-' + digest(declaration)[:16]
    result.update(name=result['id'], release_status='calibration-only', calibration=declaration)
    result['digest'] = digest({k: v for k, v in result.items() if k != 'digest'})
    return result


def previous_candidates(prior, task):
    if prior['kind'] == 'combined':
        rows, seen = [], {}
        for part in prior['parts']:
            for epochs, candidate in previous_candidates(part, task):
                key = epochs, candidate.get('seed', 0), candidate['learning_rate'], candidate['weight_decay']
                if key in seen:
                    if seen[key] != candidate:
                        raise ValueError('Conflicting previous candidate evidence')
                    continue
                seen[key] = candidate
                rows.append((epochs, candidate))
        return rows
    directory = Path(prior['output'])
    if prior['kind'] == 'run':
        result = read_json(directory / 'results.json')['tasks'][task['id']]
        return [(task['probe']['epochs'], c) for c in result['selection']['candidates']]
    plan, report = read_json(directory / 'plan.json'), read_json(directory / 'calibration.json')
    variants = plan['cases'] if prior['kind'] == 'extension' else plan['variants']
    return [(v['probe']['epochs'], c) for v in variants
            if v['calibration']['base_task'] == task['id']
            and v['id'] in report['results']
            for c in report['results'][v['id']]['candidate_scores']]


def needed(case, condition, known):
    if condition is None:
        return True
    budget = condition['reference_epochs']
    candidates = [c for epochs, c in known if epochs == budget]
    assert candidates, 'Missing declared selection evidence'
    winner = min(enumerate(candidates), key=lambda pair: (-pair[1]['development_score'], pair[1]['epoch'], pair[0]))[1]
    if condition['kind'] == 'upper-weight-decay':
        return winner['weight_decay'] == condition['weight_decay']
    if condition['kind'] == 'winner-parameters':
        return all(winner[key] == value for key, value in condition['parameters'].items())
    assert condition['kind'] == 'winner-needs-budget'
    rate, decay = case['probe']['learning_rates'], case['probe']['weight_decays']
    assert len(rate) == len(decay) == 1
    if (winner['learning_rate'], winner['weight_decay']) != (rate[0], decay[0]):
        return False
    if 'compare_epochs' not in condition:
        return True
    scores = {epochs: c['development_score'] for epochs, c in known
              if c['learning_rate'] == rate[0] and c['weight_decay'] == decay[0]}
    first, second = condition['compare_epochs']
    return first in scores and second in scores and abs(scores[second] - scores[first]) > condition['tolerance']


def checked_request(path):
    request = read_json(path)
    assert all(file_digest(p) == expected for p, expected in request['pins'].items()), 'Pinned input changed'
    assert Path(bench.__file__).resolve().parent == Path(request['package_root']).resolve()
    tasks = resolve_names(request['names'])
    assert set(request['priors']) == {task['id'] for task in tasks}
    declared = set()
    for case in request['cases']:
        task = next(t for t in tasks if t['id'] == case['calibration']['base_task'])
        assert case == variant(task, case['probe']['epochs'], case['probe']['learning_rates'],
                               case['probe']['weight_decays'], case['calibration']['purpose'])
        known = {(epochs, c['learning_rate'], c['weight_decay']) for epochs, c in previous_candidates(request['priors'][task['id']], task)}
        for rate, decay in product(case['probe']['learning_rates'], case['probe']['weight_decays']):
            key = (task['id'], case['probe']['epochs'], rate, decay)
            assert key not in declared, 'Duplicate new candidate'
            assert (case['probe']['epochs'], rate, decay) not in known, 'Completed candidate must not be refitted'
            declared.add(key)
    assert {c['calibration']['base_task'] for c in request['cases']} == {t['id'] for t in tasks}
    conditions = request.get('conditions', {})
    assert set(conditions) <= {case['id'] for case in request['cases']}
    for condition in conditions.values():
        assert condition['kind'] in ('upper-weight-decay', 'winner-parameters', 'winner-needs-budget')
        assert type(condition['reference_epochs']) is int and condition['reference_epochs'] > 0
        if condition['kind'] == 'winner-parameters':
            parameters = condition['parameters']
            assert parameters and set(parameters) <= {'learning_rate', 'weight_decay'}
            assert all(math.isfinite(value) and value >= 0 for value in parameters.values())
            assert 'learning_rate' not in parameters or parameters['learning_rate'] > 0
        if 'compare_epochs' in condition:
            assert len(condition['compare_epochs']) == 2 and all(type(e) is int and e > 0 for e in condition['compare_epochs'])
            assert math.isfinite(condition['tolerance']) and condition['tolerance'] >= 0
    return request, tasks


def run(request_path):
    request, tasks = checked_request(request_path)
    output = Path(request['output'])
    assert not output.exists(), 'Inspect existing output before any recovery'
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip()
    assert gpu == request['gpu_uuid']
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU occupied'
    torch.set_num_threads(request['cpu_threads'])
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = instantiate(OmegaConf.load(request['model_yaml']), _convert_='all')
    execution = bench.Execution(device='cuda', numerics=request['numerics'], batch_size=request['batch_size'], workers=request['workers'])
    cpu_identity = model_identity(model, bench.Execution(device='cpu', numerics=request['numerics']))
    assert cpu_identity == request['expected_model']
    model.cuda()
    started = time.monotonic()
    work, bank, bank_key = output / '.work', None, None
    with lock(output.parent / ('.' + output.name + '.bench-output-lock')), workspace_lock(work), numerical_mode(execution):
        metadata = model_identity(model, execution)
        assert metadata == cpu_identity
        initialize(work, metadata['digest'])
        storage = TemporaryStorage(work)
        plan = {'request_sha256': file_digest(request_path), 'model': metadata, 'cases': request['cases'],
                'cpu_threads': torch.get_num_threads(), 'numerics': execution.numerics,
                'gpu_uuid': gpu, 'gpu': torch.cuda.get_device_name(), 'script_sha256': file_digest(__file__),
                'scope': 'Predeclared missing development candidates only; no reserved final examples.'}
        write_json(output / 'plan.json', plan)
        report = {'status': 'running', 'plan_sha256': file_digest(output / 'plan.json'), 'results': {}, 'extraction': {}, 'skipped': {}}
        write_json(output / 'calibration.json', report)
        torch.cuda.reset_peak_memory_stats()
        try:
            for task in tasks:
                prepared, training = load_prepared(Path(request['data_root']), task, 'train')
                _, development = load_prepared(Path(request['data_root']), task, 'development')
                rows = training + development
                key = digest(feature_identity(metadata, task, rows))
                if key != bank_key:
                    if bank is not None:
                        bank.close()
                        remove(bank.directory, storage=storage)
                    bank, usage = build_bank(model, model_metadata=metadata, task=task,
                        root=Path(prepared['source_root']), rows=rows, work_dir=work, execution=execution, storage=storage)
                    bank_key = key
                else:
                    usage = {'extracted_examples': 0, 'reused_examples': len(rows), 'timing': {}}
                report['extraction'][task['id']] = {k: v for k, v in usage.items() if k != 'extracted_ids'}
                diagnostics = feature_diagnostics(bank, prepared['diagnostic_sample_ids'])
                known = previous_candidates(request['priors'][task['id']], task)
                for case in (v for v in request['cases'] if v['calibration']['base_task'] == task['id']):
                    condition = request.get('conditions', {}).get(case['id'])
                    if not needed(case, condition, known):
                        report['skipped'][case['id']] = condition
                        write_json(output / 'calibration.json', report)
                        continue
                    print('Starting', case['id'], 'epochs', case['probe']['epochs'],
                          'rates', case['probe']['learning_rates'], 'decays', case['probe']['weight_decays'], flush=True)
                    begin = time.monotonic()
                    directory = work / 'probes' / probe_identity(case, prepared, metadata)
                    selection = fit_probe(bank, training=training, development=development,
                        root=Path(prepared['source_root']), task=case, directory=directory,
                        execution=execution, diagnostics={'raw_encoder': diagnostics})
                    metrics, seeds, records = evaluate_selection(selection, directory, bank, development,
                        Path(prepared['source_root']), case, execution)
                    statistics = output / 'statistics' / (case['id'] + '.json')
                    write_json(statistics, {'task': case['id'], 'evaluation': 'development', 'records': records})
                    report['results'][case['id']] = {'metrics': metrics, 'seed_results': seeds,
                        'selection': [{k: v for k, v in c.items() if k != 'head_file'} for c in selection['seeds']],
                        'candidate_scores': [{k: v for k, v in c.items() if k != 'head_file'} for c in selection['candidates']],
                        'preparation_digest': prepared['digest'], 'sample_digests': prepared['splits'],
                        'statistics': {'file': str(statistics.relative_to(output)), 'sha256': file_digest(statistics)},
                        'fitting_and_scoring_seconds': time.monotonic() - begin}
                    write_json(output / 'calibration.json', report)
                    known.extend((case['probe']['epochs'], c) for c in report['results'][case['id']]['candidate_scores'])
                    remove(directory, storage=storage)
                    print('Completed', case['id'], metrics, flush=True)
            if bank is not None:
                bank.close()
                bank = None
            clear(work, storage=storage)
            assert model_identity(model, execution) == metadata, 'Encoder state changed'
            report.update(status='complete', total_seconds=time.monotonic() - started,
                peak_temporary_bytes=storage.peak, retained_temporary_bytes=storage.current,
                encoder_state_unchanged=True,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(), peak_host_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            write_json(output / 'calibration.json', report)
        except BaseException as error:
            if bank is not None:
                bank.close()
            report.update(status='failed', error=repr(error))
            write_json(output / 'calibration.json', report)
            raise
    checked_request(request_path)


def verify(request_path):
    request, tasks = checked_request(request_path)
    output = Path(request['output'])
    plan, report = read_json(output / 'plan.json'), read_json(output / 'calibration.json')
    assert plan['request_sha256'] == file_digest(request_path)
    assert plan['model'] == request['expected_model'] and plan['cases'] == request['cases']
    assert report['status'] == 'complete' and report['plan_sha256'] == file_digest(output / 'plan.json')
    assert report['encoder_state_unchanged'] is True
    assert report['retained_temporary_bytes'] == 0 and not os.path.lexists(output / '.work')
    assert set(report['results']) | set(report.get('skipped', {})) == {v['id'] for v in request['cases']}
    assert not (set(report['results']) & set(report.get('skipped', {})))
    proof = {'status': 'verified', 'request_sha256': file_digest(request_path),
        'plan_sha256': file_digest(output / 'plan.json'), 'calibration_sha256': file_digest(output / 'calibration.json'),
        'verifier_sha256': file_digest(__file__), 'cases': {}, 'search_summary': {}, 'skipped': {}}
    for task in tasks:
        prepared, training = load_prepared(Path(request['data_root']), task, 'train', verify=False)
        _, development = load_prepared(Path(request['data_root']), task, 'development', verify=False)
        combined = previous_candidates(request['priors'][task['id']], task)
        for case in (v for v in request['cases'] if v['calibration']['base_task'] == task['id']):
            condition = request.get('conditions', {}).get(case['id'])
            if not needed(case, condition, combined):
                assert case['id'] not in report['results']
                assert report['skipped'][case['id']] == condition
                proof['skipped'][case['id']] = condition
                continue
            assert case['id'] in report['results'] and case['id'] not in report.get('skipped', {})
            result = report['results'][case['id']]
            assert result['preparation_digest'] == prepared['digest'] and result['sample_digests'] == prepared['splits']
            statistics = result['statistics']; path = contained_path(output, statistics['file'])
            assert file_digest(path) == statistics['sha256']
            saved = read_json(path); records = saved['records']
            assert saved['task'] == case['id'] and saved['evaluation'] == 'development'
            assert [(r['id'], r['label'], r['seed']) for r in records] == [(r['id'], r['target'], 0) for r in development]
            assert all(type(r['prediction']) is int and 0 <= r['prediction'] < task['classes'] for r in records)
            score = 100 * sum(r['prediction'] == r['label'] for r in records) / len(records)
            candidates = result['candidate_scores']; probe = case['probe']
            assert [(c['learning_rate'], c['weight_decay'], c['seed']) for c in candidates] == [(lr, wd, 0) for lr, wd in product(probe['learning_rates'], probe['weight_decays'])]
            winner = min(range(len(candidates)), key=lambda i: (-candidates[i]['development_score'], candidates[i]['epoch'], i))
            assert result['selection'] == [candidates[winner]]
            assert score == result['metrics'][task['metric']] == candidates[winner]['development_score'] == result['seed_results'][0]['score']
            for c in candidates:
                assert 1 <= c['epoch'] <= probe['epochs'] and c['updates'] == c['epoch'] * math.ceil(len(training) / probe['batch_size'])
                assert c['probe_parameters'] == task['classes'] * (request['feature_width'] + 1)
                assert all(math.isfinite(c[k]) for k in ['training_loss', 'development_loss', 'development_score'])
                assert 0 <= c['development_score'] <= 100
                assert len(c['head_sha256']) == 64 and int(c['head_sha256'], 16) >= 0
            combined.extend((probe['epochs'], c) for c in candidates)
            proof['cases'][case['id']] = {'records': len(records), 'score': score, 'statistics_sha256': file_digest(path),
                'candidates': len(candidates), 'selected': candidates[winner]}
        summary = {}
        for epochs in sorted({e for e, _ in combined}):
            rows = [c for e, c in combined if e == epochs]
            assert len({(c['learning_rate'], c['weight_decay']) for c in rows}) == len(rows), 'Duplicated candidate evidence'
            best = min(enumerate(rows), key=lambda pair: (-pair[1]['development_score'], pair[1]['epoch'], pair[0]))[1]
            rates = sorted({c['learning_rate'] for c in rows})
            decays = sorted({c['weight_decay'] for c in rows})
            coverage = 'fixed-setting budget check' if len(rows) == 1 else (
                'complete declared grid' if len(rows) == len(rates) * len(decays) else 'partial declared grid')
            summary[str(epochs)] = {'score': best['development_score'], 'learning_rate': best['learning_rate'],
                'weight_decay': best['weight_decay'], 'candidates': len(rows), 'coverage': coverage,
                'learning_rates': rates, 'weight_decays': decays,
                'lower_learning_rate_boundary': best['learning_rate'] == min(rates) if len(rates) > 1 else None,
                'upper_learning_rate_boundary': best['learning_rate'] == max(rates) if len(rates) > 1 else None,
                'upper_weight_decay_boundary': best['weight_decay'] == max(decays) if len(decays) > 1 else None}
        proof['search_summary'][task['id']] = summary
    proof['verified_records'] = sum(c['records'] for c in proof['cases'].values())
    assert not torch.cuda.is_initialized()
    proof['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
    proof['scope'] = 'Reconstructed new development records and combined candidate search evidence; fixed-setting budget checks are not complete searches, and remaining release gates are separate.'
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
        print(json.dumps(verify(args.request), indent=2))
