"""Extend only a selected fuller-reference probe's unresolved epoch budget."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import types

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import bench
from bench.features import build_bank, feature_diagnostics, model_identity, numerical_mode
from bench.prepare import load_prepared
from bench.probes import evaluate_selection, fit_probe, label_canvas, probe_identity
from bench.protocols import resolve_names
from bench.storage import TemporaryStorage, contained_path, digest, file_digest, lock, read_json, remove, write_json
from bench.workspace import clear, initialize, workspace_lock
from monitor_calibration import declared_variants, dense_score, fuller_task


def declared_tail(task, count, reference_recipe=None, doublings=3):
    assert type(doublings) is int and 1 <= doublings <= 3
    if reference_recipe is None:
        assert task['kind'] == 'segmentation', 'Classification tails require explicit selected settings'
        baseline = next(v for v in declared_variants([task], 'fuller-selection', {task['id']: count})
                        if v['calibration']['budget_multiplier'] == 2 and v['calibration']['learning_rate_multiplier'] == 3)
    else:
        baseline = declared_variants([task], 'selected-fuller', {task['id']: count},
                                     {task['id']: reference_recipe})[0]
    rate = baseline['calibration']['learning_rate_multiplier']
    cases = []
    for factor in [2 ** i for i in range(1, doublings + 1)]:
        variant = deepcopy(baseline)
        variant['probe']['updates'] *= factor
        variant['calibration']['budget_multiplier'] *= factor
        variant['calibration']['reference_epochs'] *= factor
        variant['id'] = f"{fuller_task(task)['id']}-calibration-u{variant['probe']['updates']}-s1-lrm{rate}-bestdev"
        variant['name'] = variant['id']
        variant['digest'] = digest({k: v for k, v in variant.items() if k != 'digest'})
        cases.append(variant)
    return baseline, cases


def budget_check(first_updates, first_score, second_updates, second_score, tolerance=0.25):
    assert second_updates == 2 * first_updates and all(math.isfinite(x) for x in [first_score, second_score])
    assert tolerance in (0.25, 0.5)
    difference = second_score - first_score
    return {'base_updates': first_updates, 'doubled_updates': second_updates,
            'base_score': first_score, 'doubled_budget_score': second_score,
            'difference': difference, 'tolerance': tolerance, 'passed': abs(difference) <= tolerance}


def optimization_check(previous_score, measured, tolerance):
    assert all(math.isfinite(x) for x in [previous_score, measured, tolerance]) and tolerance >= 0
    difference = measured - previous_score
    return {'prior_score': previous_score, 'current_score': measured, 'difference': difference,
            'tolerance': tolerance, 'passed': abs(difference) <= tolerance,
            'scope': 'Overlapping budget under a changed fitter; subsequent budget comparisons use the current fitter on both sides.'}


def checked_request(path):
    request = read_json(path)
    assert all(file_digest(p) == h for p, h in request['pins'].items())
    assert Path(bench.__file__).resolve().parent == Path(request['package_root']).resolve()
    name = request.get('base_task', 'pascal')
    assert name in ('imagenet-100', 'pascal')
    task = resolve_names([name])[0]
    reference = fuller_task(task)
    prepared, rows = load_prepared(Path(request['data_root']), reference, 'train', verify=False)
    assert len(rows) == request['reference_training_count']
    baseline, cases = declared_tail(task, len(rows), request.get('reference_recipe'), request.get('doublings', 3))
    revalidate = request.get('revalidate_prior', False)
    assert type(revalidate) is bool
    assert request['cases'] == ([baseline, *cases] if revalidate else cases) and request['prior_variant'] == baseline
    previous = Path(request['prior_request'])
    old = read_json(previous)
    state = read_json(previous.with_name(previous.stem + '-state.json'))
    output = Path(old['output'])
    assert state['status'] == 'complete' and state['request_sha256'] == file_digest(previous)
    plan, report, proof = [read_json(output / name) for name in ['plan.json', 'calibration.json', 'verification.json']]
    assert proof['status'] == 'verified' and proof['request_sha256'] == file_digest(previous)
    assert file_digest(output / 'verification.json') == state['verification_sha256']
    assert file_digest(output / 'plan.json') == state['plan_sha256'] == proof['plan_sha256'] == report['plan_sha256']
    assert file_digest(output / 'calibration.json') == state['calibration_sha256'] == proof['calibration_sha256']
    assert plan['model'] == request['expected_model'] and baseline in plan['variants']
    result = report['results'][baseline['id']]
    assert result['preparation_digest'] == prepared['digest'] and result['sample_digests'] == prepared['splits']
    assert result['metrics'][task['metric']] == proof['cases'][baseline['id']]['score']
    return request, task, reference, prepared, result['metrics'][task['metric']]


def run(path):
    request, task, reference, prepared, previous_score = checked_request(path)
    output = Path(request['output'])
    assert not output.exists(), 'Preserve any existing fit'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() == request['gpu_uuid']
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    torch.set_num_threads(request['cpu_threads'])
    if request.get('open_clip_source'):
        namespace = types.ModuleType('open_clip')
        namespace.__path__ = [str(Path(request['open_clip_source']) / 'src/open_clip')]
        sys.modules['open_clip'] = namespace
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        model = instantiate(OmegaConf.load(request['model_yaml']), _convert_='all')
    assert model_identity(model, bench.Execution(device='cpu', numerics='fp32-v1')) == request['expected_model']
    model.cuda()
    execution = bench.Execution(device='cuda', numerics='fp32-v1', batch_size=request['batch_size'], workers=request['workers'])
    work, bank = output / '.work', None
    started = time.monotonic()
    with lock(output.parent / f'.{output.name}.bench-output-lock'), workspace_lock(work), numerical_mode(execution):
        assert model_identity(model, execution) == request['expected_model']
        initialize(work, request['expected_model']['digest'])
        storage = TemporaryStorage(work)
        torch.cuda.reset_peak_memory_stats()
        plan = {'request_sha256': file_digest(path), 'model': request['expected_model'], 'cases': request['cases'],
                'reference_preparation': reference, 'prior_score': previous_score, 'script_sha256': file_digest(__file__),
                'gpu_uuid': request['gpu_uuid'], 'execution': {'batch_size': execution.batch_size, 'workers': execution.workers},
                'gpu': torch.cuda.get_device_name(), 'cuda_runtime': torch.version.cuda, 'cpu_threads': torch.get_num_threads(),
                'scope': 'Winner-only fuller reference budget; one feature extraction, no encoder training or final examples.'}
        write_json(output / 'plan.json', plan)
        report = {'status': 'running', 'plan_sha256': file_digest(output / 'plan.json'), 'results': {}, 'skipped': {}, 'budget_series': []}
        write_json(output / 'calibration.json', report)
        try:
            prepared, training = load_prepared(Path(request['data_root']), reference, 'train')
            _, development = load_prepared(Path(request['data_root']), reference, 'development')
            root = Path(prepared['source_root'])
            bank, usage = build_bank(model, model_metadata=request['expected_model'], task=task, root=root,
                                    rows=training + development, work_dir=work, execution=execution, storage=storage)
            report['extraction'] = {k: v for k, v in usage.items() if k != 'extracted_ids'}
            diagnostics = feature_diagnostics(bank, prepared['diagnostic_sample_ids'])
            previous_updates = request['prior_variant']['probe']['updates']
            for variant in request['cases']:
                if report['budget_series'] and report['budget_series'][-1]['passed']:
                    report['skipped'][variant['id']] = report['budget_series'][-1]
                    continue
                print('Starting fuller reference:', variant['calibration']['reference_epochs'], 'probe epochs', flush=True)
                begin = time.monotonic()
                directory = work / 'probes' / probe_identity(variant, prepared, request['expected_model'])
                selected = fit_probe(bank, training=training, development=development, root=root, task=variant,
                                     directory=directory, execution=execution, diagnostics={'raw_encoder': diagnostics})
                metrics, seeds, records = evaluate_selection(selected, directory, bank, development, root, variant, execution)
                statistics = output / 'statistics' / (variant['id'] + '.json')
                write_json(statistics, {'task': variant['id'], 'evaluation': 'development', 'records': records})
                candidates = [{k: v for k, v in c.items() if k != 'head_file'} for c in selected['candidates']]
                report['results'][variant['id']] = {'metrics': metrics, 'seed_results': seeds, 'selection': candidates,
                    'candidate_scores': candidates, 'preparation_digest': prepared['digest'], 'sample_digests': prepared['splits'],
                    'statistics': {'file': str(statistics.relative_to(output)), 'sha256': file_digest(statistics)},
                    'fitting_and_scoring_seconds': time.monotonic() - begin}
                if variant == request['prior_variant']:
                    report['optimization_parity'] = optimization_check(previous_score, metrics[task['metric']],
                        task['numerical_tolerance']['absolute_percentage_points'])
                    assert report['optimization_parity']['passed'], 'Overlapping reference fit exceeded numerical tolerance'
                else:
                    report['budget_series'].append(budget_check(previous_updates, previous_score,
                        variant['probe']['updates'], metrics[task['metric']], 0.5 if task['kind'] == 'classification' else 0.25))
                previous_updates, previous_score = variant['probe']['updates'], metrics[task['metric']]
                write_json(output / 'calibration.json', report)
                remove(directory, storage=storage)
                print('Completed fuller reference:', metrics, flush=True)
            bank.close()
            bank = None
            clear(work, storage=storage)
            assert model_identity(model, execution) == request['expected_model']
            report.update(status='complete', encoder_state_unchanged=True, total_seconds=time.monotonic() - started,
                          peak_temporary_bytes=storage.peak, retained_temporary_bytes=storage.current,
                          peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                          peak_host_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
            write_json(output / 'calibration.json', report)
        except BaseException as error:
            if bank is not None:
                bank.close()
            report.update(status='failed', error=repr(error))
            write_json(output / 'calibration.json', report)
            raise
    checked_request(path)


def verify(path):
    request, task, reference, prepared, previous_score = checked_request(path)
    output = Path(request['output'])
    plan, report = read_json(output / 'plan.json'), read_json(output / 'calibration.json')
    assert plan['request_sha256'] == file_digest(path) and plan['cases'] == request['cases'] and plan['model'] == request['expected_model']
    assert plan['reference_preparation'] == reference and plan['prior_score'] == previous_score
    assert plan['script_sha256'] == file_digest(__file__) and plan['gpu_uuid'] == request['gpu_uuid']
    assert report['status'] == 'complete' and report['plan_sha256'] == file_digest(output / 'plan.json')
    assert report['encoder_state_unchanged'] and report['retained_temporary_bytes'] == 0 and not os.path.lexists(output / '.work')
    assert set(report['results']) | set(report['skipped']) == {v['id'] for v in request['cases']}
    assert not set(report['results']) & set(report['skipped'])
    _, rows = load_prepared(Path(request['data_root']), reference, 'development', verify=False)
    targets = [label_canvas(Path(prepared['source_root']), row, task)[0] for row in rows] if task['kind'] == 'segmentation' else None
    proof = {'status': 'verified', 'request_sha256': file_digest(path), 'plan_sha256': file_digest(output / 'plan.json'),
             'calibration_sha256': file_digest(output / 'calibration.json'), 'verifier_sha256': file_digest(__file__),
             'cases': {}, 'skipped': {}, 'budget_series': []}
    previous_updates = request['prior_variant']['probe']['updates']
    for variant in request['cases']:
        if proof['budget_series'] and proof['budget_series'][-1]['passed']:
            assert report['skipped'][variant['id']] == proof['budget_series'][-1]
            proof['skipped'][variant['id']] = report['skipped'][variant['id']]
            continue
        saved = report['results'][variant['id']]
        assert saved['preparation_digest'] == prepared['digest'] and saved['sample_digests'] == prepared['splits']
        statistics = contained_path(output, saved['statistics']['file'])
        assert file_digest(statistics) == saved['statistics']['sha256']
        item = read_json(statistics)
        assert item['task'] == variant['id'] and item['evaluation'] == 'development'
        records = item['records']
        assert [(r['id'], r['seed']) for r in records] == [(r['id'], 0) for r in rows]
        if targets is None:
            assert [r['label'] for r in records] == [r['target'] for r in rows]
            assert all(type(r['prediction']) is int and 0 <= r['prediction'] < task['classes'] for r in records)
            score = 100 * sum(r['label'] == r['prediction'] for r in records) / len(records)
        else:
            score = dense_score(records, targets, task['classes'])
        measured = saved['metrics'][task['metric']]
        assert math.isclose(score, measured, rel_tol=0, abs_tol=1e-10)
        assert len(saved['selection']) == len(saved['candidate_scores']) == len(saved['seed_results']) == 1
        candidate = saved['selection'][0]
        assert candidate == saved['candidate_scores'][0]
        assert measured == candidate['development_score'] == saved['seed_results'][0]['score']
        assert saved['seed_results'][0]['seed'] == 0 and saved['seed_results'][0]['sample_count'] == len(rows)
        eligible = sum(r['eligible_pixels'] for r in records) if targets is not None else len(rows)
        assert saved['seed_results'][0]['eligible_count'] == eligible
        assert candidate['seed'] == 0 and candidate['learning_rate'] == variant['probe']['learning_rates'][0]
        assert candidate['weight_decay'] == variant['probe']['weight_decays'][0]
        assert 1 <= candidate['epoch'] <= variant['calibration']['reference_epochs']
        assert candidate['updates'] == candidate['epoch'] * math.ceil(request['reference_training_count'] / task['probe']['batch_size'])
        assert candidate['probe_parameters'] == task['classes'] * (request['feature_width'] + 1)
        assert all(math.isfinite(candidate[k]) for k in ['development_loss', 'training_loss', 'development_score'])
        assert len(candidate['head_sha256']) == 64 and int(candidate['head_sha256'], 16) >= 0
        proof['cases'][variant['id']] = {'score': measured, 'reconstructed_score': score, 'records': len(rows), 'statistics_sha256': file_digest(statistics)}
        if variant == request['prior_variant']:
            proof['optimization_parity'] = optimization_check(previous_score, measured,
                task['numerical_tolerance']['absolute_percentage_points'])
            assert proof['optimization_parity'] == report['optimization_parity'] and proof['optimization_parity']['passed']
        else:
            proof['budget_series'].append(budget_check(previous_updates, previous_score, variant['probe']['updates'],
                                                      measured, 0.5 if task['kind'] == 'classification' else 0.25))
        previous_updates, previous_score = variant['probe']['updates'], measured
    assert proof['budget_series'] == report['budget_series']
    assert report['extraction']['extracted_examples'] == request['reference_training_count'] + len(rows) and report['extraction']['reused_examples'] == 0
    proof['verified_records'] = sum(row['records'] for row in proof['cases'].values())
    proof['checked_at_utc'] = datetime.now(timezone.utc).isoformat()
    proof['scope'] = 'Selected fuller-reference budget only. A failing last doubling remains unresolved; other checkpoint references, agreement, reserved acceptance, latency and overhead remain separate.'
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
