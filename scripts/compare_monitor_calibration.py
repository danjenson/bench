"""Compare committed calibration statistics without fitting or reading final data."""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from bench.storage import contained_path, digest, file_digest, read_json, write_json


def case_metadata(request_path, request, output, plan, report, proof):
    """Read recorded runtime fields and the two pinned worker report schemas."""
    if 'variants' in plan:
        caller = read_json(output / 'caller.json')
        assert file_digest(output / 'caller.json') == proof['caller_sha256']
        assert caller['request_sha256'] == file_digest(request_path) and caller['encoder_state_unchanged'] is True
        assert plan['variants'] == request['variants'] and plan['numerics'] == request['numerics']
        assert caller['gpu_uuid'] == request['gpu_uuid'] and caller['cpu_threads'] == request['cpu_threads']
        variants, software, recorded = plan['variants'], plan['software'], plan['hardware']
        execution, threads = plan['execution'], caller['cpu_threads']
        schema = 'monitor-calibration'
    else:
        # These winner-only reference extensions keep equivalent compact records,
        # but their worker records the runtime header directly rather than a caller.json.
        assert request['numerics'] == 'fp32-v1'
        assert request['cases'] == plan['cases']
        if request.get('revalidate_prior'):
            assert request['prior_variant'] in request['cases']
        assert plan['request_sha256'] == file_digest(request_path) and report['encoder_state_unchanged'] is True
        assert plan['gpu_uuid'] == request['gpu_uuid'] and plan['cpu_threads'] == request['cpu_threads']
        assert request['reference_training_count'] > 0
        variants = plan['cases']
        kinds = {v['kind'] for v in variants}
        assert kinds in ({'segmentation'}, {'classification'}) and all('reference_bank' in v for v in variants)
        if kinds == {'classification'}:
            assert request['base_task'] == 'imagenet-100' and request['reference_recipe']
        source = Path(request['package_root'])
        files = list(source.rglob('*.py'))
        assert files and all(str(p) in request['pins'] and file_digest(p) == request['pins'][str(p)] for p in files)
        software = {str(p.relative_to(source)): request['pins'][str(p)] for p in files}
        assert software['features.py'] == plan['model']['extraction_implementation_sha256']
        assert 'probes.py' in software
        recorded = {key: plan[key] for key in ['gpu', 'gpu_uuid', 'cuda_runtime', 'cpu_threads']}
        # The checksum-pinned extension worker constructs Execution(device='cuda').
        execution, threads = {'device': 'cuda', **plan['execution']}, plan['cpu_threads']
        schema = 'winner-only-dense-reference-budget' if kinds == {'segmentation'} else 'winner-only-global-reference-budget'
    assert execution == {'device': 'cuda', 'batch_size': request['batch_size'], 'workers': request['workers']}
    assert type(threads) is int and threads > 0
    hardware = {'gpu_uuid': request['gpu_uuid'], 'gpu': recorded['gpu'],
                'cuda_runtime': recorded['cuda_runtime'], 'cpu_threads': threads}
    return variants, software, hardware, execution, recorded, schema


def load_case(request_path, variant_id):
    request_path = Path(request_path).resolve()
    request = read_json(request_path)
    state = read_json(request_path.with_name(request_path.stem + '-state.json'))
    output = Path(request['output'])
    assert state['status'] == 'complete' and state['request_sha256'] == file_digest(request_path)
    for name, key in [('plan.json', 'plan_sha256'), ('calibration.json', 'calibration_sha256'),
                      ('verification.json', 'verification_sha256')]:
        assert file_digest(output / name) == state[key], f'Changed committed {name}'
    plan, report, proof = [read_json(output / name) for name in ['plan.json', 'calibration.json', 'verification.json']]
    assert proof['status'] == 'verified' and report['status'] == 'complete'
    assert proof['request_sha256'] == state['request_sha256']
    assert proof['plan_sha256'] == report['plan_sha256'] == state['plan_sha256']
    assert proof['calibration_sha256'] == state['calibration_sha256']
    assert proof['verifier_sha256'] == request['pins'][request['worker']]
    assert plan['model'] == request['expected_model']
    assert plan['model']['digest'] == digest({k: v for k, v in plan['model'].items() if k != 'digest'})
    assert request['numerics'] == plan['model']['numerics']
    variants, software, hardware, execution, recorded, schema = case_metadata(request_path, request, output, plan, report, proof)
    variant = next(v for v in variants if v['id'] == variant_id)
    result = report['results'][variant_id]
    statistics = contained_path(output, result['statistics']['file'])
    assert file_digest(statistics) == result['statistics']['sha256'] == proof['cases'][variant_id]['statistics_sha256']
    saved = read_json(statistics)
    assert saved['task'] == variant_id and saved['evaluation'] == 'development'
    records = saved['records']
    assert len(records) == proof['cases'][variant_id]['records'] == result['sample_digests']['development']['count']
    assert variant['probe']['seeds'] == [0] and all(r['seed'] == 0 for r in records)
    return {'variant': variant, 'model': plan['model'], 'software': software,
            'hardware': hardware, 'execution': execution, 'recorded_hardware': recorded,
            'preparation_digest': result['preparation_digest'], 'sample_digests': result['sample_digests'],
            'records': records, 'score': result['metrics'][variant['metric']],
            'provenance': {'request': str(request_path), 'request_sha256': file_digest(request_path),
                           'verification_sha256': state['verification_sha256'],
                           'statistics_sha256': file_digest(statistics), 'variant': variant_id,
                           'worker_report_schema': schema, 'hardware_identity': hardware,
                           'recorded_hardware': recorded}}


def procedure(model):
    """Permit checkpoint changes within the pinned public CNN/ViT procedures."""
    result = deepcopy(model)
    for key in ['digest', 'state_digest']:
        result.pop(key)
    for key in ['checkpoint_sha256', 'checkpoint_epoch', 'checkpoint_index', 'training_epochs_from_batchnorm_counters']:
        result['info']['recipe'].pop(key, None)
    return result


def arrays(case):
    task, rows = case['variant'], sorted(case['records'], key=lambda r: r['id'])
    ids = [r['id'] for r in rows]
    assert ids and len(ids) == len(set(ids)) and all(r['seed'] == 0 for r in rows)
    classes = task['classes']
    if task['kind'] == 'classification':
        assert all(type(r[k]) is int and 0 <= r[k] < classes for r in rows for k in ['label', 'prediction'])
        values = np.array([r['label'] == r['prediction'] for r in rows], dtype=float)
        labels = [r['label'] for r in rows]
        fixed = labels
    else:
        assert task['kind'] == 'segmentation'
        assert all(len(r[k]) == classes and all(type(n) is int and n >= 0 for n in r[k])
                   for r in rows for k in ['intersection', 'union'])
        assert all(type(r['eligible_pixels']) is int and r['eligible_pixels'] >= 0 for r in rows)
        values = np.array([[r['intersection'], r['union']] for r in rows], dtype=float)
        assert np.all(values[:, 0] <= values[:, 1])
        assert all(all(n <= r['eligible_pixels'] for n in r['union']) for r in rows)
        labels, fixed = None, [r['eligible_pixels'] for r in rows]
    return ids, values, labels, fixed


def metric(values, selection, kind):
    if kind == 'classification':
        return float(values[selection].mean() * 100)
    intersection, union = values[selection].sum(axis=0)
    assert (union > 0).any(), 'No eligible classes'
    return float((intersection[union > 0] / union[union > 0]).mean() * 100)


def reference_procedure(task):
    task = deepcopy(task)
    calibration = task['calibration']
    assert task.get('reference_bank', {}).get('base_task') == calibration['base_task']
    epochs = calibration['reference_epochs']
    assert type(epochs) is int and epochs > 0 and task['probe']['updates'] % epochs == 0
    assert task['probe']['selection'] == 'best-development-score-then-earlier-epoch-then-candidate-order'
    assert task['probe']['scoring'] == 'each-epoch'
    updates_per_epoch = task['probe']['updates'] // epochs
    for key in ['id', 'name', 'digest']:
        task.pop(key)
    task['probe'].pop('updates')
    for key in ['budget_multiplier', 'reference_epochs']:
        calibration.pop(key)
    return task, updates_per_epoch


def compare(first, second, *, kind, replicates=2000, seed=0, reference_budgets=None):
    assert kind in ['checkpoint', 'reference-checkpoint', 'numerical', 'budget'] and type(replicates) is int and replicates > 0
    adequacy = None
    if kind == 'reference-checkpoint':
        assert isinstance(reference_budgets, (list, tuple)) and len(reference_budgets) == 2, 'Both reference endpoints need doubled-budget evidence'
        adequacy = [compare(base, doubled, kind='budget', replicates=replicates, seed=seed)
                    for base, doubled in zip([first, second], reference_budgets)]
        assert all(report['passed'] for report in adequacy), 'Inadequate fuller-reference training budget'
    else:
        assert reference_budgets is None
    assert first['preparation_digest'] == second['preparation_digest']
    assert first['sample_digests'] == second['sample_digests'] and first['software'] == second['software']
    assert first['hardware'] == second['hardware'] and first['execution'] == second['execution']
    tasks = [deepcopy(c['variant']) for c in [first, second]]
    if kind == 'budget':
        assert first['model'] == second['model']
        assert tasks[1]['probe']['updates'] == 2 * tasks[0]['probe']['updates']
        assert tasks[1]['calibration']['budget_multiplier'] == 2 * tasks[0]['calibration']['budget_multiplier']
        epochs = [task['calibration'].get('reference_epochs') for task in tasks]
        if any(value is not None for value in epochs):
            assert all(type(value) is int and value > 0 for value in epochs), 'Both reference epoch budgets are required'
            assert epochs[1] == 2 * epochs[0], 'Reference epochs must double with optimizer updates'
            for task, count in zip(tasks, epochs):
                assert task.get('reference_bank', {}).get('base_task') == task['calibration']['base_task'], 'Epoch metadata requires a declared fuller reference'
                assert task['probe']['updates'] % count == 0
                task['calibration'].pop('reference_epochs')
        for task in tasks:
            for key in ['id', 'name', 'digest']:
                task.pop(key)
            task['probe'].pop('updates')
            task['calibration'].pop('budget_multiplier')
    elif kind == 'numerical':
        assert {first['model']['numerics'], second['model']['numerics']} == {'fp32-v1', 'bf16-v1'}
        models = [{k: v for k, v in c['model'].items() if k not in ['digest', 'numerics']} for c in [first, second]]
        assert models[0] == models[1], 'Numerical comparison requires identical checkpoint and procedure'
    else:
        assert first['model']['state_digest'] != second['model']['state_digest']
        assert procedure(first['model']) == procedure(second['model']), 'Checkpoint procedure changed'
        if kind == 'reference-checkpoint':
            a, a_steps = reference_procedure(tasks[0])
            b, b_steps = reference_procedure(tasks[1])
            assert a_steps == b_steps, 'Reference updates per epoch differ'
            tasks = [a, b]
    assert tasks[0] == tasks[1], 'Task recipes differ'
    first_ids, a, labels, fixed = arrays(first)
    second_ids, b, _, second_fixed = arrays(second)
    assert first_ids == second_ids and fixed == second_fixed, 'Source IDs, labels or scoring eligibility differ'
    task = first['variant']
    selection = np.arange(len(first_ids))
    scores = [metric(v, selection, task['kind']) for v in [a, b]]
    assert all(np.isclose(value, case['score'], rtol=0, atol=1e-9) for value, case in zip(scores, [first, second]))
    groups = [np.flatnonzero(np.asarray(labels) == label) for label in sorted(set(labels))] if labels is not None and task['calibration']['base_task'].startswith('imagenet') else [selection]
    generator = np.random.default_rng(seed)
    bootstrap = []
    for _ in range(replicates):
        chosen = np.concatenate([generator.choice(group, size=len(group), replace=True) for group in groups])
        bootstrap.append(metric(b, chosen, task['kind']) - metric(a, chosen, task['kind']))
    interval = np.percentile(bootstrap, [2.5, 97.5]).tolist()
    difference = scores[1] - scores[0]
    result = {'kind': kind, 'difference_percentage_points': difference, 'paired_95_interval': interval,
              'scores': scores, 'replicates': replicates, 'bootstrap_seed': seed, 'source_images': len(selection),
              'probe_seeds': [0], 'resampling_unit': 'source-image-with-all-probe-seeds', 'stratified': len(groups) > 1,
              'first': first['provenance'], 'second': second['provenance'],
              'analysis_source_sha256': file_digest(__file__), 'numpy': np.__version__}
    if kind in ['numerical', 'budget']:
        tolerance = task['numerical_tolerance']['absolute_percentage_points'] if kind == 'numerical' else 0.5 if task['kind'] == 'classification' else 0.25
        result.update(tolerance_percentage_points=tolerance, passed=bool(abs(difference) <= tolerance or np.isclose(abs(difference), tolerance, rtol=0, atol=1e-12)))
    result['scope'] = 'Committed development-calibration comparison; no reserved acceptance, fuller-reference adequacy, latency or training-overhead claim.'
    if adequacy is not None:
        result['reference_budget_adequacy'] = adequacy
        result['scope'] = 'Fuller-reference checkpoint comparison with independently verified adequate base/doubled budgets at both endpoints. Only training duration and its administrative identity may differ; sample bank, readout, selection and all other settings match. Does not establish monitor agreement, reserved acceptance, latency or overhead.'
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['checkpoint', 'numerical', 'budget'])
    parser.add_argument('first_request', type=Path)
    parser.add_argument('first_variant')
    parser.add_argument('second_request', type=Path)
    parser.add_argument('second_variant')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=2000)
    args = parser.parse_args()
    assert not args.output.exists(), 'Choose a fresh comparison output'
    write_json(args.output, compare(load_case(args.first_request, args.first_variant),
                                   load_case(args.second_request, args.second_variant),
                                   kind=args.kind, replicates=args.replicates))
