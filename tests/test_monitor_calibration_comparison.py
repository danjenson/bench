from copy import deepcopy
from pathlib import Path
import runpy

import pytest

from bench.storage import digest, file_digest, write_json


@pytest.fixture
def comparison():
    return runpy.run_path(str(Path(__file__).parents[1] / 'scripts/compare_monitor_calibration.py'))


def case(kind='classification'):
    variant = {'id': 'candidate', 'name': 'candidate', 'digest': 'task-digest',
        'kind': kind, 'classes': 2, 'probe': {'updates': 10, 'seeds': [0]},
        'calibration': {'base_task': 'imagenet100-linear-v1' if kind == 'classification' else 'voc2012-linear-v1', 'budget_multiplier': 1},
        'numerical_tolerance': {'absolute_percentage_points': 0.1}, 'metric': 'top1' if kind == 'classification' else 'miou'}
    model = {'state_digest': 'state-1', 'numerics': 'fp32-v1',
             'info': {'name': 'public-model', 'recipe': {'checkpoint_epoch': 1, 'normalization': [0.5]}}}
    model['digest'] = digest(model)
    records = [{'id': str(i), 'seed': 0, 'label': label, 'prediction': prediction}
               for i, (label, prediction) in enumerate([(0, 0), (0, 0), (1, 0), (1, 1)])]
    return {'variant': variant, 'model': model, 'software': {'probe': 'same'},
            'hardware': {'gpu': 'same'}, 'execution': {'batch_size': 32},
            'preparation_digest': 'prepared', 'sample_digests': {'development': {'count': 4}},
            'records': records, 'score': 75.0, 'provenance': {'fixture': True}}


def test_checkpoint_comparison_stratifies_images_and_is_reproducible(comparison):
    first = case()
    second = deepcopy(first)
    second['model']['state_digest'] = 'state-2'
    second['model']['info']['recipe']['checkpoint_epoch'] = 2
    second['records'][0]['prediction'] = 1
    second['records'][2]['prediction'] = 1
    report = comparison['compare'](first, second, kind='checkpoint', replicates=100)
    assert report['difference_percentage_points'] == 0
    assert report['paired_95_interval'] == [-50, 50]
    assert report['stratified'] and report['source_images'] == 4
    assert report == comparison['compare'](first, second, kind='checkpoint', replicates=100)


def test_dense_bootstrap_resamples_images_and_recomputes_pooled_iou(comparison):
    first = case('segmentation')
    first['records'] = [
        {'id': '0', 'seed': 0, 'eligible_pixels': 1, 'intersection': [1, 0], 'union': [1, 0]},
        {'id': '1', 'seed': 0, 'eligible_pixels': 99, 'intersection': [0, 0], 'union': [99, 99]}]
    first['score'] = 0.5
    second = deepcopy(first)
    second['model']['state_digest'] = 'state-2'
    second['records'][1].update(intersection=[49, 50], union=[49, 50])
    second['score'] = 100
    report = comparison['compare'](first, second, kind='checkpoint', replicates=100)
    assert report['difference_percentage_points'] == 99.5
    assert report['paired_95_interval'] == [0, 100]
    assert report['source_images'] == 2 and not report['stratified']


def test_numerical_and_budget_comparisons_require_their_declared_changes(comparison):
    first = case()
    second = deepcopy(first)
    second['model']['numerics'] = 'bf16-v1'
    assert comparison['compare'](first, second, kind='numerical', replicates=10)['passed']
    second['model']['state_digest'] = 'different'
    with pytest.raises(AssertionError, match='identical checkpoint'):
        comparison['compare'](first, second, kind='numerical', replicates=10)
    second = deepcopy(first)
    second['variant']['probe']['updates'] *= 2
    second['variant']['calibration']['budget_multiplier'] *= 2
    second['variant'].update(id='doubled', name='doubled', digest='different')
    assert comparison['compare'](first, second, kind='budget', replicates=10)['passed']
    second['variant']['probe']['updates'] += 1
    with pytest.raises(AssertionError):
        comparison['compare'](first, second, kind='budget', replicates=10)


def test_fuller_budget_doubles_epochs_and_updates_without_hiding_other_recipe_changes(comparison):
    first = case()
    first['variant']['reference_bank'] = {'base_task': first['variant']['calibration']['base_task'], 'training': 'larger-bank'}
    first['variant']['calibration']['reference_epochs'] = 5
    second = deepcopy(first)
    second['variant']['probe']['updates'] *= 2
    second['variant']['calibration'].update(budget_multiplier=2, reference_epochs=10)
    assert comparison['compare'](first, second, kind='budget', replicates=10)['passed']
    for key, value in [('reference_epochs', 11), ('reference_epochs', None), ('budget_multiplier', 3)]:
        changed = deepcopy(second)
        changed['variant']['calibration'][key] = value
        with pytest.raises(AssertionError):
            comparison['compare'](first, changed, kind='budget', replicates=10)
    changed = deepcopy(second)
    changed['variant']['reference_bank']['training'] = 'another-bank'
    with pytest.raises(AssertionError, match='Task recipes differ'):
        comparison['compare'](first, changed, kind='budget', replicates=10)
    for value in [first, second]:
        value['variant'].pop('reference_bank')
    with pytest.raises(AssertionError, match='declared fuller reference'):
        comparison['compare'](first, second, kind='budget', replicates=10)


def test_adaptive_fuller_reference_comparison_requires_both_verified_adequate_budgets(comparison):
    first = case()
    first['variant'].update(reference_bank={'base_task': first['variant']['calibration']['base_task']})
    first['variant']['calibration']['reference_epochs'] = 5
    first['variant']['probe'].update(selection='best-development-score-then-earlier-epoch-then-candidate-order', scoring='each-epoch')
    def doubled(base):
        result = deepcopy(base)
        result['variant']['probe']['updates'] *= 2
        for key in ['reference_epochs', 'budget_multiplier']:
            result['variant']['calibration'][key] *= 2
        return result
    second = doubled(first)
    second['model']['state_digest'] = 'another-checkpoint'
    evidence = [doubled(first), doubled(second)]
    compare = comparison['compare']
    result = compare(first, second, kind='reference-checkpoint', reference_budgets=evidence, replicates=10)
    assert result['difference_percentage_points'] == 0
    assert all(r['passed'] for r in result['reference_budget_adequacy'])
    with pytest.raises(AssertionError, match='Both reference endpoints'):
        compare(first, second, kind='reference-checkpoint', replicates=10)
    invalid = deepcopy(evidence)
    invalid[1]['records'][0]['prediction'] = 1
    invalid[1]['score'] = 50.0
    with pytest.raises(AssertionError, match='Inadequate fuller-reference'):
        compare(first, second, kind='reference-checkpoint', reference_budgets=invalid, replicates=10)
    for field in ['reference_bank', 'learning_rate']:
        changed = deepcopy(second)
        if field == 'reference_bank':
            changed['variant']['reference_bank']['different'] = True
        else:
            changed['variant']['probe']['learning_rate'] = 0.5
        with pytest.raises(AssertionError, match='Task recipes differ'):
            compare(first, changed, kind='reference-checkpoint', reference_budgets=[evidence[0], doubled(changed)], replicates=10)
    with pytest.raises(AssertionError, match='Task recipes differ'):
        compare(first, second, kind='checkpoint', replicates=10)


@pytest.mark.parametrize('change', ['software', 'hardware', 'recipe', 'labels', 'duplicates', 'score'])
def test_comparison_rejects_mixed_or_invalid_evidence(comparison, change):
    first, second = case(), case()
    second['model']['state_digest'] = 'state-2'
    if change in ['software', 'hardware']:
        second[change] = {'different': True}
    elif change == 'recipe':
        second['model']['info']['recipe']['normalization'] = [0.4]
    elif change == 'labels':
        second['records'][0]['label'] = 1
    elif change == 'duplicates':
        second['records'][0]['id'] = '1'
    else:
        second['score'] = 100
    with pytest.raises(AssertionError):
        comparison['compare'](first, second, kind='checkpoint', replicates=10)


@pytest.mark.parametrize('schema', ['calibration', 'reference-tail', 'global-reference-tail'])
def test_loader_follows_committed_request_proof_and_statistics_hashes(comparison, tmp_path, schema):
    source = case('segmentation' if schema == 'reference-tail' else 'classification')
    source['hardware'].update(cuda_runtime='13.0')
    source['execution'].update(device='cuda', workers=8)
    package = tmp_path / 'sources/bench'
    if schema != 'calibration':
        source['variant']['reference_bank'] = {'base_task': source['variant']['calibration']['base_task']}
    if schema == 'reference-tail':
        source['records'] = [{'id': str(i), 'seed': 0, 'eligible_pixels': 3,
                              'intersection': [1, 1], 'union': [2, 2]} for i in range(2)]
        source['score'] = 50.0
        source['sample_digests']['development']['count'] = 2
        source['variant'].update(kind='segmentation', reference_bank={'base_task': source['variant']['calibration']['base_task']})
    if schema != 'calibration':
        package.mkdir(parents=True)
        for name in ['features.py', 'probes.py']:
            (package / name).write_text('# Pinned CPU fixture source\n')
        source['model']['extraction_implementation_sha256'] = file_digest(package / 'features.py')
        source['model']['digest'] = digest({k: v for k, v in source['model'].items() if k != 'digest'})
    output = tmp_path / 'output'
    request_path = tmp_path / 'request.json'
    request = {'output': str(output), 'expected_model': source['model'], 'variants': [source['variant']],
               'numerics': 'fp32-v1', 'worker': 'worker.py', 'pins': {'worker.py': 'worker-hash'},
               'gpu_uuid': 'same-gpu-uuid', 'cpu_threads': 2, 'batch_size': 32, 'workers': 8}
    if schema != 'calibration':
        request.pop('variants')
        request.update(cases=[source['variant']], prior_variant=source['variant'],
                       package_root=str(package), reference_training_count=8, revalidate_prior=True)
        request['pins'].update({str(p): file_digest(p) for p in package.glob('*.py')})
        if schema == 'global-reference-tail':
            request.update(base_task='imagenet-100', reference_recipe={'base_epochs': 100, 'learning_rate_multiplier': 10})
    write_json(request_path, request)
    plan = {'model': source['model'], 'variants': [source['variant']], 'numerics': 'fp32-v1',
            'software': source['software'], 'hardware': source['hardware'], 'execution': source['execution']}
    if schema != 'calibration':
        for key in ['variants', 'numerics', 'software', 'hardware']:
            plan.pop(key)
        plan.update(cases=request['cases'], request_sha256=file_digest(request_path), gpu='same',
                    gpu_uuid=request['gpu_uuid'], cuda_runtime='13.0', cpu_threads=2,
                    execution={'batch_size': 32, 'workers': 8})
    write_json(output / 'plan.json', plan)
    statistics = output / 'statistics.json'
    write_json(statistics, {'task': 'candidate', 'evaluation': 'development', 'records': source['records']})
    write_json(output / 'calibration.json', {'status': 'complete', 'encoder_state_unchanged': True, 'plan_sha256': file_digest(output / 'plan.json'),
        'results': {'candidate': {'statistics': {'file': statistics.name, 'sha256': file_digest(statistics)},
            'sample_digests': source['sample_digests'], 'preparation_digest': source['preparation_digest'],
            'metrics': {source['variant']['metric']: source['score']}}}})
    write_json(output / 'caller.json', {'request_sha256': file_digest(request_path), 'encoder_state_unchanged': True,
                                     'gpu_uuid': 'same-gpu-uuid', 'cpu_threads': 2})
    proof = {'status': 'verified', 'request_sha256': file_digest(request_path),
        'plan_sha256': file_digest(output / 'plan.json'), 'calibration_sha256': file_digest(output / 'calibration.json'),
        'caller_sha256': file_digest(output / 'caller.json'), 'verifier_sha256': 'worker-hash',
        'cases': {'candidate': {'statistics_sha256': file_digest(statistics), 'records': len(source['records'])}}}
    if schema != 'calibration':
        proof.pop('caller_sha256')
        (output / 'caller.json').unlink()
    write_json(output / 'verification.json', proof)
    state = {'status': 'complete', **{k: proof[k] for k in ['request_sha256', 'plan_sha256', 'calibration_sha256']},
             'verification_sha256': file_digest(output / 'verification.json')}
    write_json(tmp_path / 'request-state.json', state)
    loaded = comparison['load_case'](request_path, 'candidate')
    assert loaded['records'] == source['records']
    assert loaded['hardware'] == {'gpu_uuid': 'same-gpu-uuid', 'gpu': 'same', 'cuda_runtime': '13.0', 'cpu_threads': 2}
    if schema != 'calibration':
        ids, values, _, _ = comparison['arrays'](loaded)
        assert comparison['metric'](values, list(range(len(ids))), source['variant']['kind']) == source['score']
        (package / 'probes.py').write_text('# changed\n')
        with pytest.raises(AssertionError):
            comparison['load_case'](request_path, 'candidate')
        (package / 'probes.py').write_text('# Pinned CPU fixture source\n')
    statistics.write_text('{}')
    with pytest.raises(AssertionError):
        comparison['load_case'](request_path, 'candidate')
