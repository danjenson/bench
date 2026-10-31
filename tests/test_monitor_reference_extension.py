from copy import deepcopy
from pathlib import Path
import runpy
from contextlib import nullcontext

import pytest

from bench.protocols import resolve_names


pytestmark = pytest.mark.usefixtures('historical_monitor_recipes')


@pytest.fixture
def extension(monkeypatch):
    scripts = Path(__file__).parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    return runpy.run_path(str(scripts / 'extend_monitor_reference.py'))


def test_reference_tail_extends_only_the_winning_setting_and_budgets(extension):
    task = resolve_names(['pascal'])[0]
    original = deepcopy(task)
    baseline, cases = extension['declared_tail'](task, 1208)
    assert task == original
    assert baseline['calibration']['reference_epochs'] == 80 and baseline['probe']['updates'] == 6080
    assert [v['calibration']['reference_epochs'] for v in cases] == [160, 320, 640]
    assert [v['probe']['updates'] for v in cases] == [12160, 24320, 48640]
    assert len({v['digest'] for v in [baseline, *cases]}) == 4
    for v in cases:
        assert v['probe']['learning_rates'] == [0.003] and v['probe']['weight_decays'] == [0.01]
        assert v['probe']['seeds'] == [0] and v['probe']['scoring'] == 'each-epoch'
        assert v['probe']['selection'] == 'best-development-score-then-earlier-epoch-then-candidate-order'
        assert v['bank'] == baseline['bank'] and v['calibration']['base_task_digest'] == task['digest']
    cases[0]['probe']['learning_rates'][0] = 1
    assert baseline['probe']['learning_rates'] == cases[1]['probe']['learning_rates'] == [0.003]


def test_reference_budget_gate_preserves_the_declared_tolerance(extension):
    check = extension['budget_check']
    assert check(80, 64.5, 160, 64.75)['passed']
    assert check(80, 64.5, 160, 64.25)['passed']
    assert not check(80, 64.5, 160, 64.750001)['passed']
    with pytest.raises(AssertionError):
        check(80, 64.5, 240, 64.5)
    with pytest.raises(AssertionError):
        check(80, float('nan'), 160, 64.5)


def test_classification_tail_reuses_completed_budget_and_retains_canonical_variants(extension):
    task = resolve_names(['imagenet-100'])[0]
    setting = {'base_epochs': 100, 'learning_rate_multiplier': 10}
    baseline, cases = extension['declared_tail'](task, 124189, setting, 2)
    canonical = extension['declared_variants']([task], 'selected-fuller', {task['id']: 124189},
                                               {task['id']: setting})
    assert baseline == canonical[0] and cases[0] == canonical[1]
    assert baseline['calibration']['reference_epochs'] == 100
    assert [v['calibration']['reference_epochs'] for v in cases] == [200, 400]
    assert all(v['probe']['learning_rates'] == [0.03] for v in [baseline, *cases])
    assert baseline not in cases
    with pytest.raises(AssertionError, match='explicit selected settings'):
        extension['declared_tail'](task, 124189)
    for count in [0, 4, True]:
        with pytest.raises(AssertionError):
            extension['declared_tail'](task, 124189, setting, count)
    assert extension['budget_check'](100, 47.12, 200, 47.62, 0.5)['passed']
    assert not extension['budget_check'](100, 47.12, 200, 47.62001, 0.5)['passed']


def test_reference_overlap_checks_implementation_before_budget_comparisons(extension):
    check = extension['optimization_check']
    assert check(10, 10.1, 0.1)['passed']
    assert not check(10, 10.100001, 0.1)['passed']
    assert not check(10, 9.899999, 0.1)['passed']
    assert check(10, 10, 0)['difference'] == 0
    with pytest.raises(AssertionError):
        check(10, float('nan'), 0.1)


def stub_worker_runtime(extension, monkeypatch, tmp_path):
    """Replace hardware/data boundaries while exercising the worker orchestration."""
    run = extension['run']
    namespace = run.__globals__
    request_path = tmp_path / 'request.json'
    request_path.write_text('{}')
    model_yaml = tmp_path / 'model.yaml'
    model_yaml.write_text('{}')
    request = dict(output=str(tmp_path / 'result'), gpu_uuid='test-gpu', cpu_threads=1,
                   open_clip_source=None, model_yaml=str(model_yaml), expected_model={'digest': 'test'},
                   batch_size=1, workers=0, cases=[], data_root=str(tmp_path))
    prepared = {'source_root': str(tmp_path)}
    task = {'kind': 'segmentation', 'metric': 'miou', 'numerical_tolerance': {'absolute_percentage_points': 0.1}}
    monkeypatch.setitem(namespace, 'checked_request', lambda _: (request, task, {}, prepared, 64.5))
    monkeypatch.setattr(namespace['subprocess'], 'check_output',
                        lambda command, **kwargs: 'test-gpu\n' if '--query-gpu=uuid' in command else '')
    model = type('Model', (), {'cuda': lambda self: self})()
    monkeypatch.setitem(namespace, 'instantiate', lambda *args, **kwargs: model)
    monkeypatch.setitem(namespace, 'model_identity', lambda *args: request['expected_model'])
    monkeypatch.setitem(namespace, 'numerical_mode', lambda _: nullcontext())
    monkeypatch.setattr(namespace['torch'].cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(namespace['torch'].cuda, 'get_device_name', lambda: 'test-device')
    monkeypatch.setitem(namespace, 'load_prepared', lambda *args: (prepared, [{'id': 'test'}]))
    return run, namespace, request_path, request, prepared


def test_reference_worker_records_startup_before_extraction(extension, monkeypatch, tmp_path):
    run, namespace, request_path, request, prepared = stub_worker_runtime(extension, monkeypatch, tmp_path)
    def stop_before_fitting(*args, **kwargs):
        raise RuntimeError('extraction reached')
    monkeypatch.setitem(namespace, 'build_bank', stop_before_fitting)
    with pytest.raises(RuntimeError, match='extraction reached'):
        run(request_path)
    plan = namespace['read_json'](tmp_path / 'result/plan.json')
    report = namespace['read_json'](tmp_path / 'result/calibration.json')
    assert plan['prior_score'] == 64.5 and plan['gpu_uuid'] == 'test-gpu'
    assert report['status'] == 'failed' and 'extraction reached' in report['error']


def test_reference_recovery_uses_new_overlap_for_budget_and_skips_stable_tails(extension, monkeypatch, tmp_path):
    run, namespace, request_path, request, prepared = stub_worker_runtime(extension, monkeypatch, tmp_path)
    cases = [{'id': str(epochs), 'probe': {'updates': epochs},
              'calibration': {'reference_epochs': epochs}} for epochs in [80, 160, 320, 640]]
    request.update(cases=cases, prior_variant=cases[0], revalidate_prior=True)
    prepared.update(diagnostic_sample_ids=[], digest='prepared', splits={})
    closed = []
    bank = type('Bank', (), {'close': lambda self: closed.append(True)})()
    monkeypatch.setitem(namespace, 'build_bank', lambda *args, **kwargs: (bank, {'extracted_examples': 2, 'reused_examples': 0}))
    monkeypatch.setitem(namespace, 'feature_diagnostics', lambda *args: {})
    monkeypatch.setitem(namespace, 'probe_identity', lambda variant, *args: variant['id'])
    monkeypatch.setattr(namespace['torch'].cuda, 'max_memory_allocated', lambda: 0)
    fitted = []
    scores = {'80': 64.55, '160': 64.77}
    def fit(*args, **kwargs):
        variant = kwargs['task']
        fitted.append(variant['id'])
        kwargs['directory'].mkdir(parents=True)
        return {'candidates': [{'development_score': scores[variant['id']]}]}
    monkeypatch.setitem(namespace, 'fit_probe', fit)
    monkeypatch.setitem(namespace, 'evaluate_selection',
                        lambda selection, directory, bank, rows, root, variant, execution:
                        ({'miou': scores[variant['id']]}, [], []))
    run(request_path)
    report = namespace['read_json'](tmp_path / 'result/calibration.json')
    assert fitted == ['80', '160'] and closed == [True]
    assert report['status'] == 'complete' and report['optimization_parity']['passed']
    check, = report['budget_series']
    assert check['base_score'] == 64.55 and check['doubled_budget_score'] == 64.77
    assert check['passed']  # Comparing the old 64.5 score would incorrectly continue.
    assert set(report['skipped']) == {'320', '640'}
    assert report['retained_temporary_bytes'] == 0 and not (tmp_path / 'result/.work').exists()


def test_global_reference_verifier_reconstructs_labels_and_conditional_skip(extension, monkeypatch, tmp_path):
    verify = extension['verify']
    ns = verify.__globals__
    path = tmp_path / 'request.json'
    path.write_text('{}')
    output = tmp_path / 'output'
    task = {'kind': 'classification', 'metric': 'top1', 'classes': 2,
            'probe': {'batch_size': 2}, 'numerical_tolerance': {'absolute_percentage_points': 0.1}}
    baseline = {'probe': {'updates': 100}}
    cases = [{'id': str(e), 'probe': {'updates': e, 'learning_rates': [0.03], 'weight_decays': [0.01]},
              'calibration': {'reference_epochs': e}} for e in [200, 400]]
    q = {'output': str(output), 'data_root': str(tmp_path), 'cases': cases, 'prior_variant': baseline,
         'reference_training_count': 2, 'feature_width': 2, 'expected_model': {'digest': 'test'}, 'gpu_uuid': 'test-gpu'}
    prepared = {'source_root': str(tmp_path), 'digest': 'prepared', 'splits': {}}
    rows = [{'id': str(i), 'target': i % 2} for i in range(4)]
    records = [{'id': r['id'], 'seed': 0, 'label': r['target'], 'prediction': p}
               for r, p in zip(rows, [0, 1, 1, 1])]
    monkeypatch.setitem(ns, 'checked_request', lambda _: (q, task, {'id': 'reference'}, prepared, 75.0))
    monkeypatch.setitem(ns, 'load_prepared', lambda *a, **k: (prepared, rows))
    write, sha = ns['write_json'], ns['file_digest']
    plan = {'request_sha256': sha(path), 'cases': cases, 'model': q['expected_model'],
            'reference_preparation': {'id': 'reference'}, 'prior_score': 75.0,
            'script_sha256': sha(ns['__file__']), 'gpu_uuid': 'test-gpu'}
    write(output / 'plan.json', plan)
    statistics = output / 'statistics.json'
    write(statistics, {'task': '200', 'evaluation': 'development', 'records': records})
    candidate = {'seed': 0, 'learning_rate': 0.03, 'weight_decay': 0.01, 'epoch': 150, 'updates': 150,
                 'development_score': 75.0, 'development_loss': 1.0, 'training_loss': 1.0,
                 'probe_parameters': 6, 'head_sha256': 'a' * 64}
    check = extension['budget_check'](100, 75.0, 200, 75.0, 0.5)
    saved = {'preparation_digest': 'prepared', 'sample_digests': {}, 'metrics': {'top1': 75.0},
             'selection': [candidate], 'candidate_scores': [candidate],
             'seed_results': [{'score': 75.0, 'seed': 0, 'sample_count': 4, 'eligible_count': 4}],
             'statistics': {'file': 'statistics.json', 'sha256': sha(statistics)}}
    report = {'status': 'complete', 'plan_sha256': sha(output / 'plan.json'), 'encoder_state_unchanged': True,
              'retained_temporary_bytes': 0, 'results': {'200': saved}, 'skipped': {'400': check},
              'budget_series': [check], 'extraction': {'extracted_examples': 6, 'reused_examples': 0}}
    write(output / 'calibration.json', report)
    proof = verify(path)
    assert proof['verified_records'] == 4 and proof['cases']['200']['reconstructed_score'] == 75.0
    assert proof['budget_series'] == [check] and proof['skipped'] == {'400': check}
    records[0]['label'] = 1
    write(statistics, {'task': '200', 'evaluation': 'development', 'records': records})
    saved['statistics']['sha256'] = sha(statistics)
    write(output / 'calibration.json', report)
    with pytest.raises(AssertionError):
        verify(path)
