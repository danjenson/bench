"""Verify global calibration records, with explicit checks across GPU hardware."""

import argparse
from copy import deepcopy
from itertools import product
import math
from pathlib import Path

from bench.prepare import load_prepared
from bench.protocols import resolve_names
from bench.storage import contained_path, digest, file_digest, read_json, write_json


def compare_candidates(actual, baseline, *, hardware_pair=None):
    """Keep scores/selection exact; account for FP32 loss and head-byte changes.

    Hash equality is a same-execution claim, not a cross-device score check.
    The loss diagnostic bound is eight FP32 epsilons at unit/reference scale;
    this does not replace or relax the registered task score tolerances.
    """
    assert len(actual) == len(baseline) > 0, 'Baseline candidate coverage changed'
    if hardware_pair is None:
        assert actual == baseline, 'Same-hardware candidate records changed'
    else:
        current, previous = hardware_pair
        assert current.get('gpu') and previous.get('gpu') and current['gpu'] != previous['gpu'], 'A GPU transition must be explicit'
    largest_loss_difference, equal_heads = 0.0, True
    diagnostic_fields = {'training_loss', 'development_loss', 'head_sha256'}
    for current, previous in zip(actual, baseline):
        assert current.keys() == previous.keys(), 'Candidate record fields changed'
        assert {k: v for k, v in current.items() if k not in diagnostic_fields} == {k: v for k, v in previous.items() if k not in diagnostic_fields}, 'Candidate score or selection metadata changed'
        for field in ('training_loss', 'development_loss'):
            a, b = current[field], previous[field]
            assert math.isfinite(a) and math.isfinite(b), 'Non-finite candidate loss'
            difference = abs(a - b)
            assert difference <= 8 * 2**-23 * max(1.0, abs(b)), 'Candidate loss differs beyond FP32 roundoff bound'
            largest_loss_difference = max(largest_loss_difference, difference)
        for item in (current, previous):
            checksum = item['head_sha256']
            assert isinstance(checksum, str) and len(checksum) == 64 and all(c in '0123456789abcdef' for c in checksum), 'Invalid saved head hash'
        equal_heads &= current['head_sha256'] == previous['head_sha256']
    return {'candidates': len(actual), 'scores_and_selection_metadata_identical': True,
            'head_hashes_identical': equal_heads, 'largest_absolute_loss_difference': largest_loss_difference,
            'loss_relative_bound': 8 * 2**-23, 'hardware_transition': hardware_pair is not None}


def verify(directory, baseline_directory, previous_directory=None, *, names=('clevr-count', 'clevr-distance'), allow_hardware_change=False):
    plan = read_json(directory / 'plan.json')
    report = read_json(directory / 'calibration.json')
    baseline = read_json(baseline_directory / 'results.json')
    assert report['status'] == baseline['status'] == 'complete'
    assert report['plan_sha256'] == file_digest(directory / 'plan.json')
    assert report['retained_temporary_bytes'] == 0 and not (directory / '.work').exists()
    from c3_reference import relocated_model
    assert plan['model'] == relocated_model(baseline['model'])
    assert plan['numerics'] == 'fp32-v1'
    hardware_pair = (plan['hardware'], baseline['hardware']) if allow_hardware_change else None
    if hardware_pair:
        assert hardware_pair[0].get('gpu') and hardware_pair[1].get('gpu') and hardware_pair[0]['gpu'] != hardware_pair[1]['gpu'], 'No GPU transition to verify'
    stage = plan['stage']
    assert stage in ('search-boundary', 'search-extension', 'search-weight-decay', 'search-grid')
    weight_search = stage in ('search-weight-decay', 'search-grid')
    rates, budgets = ((1, 3, 10, 30, 100), (2, 4)) if stage == 'search-extension' else ((1, 3, 10), (1, 2))
    if weight_search:
        rates = (1,)
    tasks = {task['id']: task for task in resolve_names(list(names))}
    assert tasks and all(task['name'] in ('imagenet-1k', 'clevr-count', 'clevr-distance') for task in tasks.values())
    assert plan['gates'] == {'budget_absolute_percentage_points': {}, 'scale_absolute_percentage_points': {key: task['numerical_tolerance']['absolute_percentage_points'] for key, task in tasks.items()}}
    assert len(plan['variants']) == len(report['results']) == len(tasks) * len(rates) * len(budgets)
    assert {variant['id'] for variant in plan['variants']} == set(report['results'])
    assert set(report['comparisons']) == set(tasks)
    previous, previous_variants = None, {}
    if stage == 'search-extension':
        assert previous_directory is not None
        previous_plan = read_json(previous_directory / 'plan.json')
        previous = read_json(previous_directory / 'calibration.json')
        assert previous_plan['stage'] == 'search-boundary' and previous['status'] == 'complete'
        assert previous['plan_sha256'] == file_digest(previous_directory / 'plan.json')
        assert previous_plan['model'] == plan['model']
        previous_variants = {variant['id']: variant for variant in previous_plan['variants']}
    evidence = {'status': 'verified', 'stage': stage, 'verifier_sha256': file_digest(__file__), 'plan_sha256': file_digest(directory / 'plan.json'), 'calibration_sha256': file_digest(directory / 'calibration.json'), 'baseline_sha256': file_digest(baseline_directory / 'results.json'), 'variants': {}, 'comparisons': {}, 'scope': 'Retained development metadata and records; no source-image reread or model rerun, and not release acceptance.'}
    if hardware_pair:
        evidence['hardware_comparison'] = {'current': hardware_pair[0], 'baseline': hardware_pair[1], 'scope': 'Exact overlapping candidate scores and selection metadata, bounded FP32 loss differences and retained hash provenance; no claim of identical head bytes across GPUs.'}
    if previous is not None:
        evidence['previous_calibration_sha256'] = file_digest(previous_directory / 'calibration.json')
    overlaps = 0
    for task_id, task in tasks.items():
        prepared, training = load_prepared(Path(plan['data_root']), task, 'train', verify=False)
        _, development = load_prepared(Path(plan['data_root']), task, 'development', verify=False)
        reference = baseline['tasks'][task_id]
        assert reference['task_digest'] == task['digest']
        assert reference['preparation_digest'] == prepared['digest']
        scores = {}
        for variant in (v for v in plan['variants'] if v['calibration']['base_task'] == task_id):
            calibration = variant['calibration']
            budget, rate = calibration['budget_multiplier'], calibration['learning_rate_multiplier']
            assert budget in budgets and rate in rates and (budget, rate) not in scores
            base_rates = task['probe']['learning_rates'] if weight_search else [max(task['probe']['learning_rates'])]
            expected_calibration = {'base_task': task_id, 'base_task_digest': task['digest'], 'feature_scale_before_standardization': 1.0, 'scale_arithmetic': 'float32', 'budget_multiplier': budget, 'learning_rate_multiplier': rate, 'learning_rate_reference': base_rates}
            if weight_search:
                assert len(task['probe']['weight_decays']) > 1 and max(task['probe']['weight_decays']) > 0
                expected_calibration.update(weight_decay_reference=task['probe']['weight_decays'], weight_decay_multiplier=10)
            if stage == 'search-grid':
                expected_calibration['learning_rate_extension_multipliers'] = [3, 10]
            assert calibration == expected_calibration
            expected = deepcopy(task)
            expected['probe'].pop('search_plan', None)
            expected['probe'].pop('budget_validation', None)
            expected['probe'].update(epochs=task['probe']['epochs'] * budget, learning_rates=[lr * rate for lr in base_rates], seeds=[0])
            identity = f"{task_id}-calibration-e{expected['probe']['epochs']}-s1-lrm{rate}"
            if weight_search:
                expected['probe']['weight_decays'] += [max(task['probe']['weight_decays']) * 10]
                identity += '-wdm10'
            if stage == 'search-grid':
                expected['probe']['learning_rates'] += [max(base_rates) * extension for extension in (3, 10)]
                identity += '-joint'
            expected.update(id=identity, name=identity, release_status='calibration-only', calibration=calibration)
            expected['digest'] = digest({key: value for key, value in expected.items() if key != 'digest'})
            assert variant == expected
            result = report['results'][identity]
            assert result['preparation_digest'] == prepared['digest']
            assert result['sample_digests'] == prepared['splits']
            assert {split: {key: value for key, value in values.items() if key != 'file'} for split, values in result['sample_digests'].items()} == reference['preparation']['splits']
            statistics = contained_path(directory, result['statistics']['file'])
            assert file_digest(statistics) == result['statistics']['sha256']
            saved = read_json(statistics)
            assert saved['task'] == identity and saved['evaluation'] == 'development'
            records = saved['records']
            assert [row['id'] for row in records] == [row['id'] for row in development]
            assert len({row['id'] for row in records}) == len(development)
            assert [row['label'] for row in records] == [row['target'] for row in development]
            assert {row['seed'] for row in records} == {0}
            assert all(isinstance(row['prediction'], int) and 0 <= row['prediction'] < task['classes'] for row in records)
            score = 100 * sum(row['prediction'] == row['label'] for row in records) / len(records)
            assert len(result['selection']) == len(result['seed_results']) == 1
            selected, measured = result['selection'][0], result['seed_results'][0]
            assert score == result['metrics'][task['metric']] == measured['score'] == selected['development_score']
            assert measured['seed'] == 0 and measured['eligible_count'] == measured['sample_count'] == len(records)
            candidates = result['candidate_scores']
            assert len(candidates) == len(expected['probe']['learning_rates']) * len(expected['probe']['weight_decays'])
            assert [(candidate['learning_rate'], candidate['weight_decay'], candidate['seed']) for candidate in candidates] == [(lr, wd, 0) for lr, wd in product(expected['probe']['learning_rates'], expected['probe']['weight_decays'])]
            for candidate in candidates:
                assert 1 <= candidate['epoch'] <= expected['probe']['epochs']
                assert candidate['updates'] == candidate['epoch'] * math.ceil(len(training) / task['probe']['batch_size'])
                assert 0 <= candidate['development_score'] <= 100
                assert all(math.isfinite(candidate[key]) for key in ('development_loss', 'training_loss'))
            winner = min(range(len(candidates)), key=lambda index: (-candidates[index]['development_score'], candidates[index]['epoch'], index))
            assert selected == candidates[winner]
            check = {'score': score, 'records': len(records), 'epochs': expected['probe']['epochs'], 'learning_rates': expected['probe']['learning_rates'], 'weight_decays': expected['probe']['weight_decays'], 'selected_learning_rate': selected['learning_rate'], 'selected_weight_decay': selected['weight_decay']}
            if budget == rate == 1:
                if weight_search:
                    original_candidates = [c for c in candidates if c['weight_decay'] in task['probe']['weight_decays'] and c['learning_rate'] in task['probe']['learning_rates']]
                    check['baseline_candidate_comparison'] = compare_candidates(original_candidates, reference['selection']['candidates'], hardware_pair=hardware_pair)
                    check['baseline_candidates_and_heads_identical'] = original_candidates == reference['selection']['candidates']
                else:
                    check['baseline_candidate_comparison'] = compare_candidates(candidates, [c for c in reference['selection']['candidates'] if c['learning_rate'] == expected['probe']['learning_rates'][0]], hardware_pair=hardware_pair)
                if selected['weight_decay'] in task['probe']['weight_decays'] and selected['learning_rate'] in task['probe']['learning_rates']:
                    reference_path = contained_path(baseline_directory, reference['statistics']['file'])
                    assert file_digest(reference_path) == reference['statistics']['sha256']
                    assert records == [row for row in read_json(reference_path)['records'] if row['seed'] == 0]
                    check['baseline_selected_comparison'] = compare_candidates([selected], [reference['selection']['seeds'][0]], hardware_pair=hardware_pair)
                    check['baseline_seed_zero_records_identical'] = True
                    check['baseline_seed_zero_records_and_heads_identical'] = selected == reference['selection']['seeds'][0]
            if stage == 'search-extension' and budget == 2 and rate in (1, 3, 10):
                assert variant == previous_variants[identity]
                prior = previous['results'][identity]
                for key in ('metrics', 'seed_results', 'selection', 'candidate_scores', 'preparation_digest', 'sample_digests', 'statistics'):
                    assert result[key] == prior[key]
                prior_statistics = contained_path(previous_directory, prior['statistics']['file'])
                assert file_digest(prior_statistics) == prior['statistics']['sha256']
                assert saved == read_json(prior_statistics)
                check['previous_records_and_heads_identical'] = True
                overlaps += 1
            evidence['variants'][identity] = check
            scores[budget, rate] = score
        assert set(scores) == {(budget, rate) for budget in budgets for rate in rates}
        comparison = report['comparisons'][task_id]
        assert comparison['extraction']['extracted_examples'] == len(training) + len(development)
        assert comparison['extraction']['reused_examples'] == 0
        checks = {str(rate): {'base_score': scores[budgets[0], rate], 'doubled_budget_score': scores[budgets[1], rate], 'doubled_budget_difference': scores[budgets[1], rate] - scores[budgets[0], rate]} for rate in rates}
        assert comparison['learning_rate_checks'] == checks
        evidence['comparisons'][task_id] = {'budget_checks': checks}
        if weight_search:
            selections = {}
            for variant in (v for v in plan['variants'] if v['calibration']['base_task'] == task_id):
                selected = report['results'][variant['id']]['selection'][0]
                selections[str(variant['probe']['epochs'])] = {'learning_rate': selected['learning_rate'], 'weight_decay': selected['weight_decay'], 'upper_learning_rate_boundary': selected['learning_rate'] == max(variant['probe']['learning_rates']), 'upper_weight_decay_boundary': selected['weight_decay'] == max(variant['probe']['weight_decays'])}
            evidence['comparisons'][task_id]['selected_settings_by_epochs'] = selections
        else:
            evidence['comparisons'][task_id]['best_rates_by_epochs'] = {}
            for budget in budgets:
                best = max(scores[budget, rate] for rate in rates)
                winners = [rate for rate in rates if scores[budget, rate] == best]
                evidence['comparisons'][task_id]['best_rates_by_epochs'][str(task['probe']['epochs'] * budget)] = {'score': best, 'learning_rates': [max(task['probe']['learning_rates']) * rate for rate in winners], 'upper_boundary_in_best': max(rates) in winners}
    assert overlaps == (len(tasks) * 3 if stage == 'search-extension' else 0)
    evidence['verified_records'] = sum(value['records'] for value in evidence['variants'].values())
    return evidence


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--previous-calibration', type=Path)
    parser.add_argument('--allow-hardware-change', action='store_true', help='Require identical overlapping scores/selection with bounded FP32 loss drift, preserving changed head hashes')
    parser.add_argument('--names', nargs='+', choices=['imagenet-1k', 'clevr-count', 'clevr-distance'], default=['clevr-count', 'clevr-distance'])
    args = parser.parse_args()
    evidence = verify(args.calibration, args.baseline, args.previous_calibration, names=args.names, allow_hardware_change=args.allow_hardware_change)
    write_json(args.calibration / 'verification.json', evidence)
    print(f"Verified {len(evidence['variants'])} variants and {evidence['verified_records']} prediction records, comparisons, overlapping heads and cleanup")
