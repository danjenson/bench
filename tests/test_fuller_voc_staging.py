from copy import deepcopy
from pathlib import Path
import runpy

import pytest


@pytest.fixture
def staging():
    return runpy.run_path(str(Path(__file__).parents[1] / 'scripts/stage_fuller_voc_pairs.py'))


def evidence(scores):
    epochs = [80, 160, 320, 640]
    cases = [{'id': str(e), 'probe': {'updates': 2 * e},
              'calibration': {'reference_epochs': e, 'learning_rate_multiplier': 3}} for e in epochs]
    checks = []
    for index in range(len(scores) - 1):
        difference = scores[index + 1] - scores[index]
        checks.append({'base_updates': 2 * epochs[index], 'doubled_updates': 2 * epochs[index + 1],
                       'base_score': scores[index], 'doubled_budget_score': scores[index + 1],
                       'difference': difference, 'tolerance': 0.25, 'passed': abs(difference) <= 0.25})
    report = {'status': 'complete', 'budget_series': checks,
              'results': {str(e): {'metrics': {'miou': score}} for e, score in zip(epochs, scores)}}
    proof = {'status': 'verified', 'budget_series': deepcopy(checks),
             'cases': {str(e): {'score': score} for e, score in zip(epochs, scores)}}
    return {'cases': cases}, report, proof


@pytest.mark.parametrize('scores,expected', [([10, 10.25], [80, 160]), ([10, 10.4, 10.6], [160, 320]),
                                            ([10, 10.4, 10.8, 10.9], [320, 640])])
def test_reuses_first_stable_pair_with_current_fitter_on_both_sides(staging, scores, expected):
    cases = staging['select_stable_pair'](*evidence(scores))
    assert [v['calibration']['reference_epochs'] for v in cases] == expected


def test_unresolved_terminal_budget_cannot_stage_a_reference_recipe(staging):
    with pytest.raises(ValueError, match='unresolved'):
        staging['select_stable_pair'](*evidence([10, 10.4, 10.8, 11.2]))


def test_missing_overlap_or_false_stopping_decision_is_rejected(staging):
    request, report, proof = evidence([10, 10.2])
    report['results'].pop('80')
    with pytest.raises(AssertionError, match='current fitter'):
        staging['select_stable_pair'](request, report, proof)
    with pytest.raises(AssertionError, match='first stable'):
        staging['select_stable_pair'](*evidence([10, 10.2, 10.3]))
    request, report, proof = evidence([10, 10.2])
    report['budget_series'][0]['passed'] = proof['budget_series'][0]['passed'] = False
    with pytest.raises(AssertionError):
        staging['select_stable_pair'](request, report, proof)


def test_unfinished_tails_create_no_staging_directory(staging, tmp_path):
    with pytest.raises(ValueError, match='unfinished'):
        staging['stage'](tmp_path)
    assert list(tmp_path.iterdir()) == []
