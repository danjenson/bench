from copy import deepcopy

import pytest

from bench.probe_search import select_full_probe


@pytest.fixture
def recipe():
    return {'epochs': 2, 'learning_rates': [.01, .1], 'weight_decays': [0., .1], 'seeds': [0, 1, 2],
        'search_plan': {'method': 'coordinate', 'learning_rate_ladder': [.001, .01, .1, 1.],
                        'weight_decay_ladder': [0., .1, 1., 10.]},
        'budget_validation': {'multipliers': [1, 2, 4, 8], 'absolute_percentage_points': .1}}


def result(seed, epochs, rate, decay, score, *, selected_epoch=None):
    return {'seed': seed, 'epoch': selected_epoch or epochs, 'learning_rate': rate,
        'weight_decay': decay, 'development_score': score}


def test_coordinate_search_rechecks_both_axes_and_validates_only_its_winner(recipe):
    calls = []
    scores = {(.01, 0.): 1., (.01, .1): 2., (.1, 0.): 3., (.1, .1): 4.,
              (.1, 1.): 5., (.01, 1.): 6., (.01, 10.): 7., (.001, 10.): 6., (.1, 10.): 6.}

    def fit(seed, epochs, rate, decay):
        calls.append((seed, epochs, rate, decay))
        return result(seed, epochs, rate, decay, scores.get((rate, decay), 0.) + {2: 0, 4: .3, 8: .35}[epochs])

    selected = select_full_probe(recipe, fit)
    assert selected['search_winner']['learning_rate'] == .01
    assert selected['search_winner']['weight_decay'] == 10.
    assert selected['search_boundaries']['upper_weight_decay']
    assert selected['reporting_epochs'] == 4 and selected['budget_adequate']
    assert [c['passed'] for c in selected['budget_checks']] == [False, True]
    assert len(calls) == len(set(calls))
    assert all((rate, decay) == (.01, 10.) for _, epochs, rate, decay in calls if epochs != 2)
    assert [(s, e) for s, e, _, _ in calls if s] == [(1, 4), (2, 4)]


def test_reporting_uses_validated_base_budget_without_selecting_budgets_by_score(recipe):
    def fit(seed, epochs, rate, decay):
        return result(seed, epochs, rate, decay, 10. if epochs == 2 else 9.95, selected_epoch=1)

    selected = select_full_probe(recipe, fit)
    assert selected['search_winner_index'] == 0
    assert selected['reporting_epochs'] == 2
    assert selected['seeds'][0]['development_score'] == 10.
    assert selected['budget_checks'][0]['passed']
    assert all(f['epochs'] <= 4 for f in selected['evaluated_fits'])


def test_unstable_terminal_budget_remains_a_failure(recipe):
    def fit(seed, epochs, rate, decay):
        return result(seed, epochs, rate, decay, float(epochs))

    selected = select_full_probe(recipe, fit)
    assert not selected['budget_adequate'] and selected['reporting_epochs'] == 16
    assert len(selected['budget_checks']) == 3 and not any(c['passed'] for c in selected['budget_checks'])


def test_rate_boundary_extension_does_not_stop_at_an_intermediate_dip(recipe):
    recipe['search_plan']['learning_rate_ladder'] = [.001, .003, .01, .1]
    scores = {.001: 12., .003: 9., .01: 10., .1: 8.}

    def fit(seed, epochs, rate, decay):
        return result(seed, epochs, rate, decay, scores[rate] - decay)

    selected = select_full_probe(recipe, fit)
    assert selected['search_winner']['learning_rate'] == .001
    assert any(c['learning_rate'] == .003 for c in selected['candidates'])
    assert selected['search_boundaries']['lower_learning_rate']


@pytest.mark.parametrize('field,value', [('seed', 3), ('epoch', 999), ('development_score', float('nan'))])
def test_wrong_or_invalid_evaluator_metadata_is_rejected(recipe, field, value):
    def fit(seed, epochs, rate, decay):
        return {**result(seed, epochs, rate, decay, 1.), field: value}

    with pytest.raises(ValueError):
        select_full_probe(recipe, fit)


@pytest.mark.parametrize('change', ['unordered-axis', 'outside-grid', 'bad-multipliers'])
def test_invalid_protocol_is_rejected_before_any_fit(recipe, change):
    invalid = deepcopy(recipe)
    if change == 'unordered-axis':
        invalid['search_plan']['learning_rate_ladder'].reverse()
    elif change == 'outside-grid':
        invalid['learning_rates'] = [.03]
    else:
        invalid['budget_validation']['multipliers'] = [1, 3]
    with pytest.raises(ValueError):
        select_full_probe(invalid, lambda *_: pytest.fail('An invalid protocol launched a fit'))
