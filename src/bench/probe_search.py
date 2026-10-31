"""Deterministic search and budget decisions for full frozen probes.

The evaluator owns fitting, persistence and tensor cleanup. This module only
chooses which independent fits to request, so calibration evidence can replay
the exact same decisions without loading encoders or starting training.
"""

from itertools import product
import math


def ordered_axis(values, *, positive):
    if not isinstance(values, list) or not values:
        raise ValueError('Search axes must be nonempty ordered lists')
    if any(type(v) not in (int, float) or not math.isfinite(v) or (v <= 0 if positive else v < 0) for v in values):
        raise ValueError('Invalid search axis value')
    if values != sorted(set(values)):
        raise ValueError('Search axes must be strictly increasing')
    return values


def select_full_probe(recipe, evaluate, *, checkpoint=None):
    """Call evaluate(seed, epochs, learning_rate, weight_decay) for needed fits.

    Coordinate search starts with the declared Cartesian grid at the native
    epoch budget. It sweeps observed coordinates, extends an upper decay boundary
    by one ladder position, and tests all declared rates beyond a winning rate
    boundary. Testing the whole extension avoids stopping at a nonmonotone dip.
    Only the selected setting receives doubled-budget checks. The first stable
    doubling validates its smaller reporting budget, regardless of which score
    is higher. Seeds one and two use that same setting and budget without another
    search. The optional checkpoint callback identifies heads that remain needed.
    """
    plan, validation = recipe['search_plan'], recipe['budget_validation']
    if 'updates' in recipe or recipe.get('search_seed', 0) != 0:
        raise ValueError('Validated full probes require epoch budgets and search seed zero')
    if recipe.get('scoring', 'each-epoch') != 'each-epoch':
        raise ValueError('Validated full probes score development examples each epoch')
    if plan['method'] != 'coordinate':
        raise ValueError('Unsupported full-probe search method')
    rates = ordered_axis(plan['learning_rate_ladder'], positive=True)
    decays = ordered_axis(plan['weight_decay_ladder'], positive=False)
    initial_rates = ordered_axis(recipe['learning_rates'], positive=True)
    initial_decays = ordered_axis(recipe['weight_decays'], positive=False)
    if not set(initial_rates) <= set(rates) or not set(initial_decays) <= set(decays):
        raise ValueError('Initial grid must lie inside the declared ladders')
    epochs, seeds = recipe['epochs'], recipe['seeds']
    if type(epochs) is not int or epochs < 1:
        raise ValueError('Native epoch budget must be a positive integer')
    if not isinstance(seeds, list) or not seeds or seeds[0] != 0 or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('Declare distinct probe seeds starting with zero')
    multipliers = validation['multipliers']
    if not isinstance(multipliers, list) or len(multipliers) < 2 or multipliers[0] != 1 or any(type(m) is not int for m in multipliers) or any(b != 2 * a for a, b in zip(multipliers, multipliers[1:])):
        raise ValueError('Budget multipliers must start at one and double')
    tolerance = validation['absolute_percentage_points']
    if type(tolerance) not in (int, float) or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid budget tolerance')
    fits, search = {}, []

    def publish(phase, keep):
        if checkpoint is not None:
            checkpoint(phase, keep)

    def fit(seed, budget, rate, decay):
        key = seed, budget, rate, decay
        if key not in fits:
            value = dict(evaluate(*key))
            if (value['seed'], value['learning_rate'], value['weight_decay']) != (seed, rate, decay):
                raise ValueError('Evaluator returned another fit')
            if type(value['epoch']) is not int or not 1 <= value['epoch'] <= budget or not math.isfinite(value['development_score']):
                raise ValueError('Invalid selected epoch or development score')
            fits[key] = value
        return fits[key]

    def candidate(rate, decay):
        key = 0, epochs, rate, decay
        if key not in fits:
            value = fit(*key)
            search.append(value)
            publish('search', [best()[1]])

    def best():
        return min(enumerate(search), key=lambda item: (-item[1]['development_score'], item[1]['epoch'], item[0]))

    for rate, decay in product(initial_rates, initial_decays):
        candidate(rate, decay)
    rounds = []
    for _ in range(len(rates) * len(decays) + 1):
        before, winner = best()
        count_before = len(search)
        observed_rates = sorted({c['learning_rate'] for c in search})
        for rate in observed_rates:
            candidate(rate, winner['weight_decay'])
        _, winner = best()
        observed_decays = sorted({c['weight_decay'] for c in search})
        if winner['weight_decay'] == observed_decays[-1] and observed_decays[-1] < decays[-1]:
            next_decay = decays[decays.index(observed_decays[-1]) + 1]
            candidate(winner['learning_rate'], next_decay)
        _, winner = best()
        observed_rates = sorted({c['learning_rate'] for c in search})
        extension = ([r for r in rates if r < observed_rates[0]] if winner['learning_rate'] == observed_rates[0]
            else [r for r in rates if r > observed_rates[-1]] if winner['learning_rate'] == observed_rates[-1] else [])
        for rate in extension:
            candidate(rate, winner['weight_decay'])
        after, winner = best()
        for decay in sorted({c['weight_decay'] for c in search}):
            candidate(winner['learning_rate'], decay)
        after, winner = best()
        rounds.append({'winner_before': before, 'winner_after': after, 'measured_candidates': len(search)})
        if before == after and count_before == len(search):
            break
    else:
        raise RuntimeError('Finite coordinate search did not terminate')
    winner_index, winner = best()
    rate, decay = winner['learning_rate'], winner['weight_decay']
    selected, previous_epochs = winner, epochs
    checks = []
    for multiplier in multipliers[1:]:
        budget = epochs * multiplier
        measured = fit(0, budget, rate, decay)
        difference = measured['development_score'] - selected['development_score']
        passed = abs(difference) <= tolerance + 1e-12
        checks.append({'base_epochs': previous_epochs, 'doubled_epochs': budget,
            'scores': [selected['development_score'], measured['development_score']],
            'difference_percentage_points': difference, 'tolerance': tolerance, 'passed': passed})
        publish('validation', [selected if passed else measured])
        if passed:
            break
        selected, previous_epochs = measured, budget
    reporting = [selected]
    for seed in seeds[1:]:
        reporting.append(fit(seed, previous_epochs, rate, decay))
        publish('reporting', reporting)
    return {'candidates': search, 'search_winner_index': winner_index, 'search_winner': winner,
        'search_rounds': rounds, 'reporting_epochs': previous_epochs, 'seeds': reporting,
        'budget_checks': checks, 'budget_adequate': checks[-1]['passed'],
        'search_boundaries': {'lower_learning_rate': rate == rates[0],
            'upper_learning_rate': rate == rates[-1],
            'upper_weight_decay': len(decays) > 1 and decay == decays[-1]},
        'evaluated_fits': [{'seed': k[0], 'epochs': k[1], 'learning_rate': k[2], 'weight_decay': k[3],
                           'selected': value} for k, value in fits.items()]}


def fit_full_probe(recipe, *, directory, storage, train, header):
    """Persist individual fits and prune heads at each verified decision boundary."""
    from pathlib import Path

    from .storage import contained_path, digest, file_digest, read_json, remove, write_json

    directory = Path(directory)
    path = directory / 'search.json'
    committed = read_json(path) if path.exists() else {}
    state = committed.get('record', {})
    valid = (state.get('identity') == directory.name and state.get('schema') == 'validated-full-probe-1'
        and committed.get('sha256') == digest(state))
    if valid:
        selected = {record['selected']['head_file']: record['selected'] for record in state['fits']}
        valid = all(name in selected and contained_path(directory, name).exists()
            and file_digest(contained_path(directory, name)) == selected[name]['head_sha256'] for name in state['retained'])
    restarted = bool(committed) and not valid
    if not valid:
        state = {'identity': directory.name, 'schema': 'validated-full-probe-1', 'fits': [], 'retained': []}
    for head in directory.glob('validated-fit-*.pt'):
        if head.name not in state['retained']:
            remove(head, storage=storage)
    by_key = {tuple(record['key']): record['selected'] for record in state['fits']}
    dirty = False

    def evaluate(seed, epochs, rate, decay):
        nonlocal dirty
        key = seed, epochs, rate, decay
        if key not in by_key:
            destination = directory / f'validated-fit-{len(state["fits"])}.pt'
            value = train(seed, epochs, rate, decay, destination)
            assert value['head_file'] == destination.name and file_digest(destination) == value['head_sha256']
            by_key[key] = value
            state['fits'].append({'key': list(key), 'selected': value})
            dirty = True
        return by_key[key]

    def checkpoint(phase, keep):
        nonlocal dirty
        # Cached decisions replay before the next unfinished fit. They must not
        # prune the more advanced heads already retained by the journal.
        if not dirty:
            return
        for value in keep:
            assert file_digest(contained_path(directory, value['head_file'])) == value['head_sha256']
        state.update(phase=phase, retained=sorted({value['head_file'] for value in keep}))
        write_json(path, {'record': state, 'sha256': digest(state)}, storage=storage)
        for head in directory.glob('validated-fit-*.pt'):
            if head.name not in state['retained']:
                remove(head, storage=storage)
        dirty = False

    decision = select_full_probe(recipe, evaluate, checkpoint=checkpoint)
    assert not dirty
    selection = {**header, 'identity': directory.name, 'seeds': decision['seeds'], 'candidates': decision['candidates'],
        'search': {'method': recipe['search_plan']['method'], 'native_epochs': recipe['epochs'],
            'winner_index': decision['search_winner_index'], 'rounds': decision['search_rounds'],
            'boundaries': decision['search_boundaries']},
        'budget_validation': {'passed': decision['budget_adequate'], 'checks': decision['budget_checks'],
            'reporting_epochs': decision['reporting_epochs']},
        'validation_fits': [fit for fit in decision['evaluated_fits'] if fit['seed'] == 0 and fit['epochs'] != recipe['epochs']]}
    if restarted:
        selection['search_restarted'] = 'missing-or-corrupt-committed-search-state'
    for value in selection['seeds']:
        assert file_digest(contained_path(directory, value['head_file'])) == value['head_sha256']
    selection['digest'] = digest(selection)
    write_json(directory / 'selection.json', selection, storage=storage)
    keep = {'normalization.pt', *[value['head_file'] for value in selection['seeds']]}
    for head in directory.glob('*.pt'):
        if head.name not in keep:
            remove(head, storage=storage)
    return selection
