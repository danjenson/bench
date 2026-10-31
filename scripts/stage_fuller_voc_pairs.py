"""Stage missing VOC calibration references after both late budgets are verified.

Uses the first stable base/doubled pair from each already verified tail and
reuses those late fits. This command prepares requests; it does not enqueue or
run GPU work. Failed or unfinished tails cannot create a staging directory.
"""
from copy import deepcopy
import argparse
from datetime import datetime, timezone
import json
import hashlib
import math
from pathlib import Path
import runpy
import shutil
import sys
import types


def select_stable_pair(request, report, proof):
    assert report['status'] == 'complete' and proof['status'] == 'verified'
    checks = report['budget_series']
    assert checks and checks == proof['budget_series']
    previous = None
    for index, check in enumerate(checks):
        assert check['doubled_updates'] == 2 * check['base_updates']
        assert check['tolerance'] == 0.25
        assert all(math.isfinite(check[key]) for key in ['base_score', 'doubled_budget_score', 'difference'])
        assert check['difference'] == check['doubled_budget_score'] - check['base_score']
        assert check['passed'] is (abs(check['difference']) <= 0.25)
        if previous is not None:
            assert previous['doubled_updates'] == check['base_updates']
            assert previous['doubled_budget_score'] == check['base_score']
        previous = check
        if not check['passed']:
            continue
        assert index == len(checks) - 1, 'The frozen tail stops after its first stable doubling'
        cases = []
        for key, score_key in [('base_updates', 'base_score'), ('doubled_updates', 'doubled_budget_score')]:
            candidates = [v for v in request['cases'] if v['probe']['updates'] == check[key]]
            assert len(candidates) == 1
            variant = candidates[0]
            assert variant['id'] in report['results'] and variant['id'] in proof['cases'], 'Use the current fitter on both sides'
            assert report['results'][variant['id']]['metrics']['miou'] == proof['cases'][variant['id']]['score'] == check[score_key]
            cases.append(variant)
        assert cases[1]['calibration']['reference_epochs'] == 2 * cases[0]['calibration']['reference_epochs']
        assert cases[0]['calibration']['learning_rate_multiplier'] == cases[1]['calibration']['learning_rate_multiplier']
        return cases
    raise ValueError('Fuller VOC reference remains unresolved at the last measured budget')


def stage(run_root, baseline_requests=None):
    run = Path(run_root).resolve()
    repo = Path(__file__).resolve().parents[1]
    root = run / 'monitor-fuller-voc-pairs-20261027'
    assert not root.exists(), 'Inspect any existing requests before restaging'
    paths = baseline_requests or {'cnn': run / 'monitor-cnn-reference-tail-20261027/request.json',
                                  'vit': run / 'monitor-vit-reference-tail-retry-20261027/request-v2.json'}
    assert set(paths) == {'cnn', 'vit'}
    paths = {key: Path(value).resolve() for key, value in paths.items()}
    # Check terminal states before importing models or creating new output.
    for family, path in paths.items():
        state_path = path.with_name(path.stem + '-state.json')
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        if state is None or state['status'] != 'complete':
            raise ValueError(f'{family} fuller VOC tail is unfinished; no staging directory created')
        assert state['request_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    original_requests = {family: json.loads(path.read_text()) for family, path in paths.items()}
    packages = {q['package_root'] for q in original_requests.values()}
    assert len(packages) == 1, 'Use the same qualified fitter for the two families'
    package = Path(packages.pop())
    for q in original_requests.values():
        files = list(package.rglob('*.py'))
        assert files and all(hashlib.sha256(p.read_bytes()).hexdigest() == q['pins'][str(p)] for p in files)
    sys.path.insert(0, str(package.parent))
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from bench import Execution
    from bench.features import model_identity
    from bench.protocols import resolve_names
    from bench.storage import file_digest, read_json, write_json

    compare = runpy.run_path(str(repo / 'scripts/compare_monitor_calibration.py'))
    baselines, loaded, pins = {}, {}, {}
    for family, path in paths.items():
        q = read_json(path)
        state_path = path.with_name(path.stem + '-state.json')
        state = read_json(state_path)
        output = Path(q['output'])
        assert state['request_sha256'] == file_digest(path)
        for name, key in [('plan.json', 'plan_sha256'), ('calibration.json', 'calibration_sha256'), ('verification.json', 'verification_sha256')]:
            assert file_digest(output / name) == state[key]
        report, proof = read_json(output / 'calibration.json'), read_json(output / 'verification.json')
        assert proof['optimization_parity']['passed'], 'Qualify the overlapping larger-bank fit first'
        cases = select_stable_pair(q, report, proof)
        loaded[family] = [compare['load_case'](path, v['id']) for v in cases]
        baselines[family] = {'request': str(path), 'request_sha256': file_digest(path),
            'base': cases[0], 'doubled': cases[1], 'budget': report['budget_series'][-1]}
        for p in [path, state_path, output / 'plan.json', output / 'calibration.json', output / 'verification.json']:
            pins[str(p)] = file_digest(p)
    for key in ['software', 'hardware', 'execution']:
        assert loaded['cnn'][0][key] == loaded['vit'][0][key]
    root.mkdir()
    base = run / 'monitor-candidate-screening-20261027'
    for source, name in [(repo / 'scripts/monitor_calibration.py', 'monitor_calibration.py'),
                         (repo / 'examples/calibrate_monitor_probes.py', 'calibrate_monitor_probes.py'),
                         (repo / 'scripts/compare_monitor_calibration.py', 'compare.py'),
                         (repo / 'examples/monitoring_pair_gates.py', 'pair_gates.py'),
                         (base / 'run.py', 'run.py'), (base / 'gate.py', 'gate.py'),
                         (base / 'public_vit_adapter.py', 'public_vit_adapter.py'),
                         (base / 'cifar10_cnn_adapter.py', 'cifar10_cnn_adapter.py')]:
        shutil.copy2(source, root / name)
    sys.path.insert(0, str(root))
    torch.set_num_threads(2)
    task = resolve_names(['pascal'])[0]
    policy = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'baselines': baselines,
        'reference_recipe': {family: {task['id']: {'base_epochs': row['base']['calibration']['reference_epochs'],
            'learning_rate_multiplier': row['base']['calibration']['learning_rate_multiplier']}}
            for family, row in baselines.items()},
        'scope': 'Only eight missing VOC fuller calibration checkpoints, two selected-setting fits each. Reuse four verified late fits from the first stable doubling in each family; both sides use the qualified fitter. Preserve all remaining budget failures, original checkpoint/pair declarations and interval publication. Frozen encoders, no reserved/final images, no cadence or v1 claim.'}
    write_json(root / 'reference-policy.json', policy)
    for p in list(root.glob('*.py')) + [Path(__file__).resolve(), root / 'reference-policy.json']:
        pins[str(p)] = file_digest(p)
    for p in package.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            pins[str(p)] = file_digest(p)
    worker = runpy.run_path(str(root / 'monitor_calibration.py'))
    requests = {}
    for family, indices in [('cnn', [1, 24, 25, 49]), ('vit', [1, 9, 10, 19])]:
        for index in indices:
            label = f'{family}-{index}'
            prior_path = base / 'requests' / label / 'request.json'
            prior = read_json(prior_path)
            if family == 'vit':
                namespace = types.ModuleType('open_clip')
                namespace.__path__ = [str(Path(prior['open_clip_source']) / 'src/open_clip')]
                sys.modules['open_clip'] = namespace
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(0)
                model = instantiate(OmegaConf.load(prior['model_yaml']), _convert_='all')
            assert model_identity(model, Execution(device='cpu', numerics='fp32-v1')) == prior['expected_model']
            del model
            recipe = policy['reference_recipe'][family]
            variants = worker['declared_variants']([task], 'selected-fuller', {task['id']: 1208}, recipe)
            assert variants == [baselines[family]['base'], baselines[family]['doubled']]
            q = deepcopy(prior)
            q.pop('parity_priors', None)
            q.pop('expected_parity_cases', None)
            q.update(label=label + '-fuller-voc', created_at_utc=datetime.now(timezone.utc).isoformat(),
                pins={**prior['pins'], **pins, str(prior_path): file_digest(prior_path)}, package_root=str(package),
                worker=str(root / 'monitor_calibration.py'), measure_script=str(root / 'calibrate_monitor_probes.py'),
                pinned_caller=str(root / 'run.py'), names=['pascal'], stage='selected-fuller', reference_recipe=recipe,
                reference_counts={task['id']: 1208}, variants=variants, output=str(root / 'results' / label),
                cpu_threads=2, batch_size=32, workers=8, scope=policy['scope'])
            directory = root / 'requests' / label
            directory.mkdir(parents=True)
            path = directory / 'request.json'
            write_json(path, q)
            worker['checked_request'](path)
            requests[label] = {'path': str(path), 'sha256': file_digest(path), 'family': family, 'index': index}
    assert not torch.cuda.is_initialized()
    write_json(root / 'staging.json', {'status': 'validated-cpu', 'requests': requests,
        'new_fits': 16, 'reused_late_fits': 4, 'scope': policy['scope']})
    print('Validated eight missing VOC reference requests; sixteen fits, four late fits reused.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root', type=Path)
    parser.add_argument('--cnn-tail', type=Path)
    parser.add_argument('--vit-tail', type=Path)
    args = parser.parse_args()
    if bool(args.cnn_tail) != bool(args.vit_tail):
        parser.error('Supply both verified tail requests together')
    stage(args.run_root, {'cnn': args.cnn_tail, 'vit': args.vit_tail} if args.cnn_tail else None)
