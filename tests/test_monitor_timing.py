from copy import deepcopy
import importlib
from pathlib import Path
import runpy
import sys

import pytest

from bench.prepare import load_prepared, prepare
from bench.protocols import resolve_names
from bench.storage import digest, file_digest, read_json, write_json


@pytest.fixture
def timing():
    return runpy.run_path(str(Path(__file__).parents[1] / 'scripts/time_monitoring.py'))


def candidate(task):
    variant = deepcopy(task)
    variant.update(id=task['id'] + '-timing-fixture', name=task['id'] + '-timing-fixture', release_status='calibration-only',
                   calibration={'base_task': task['id'], 'base_task_digest': task['digest']})
    variant['probe'].update(updates=2, learning_rates=[0.03])
    variant['digest'] = digest({k: v for k, v in variant.items() if k != 'digest'})
    return variant


@pytest.mark.parametrize('regularized', [False, True])
def test_timing_alias_preserves_source_splits_and_runs_without_final_data(timing, clevr_data, tmp_path, monkeypatch, regularized):
    import bench
    from test_features import GlobalAdapter

    root, source = clevr_data
    protocols = importlib.import_module('bench.protocols').recipes()
    task = protocols['clevr-count']
    task['probe']['seeds'] = [0]
    task['digest'] = digest({k: v for k, v in task.items() if k != 'digest'})
    assert prepare(names=['clevr-count'], data_root=root)['ready']
    original, training = load_prepared(root, task, 'train', verify=False)
    _, development = load_prepared(root, task, 'development', verify=False)
    base = root / '.bench/prepared' / task['id']
    source_hash = file_digest(base / 'manifest.json')
    variant = candidate(task)
    if regularized:
        variant['calibration'].update(weight_decay_reference=task['probe']['weight_decays'], weight_decay_candidate=1.0)
        variant['probe']['weight_decays'] = [1.0]
        variant['digest'] = digest({k: v for k, v in variant.items() if k != 'digest'})
    destination = tmp_path / 'alias-data'
    final_manifest = base / original['splits']['final']['file']
    original_open = Path.open

    def forbid_final(path, *args, **kwargs):
        assert path.name != final_manifest.name, 'Preparation must not read final rows'
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as guarded:
        guarded.setattr(Path, 'open', forbid_final)
        receipt = timing['prepare_aliases'](root, destination, [variant])
    alias = destination / '.bench/prepared' / variant['id']
    assert (alias / final_manifest.name).stat().st_ino == final_manifest.stat().st_ino
    assert load_prepared(destination, variant, 'train', verify=False)[1] == training
    assert load_prepared(destination, variant, 'development', verify=False)[1] == development
    assert receipt['tasks'][variant['id']]['base_manifest_sha256'] == source_hash == file_digest(base / 'manifest.json')
    for path in (source / 'images/val').iterdir():
        path.unlink()
    protocols[variant['id']] = variant
    result = bench.run(GlobalAdapter(), names=[variant['id']], data_root=destination, output_dir=tmp_path / 'result',
                       execution=bench.Execution(device='cpu', numerics='fp32-v1', workers=0))
    assert result.status == 'complete'
    saved = read_json(result.output_dir / 'results.json')['tasks'][variant['id']]
    assert saved['selection']['seeds'][0]['updates'] == 2
    if regularized:
        assert saved['selection']['seeds'][0]['weight_decay'] == 1.0
    assert not (result.output_dir / '.work').exists()


@pytest.mark.parametrize('invalid', ['reference', 'boolean', 'mismatched_candidate'])
def test_timing_alias_validates_declared_regularization(timing, clevr_data, tmp_path, invalid):
    root, _ = clevr_data
    task = resolve_names(['clevr-count'])[0]
    task['probe']['seeds'] = [0]
    task['digest'] = digest({k: v for k, v in task.items() if k != 'digest'})
    assert prepare(names=['clevr-count'], data_root=root)['ready']
    variant = candidate(task)
    variant['calibration'].update(weight_decay_reference=task['probe']['weight_decays'], weight_decay_candidate=1.0)
    variant['probe']['weight_decays'] = [1.0]
    if invalid == 'reference':
        variant['calibration']['weight_decay_reference'] = [10.0]
    elif invalid == 'boolean':
        variant['calibration']['weight_decay_candidate'] = True
    else:
        variant['calibration']['weight_decay_candidate'] = 0.5
    variant['digest'] = digest({k: v for k, v in variant.items() if k != 'digest'})
    with pytest.raises(AssertionError):
        timing['prepare_aliases'](root, tmp_path / 'alias-data', [variant])


@pytest.mark.parametrize('change', ['canvas', 'bank', 'standardization', 'weight_decay'])
def test_timing_alias_rejects_hidden_protocol_changes(timing, clevr_data, tmp_path, change):
    root, _ = clevr_data
    task = resolve_names(['clevr-count'])[0]
    task['probe']['seeds'] = [0]
    task['digest'] = digest({k: v for k, v in task.items() if k != 'digest'})
    assert prepare(names=['clevr-count'], data_root=root)['ready']
    variant = candidate(task)
    if change == 'weight_decay':
        variant['probe']['weight_decays'] = [1.0]
    elif change == 'canvas':
        variant['canvas'] = 16
    elif change == 'bank':
        variant['bank'] = {'train': 2}
    else:
        variant['standardization']['variance_floor'] = 1e-8
    variant['digest'] = digest({k: v for k, v in variant.items() if k != 'digest'})
    with pytest.raises(AssertionError, match='more than the declared'):
        timing['prepare_aliases'](root, tmp_path / 'alias-data', [variant])


@pytest.mark.parametrize('results_name', ['results.json', 'monitoring.json'])
def test_parent_times_fresh_child_and_observes_first_result_before_verification(tmp_path, monkeypatch, results_name):
    worker = tmp_path / 'worker.py'
    worker.write_text('''import hashlib, json, os, sys, time
from pathlib import Path
p=Path(sys.argv[2]); q=json.loads(p.read_text()); output=Path(q['output'])
results=output/q.get('partial_result_file','results.json')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v));os.replace(t,p)
if sys.argv[1]=='run':
 output.mkdir();time.sleep(0.1)
 save(results,{'status':'running','metrics':{'first':1}})
 time.sleep(0.2)
 save(results,{'status':'complete','metrics':{'first':1,'second':2}})
else:
 assert os.environ['CUDA_VISIBLE_DEVICES']==''
 time.sleep(0.2)
 save(output/'verification.json',{'status':'verified','verified_records':2,'request_sha256':sha(p),
  'results_sha256':sha(results),'timing_sha256':sha(output/'timing.json')})
''')
    request = tmp_path / 'request.json'
    output = tmp_path / 'output'
    write_json(request, {'python_binary': str(Path(sys.executable).resolve()), 'package_root': str(tmp_path / 'sources/bench'),
                        'output': str(output), 'worker': str(worker), 'pins': {str(worker): file_digest(worker)},
                        'first_result_poll_seconds': 0.005, 'latency_limit_seconds': 120,
                        **({'partial_result_file': results_name} if results_name != 'results.json' else {})})
    monkeypatch.setenv('BENCH_EXTENSION_REQUEST_SHA256', file_digest(request))
    parent = runpy.run_path(str(Path(__file__).parents[1] / 'scripts/run_monitor_timing.py'))
    parent['main'](request)
    measured, state = read_json(output / 'timing.json'), read_json(tmp_path / 'request-state.json')
    assert 0 < measured['time_to_first_result_seconds'] < measured['wall_seconds']
    assert state['independent_verification_seconds'] >= 0.2
    assert state['status'] == 'complete' and state['verified_records'] == 2
    assert state['results_file'] == results_name
    assert state['results_sha256'] == file_digest(output / results_name)
