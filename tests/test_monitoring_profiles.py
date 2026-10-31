from pathlib import Path
import runpy

import pytest

from bench import RunResult
from bench.storage import file_digest, read_json, write_json


@pytest.fixture
def caller():
    return runpy.run_path(str(Path(__file__).parents[1] / 'scripts/time_monitoring_profiles.py'))


@pytest.mark.parametrize('outcome', ['complete', 'second-task-fails', 'incorrect-profile'])
def test_per_task_profiles_keep_ordinary_artifacts_and_preserve_partial_results(caller, tmp_path, outcome):
    variants = [{'id': 'global', 'kind': 'classification'}, {'id': 'dense', 'kind': 'segmentation'}]
    profiles = {'global': 'fp32-v1', 'dense': 'bf16-v1'}
    models = {profile: {'digest': profile, 'numerics': profile} for profile in profiles.values()}
    output = tmp_path / 'pair'
    calls = []
    first_hash = None

    def ordinary_run(model, *, names, evaluation, data_root, output_dir, execution):
        nonlocal first_hash
        name, = names
        calls.append((name, execution.numerics))
        assert evaluation == 'development' and execution.device == 'cuda'
        assert execution.batch_size == 32 and execution.workers == 8
        if name == 'dense':
            partial = read_json(output / 'monitoring.json')
            assert partial['status'] == 'running' and set(partial['metrics']) == {'global'}
            assert partial['tasks']['global']['status'] == 'complete'
            first_hash = file_digest(output / 'tasks/global/results.json')
            if outcome == 'second-task-fails':
                raise RuntimeError('dense failed')
        actual_profile = 'bf16-v1' if name == 'global' and outcome == 'incorrect-profile' else execution.numerics
        metrics = {name: {'score': 20.0 if name == 'global' else 30.0}}
        statistics = Path(output_dir) / 'statistics.json'
        write_json(statistics, {'task': name, 'evaluation': evaluation, 'records': []})
        write_json(Path(output_dir) / 'results.json', {'schema_version': '1.0.0', 'status': 'complete',
            'evaluation': evaluation, 'model': models[actual_profile], 'numerics': actual_profile,
            'metrics': metrics, 'tasks': {name: {'status': 'complete',
                'statistics': {'file': statistics.name, 'sha256': file_digest(statistics)}}},
            'storage': {'retained_temporary_bytes': 0, 'retained_workspace_bytes': 0}})
        return RunResult('1.0.0', 'complete', metrics, Path(output_dir))

    arguments = dict(variants=variants, task_numerics=profiles, data_root=tmp_path / 'data', output=output,
                     batch_size=32, workers=8, expected_models=models, run_task=ordinary_run)
    if outcome == 'complete':
        result = caller['run_tasks'](object(), **arguments)
        assert result['status'] == 'complete' and set(result['metrics']) == {'global', 'dense'}
        assert calls == [('global', 'fp32-v1'), ('dense', 'bf16-v1')]
        assert result['retained_temporary_bytes'] == 0
        assert file_digest(output / 'tasks/global/results.json') == first_hash
        # This caller index never masquerades as an ordinary bench result.
        assert result['kind'] == 'two-task-monitoring-caller' and 'schema_version' not in result
    elif outcome == 'second-task-fails':
        with pytest.raises(RuntimeError, match='dense failed'):
            caller['run_tasks'](object(), **arguments)
        saved = read_json(output / 'monitoring.json')
        assert saved['status'] == 'failed' and saved['tasks']['dense']['status'] == 'failed'
        assert set(saved['metrics']) == {'global'} and saved['tasks']['global']['status'] == 'complete'
        assert file_digest(output / 'tasks/global/results.json') == first_hash
    else:
        with pytest.raises(AssertionError):
            caller['run_tasks'](object(), **arguments)
        assert calls == [('global', 'fp32-v1')]
        saved = read_json(output / 'monitoring.json')
        assert saved['status'] == 'failed' and saved['metrics'] == {}


def test_unsupported_profile_map_is_rejected_before_creating_results(caller, tmp_path):
    output = tmp_path / 'pair'
    with pytest.raises(AssertionError):
        caller['run_tasks'](object(), variants=[{'id': 'global', 'kind': 'classification'}, {'id': 'dense', 'kind': 'segmentation'}],
            task_numerics={'global': 'bf16-v1', 'dense': 'bf16-v1'}, data_root=tmp_path, output=output,
            batch_size=32, workers=8, expected_models={})
    assert not output.exists()
