"""Assemble compact public reference evidence; release acceptance stays separate."""

import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

import numpy as np

from bench.prepare import load_prepared
from bench.protocols import resolve_names
from bench.runner import read_result
from bench.storage import contained_path, digest, file_digest, read_json, write_json


def reconstruct(task, result, records, rows):
    ids = [row['id'] for row in rows]
    assert ids and len(ids) == len(set(ids)) == result['sample_count']
    seeds = task.get('probe', {}).get('seeds', [None])
    index = {(r['id'], r.get('seed')): r for r in records}
    assert len(index) == len(records) and set(index) == {(sample, seed) for sample in ids for seed in seeds}
    scores = []
    for seed in seeds:
        selected = [index[sample, seed] for sample in ids]
        if task['kind'] == 'classification':
            assert [r['label'] for r in selected] == [row['target'] for row in rows]
            assert all(type(r['prediction']) is int and 0 <= r['prediction'] < task['classes'] for r in selected)
            score = 100 * sum(r['prediction'] == r['label'] for r in selected) / len(ids)
            eligible = len(ids)
        elif task['kind'] == 'segmentation':
            classes = task['classes']
            assert all(type(r['eligible_pixels']) is int and r['eligible_pixels'] >= 0 for r in selected)
            assert all(len(r[k]) == classes and all(type(n) is int and 0 <= n <= r['eligible_pixels'] for n in r[k])
                       for r in selected for k in ['intersection', 'union'])
            counts = np.array([[r['intersection'], r['union']] for r in selected], dtype=np.int64)
            assert np.all(counts[:, 0] <= counts[:, 1])
            intersection, union = counts.sum(0)
            assert (union > 0).any()
            score = float((intersection[union > 0] / union[union > 0]).mean() * 100)
            eligible = sum(r['eligible_pixels'] for r in selected)
            assert [r['eligible_pixels'] for r in selected] == [index[sample, seeds[0]]['eligible_pixels'] for sample in ids]
        else:
            assert task['kind'] == 'correspondence'
            assert all(all(type(r[k]) is int for k in ['correct', 'grid_ceiling_correct', 'eligible']) and
                       0 <= r['correct'] <= r['grid_ceiling_correct'] <= r['eligible'] for r in selected)
            eligible = sum(r['eligible'] for r in selected)
            assert eligible > 0
            score = 100 * sum(r['correct'] for r in selected) / eligible
        if seed is not None:
            saved = next(r for r in result['seed_results'] if r['seed'] == seed)
            assert saved['sample_count'] == len(ids) and saved['eligible_count'] == eligible
            assert math.isclose(saved['score'], score, rel_tol=0, abs_tol=1e-9)
        scores.append(score)
    metrics = {task['metric']: float(np.mean(scores))}
    if len(seeds) > 1:
        metrics[task['metric'] + '_std'] = float(np.std(scores))
    if task['kind'] == 'correspondence':
        metrics['grid_limited_pck_ceiling'] = 100 * sum(r['grid_ceiling_correct'] for r in selected) / eligible
    assert set(metrics) == set(result['metrics'])
    assert all(math.isclose(value, result['metrics'][name]['value'], rel_tol=0, abs_tol=1e-9) for name, value in metrics.items())
    if seeds != [None]:
        assert [row['seed'] for row in result['seed_results']] == seeds
        assert [row['seed'] for row in result['selection']['seeds']] == seeds
    return {'metrics': metrics, 'seed_scores': dict(zip(map(str, seeds), scores)), 'records': len(records)}


def build(request_path):
    request_path = Path(request_path)
    request = read_json(request_path)
    assert all(file_digest(p) == expected for p, expected in request['pins'].items())
    references = {r['architecture']: r for r in read_json(request['reference_models'])['pretrained_references']}
    tasks = {task['id']: task for task in resolve_names(request['names'])}
    prepared = {key: load_prepared(Path(request['data_root']), task, request['evaluation'], verify=False)
                for key, task in tasks.items()}
    entries, seen = [], set()
    for source in request['inputs']:
        output = Path(source['output'])
        assert file_digest(output / 'results.json') == source['results_sha256']
        assert file_digest(output / 'manifest.json') == source['manifest_sha256']
        result, manifest = read_result(output), read_json(output / 'manifest.json')
        assert result['status'] == 'complete' and result['evaluation'] == request['evaluation']
        assert result['storage']['retained_temporary_bytes'] == result['storage']['retained_workspace_bytes'] == 0
        model = result['model']
        assert manifest['model'] == model and manifest['identity']['model'] == model['digest']
        assert manifest['identity_digest'] == digest(manifest['identity'])
        assert model['digest'] == digest({k: v for k, v in model.items() if k != 'digest'})
        architecture = source['architecture']
        reference = references[architecture]
        assert model['info']['recipe']['architecture'] == architecture
        checkpoint = model['info']['recipe'].get('sha256', model['info']['recipe'].get('checkpoint_sha256'))
        assert checkpoint == reference['sha256'], 'Source is not the declared public pretrained checkpoint'
        assert model['numerics'] == result['numerics']
        for task_id, measured in result['tasks'].items():
            assert task_id in tasks and measured['status'] == 'complete'
            task = tasks[task_id]
            preparation, rows = prepared[task_id]
            assert measured['task_digest'] == task['digest'] and measured['preparation_digest'] == preparation['digest']
            assert measured['evaluation'] == request['evaluation']
            assert measured['model_identity'] == model['digest'] and measured['method'] == task['method']
            assert measured['readout'] == model['info']['readouts'][task['method']]
            identity = next(t for t in manifest['identity']['tasks'] if t['id'] == task_id)
            assert identity['digest'] == task['digest'] and identity['preparation'] == preparation['digest']
            split = preparation['splits'][request['evaluation']]
            assert measured['data_digest'] == split['digest'] and measured['sample_count'] == split['count']
            statistics = contained_path(output, measured['statistics']['file'])
            saved = read_json(statistics)
            assert saved['task'] == task_id and saved['evaluation'] == request['evaluation']
            proof = reconstruct(task, measured, saved['records'], rows)
            key = architecture, task_id, result['numerics']
            assert key not in seen, 'Choose one explicit source for each reference/task/profile'
            seen.add(key)
            entries.append({'architecture': architecture, 'checkpoint': reference,
                'task': task_id, 'task_digest': task['digest'], 'protocol': task,
                'evaluation': request['evaluation'], 'numerics': result['numerics'],
                'metrics': measured['metrics'], 'seed_scores': proof['seed_scores'],
                'sample_count': len(rows), 'sample_manifests': measured['preparation']['splits'],
                'dataset': measured['preparation']['source'], 'model': model,
                'selection': measured.get('selection'), 'exposure': result['exposure'],
                'software': result['software'], 'hardware': result['hardware'],
                'provenance': {'output': str(output), 'results_sha256': source['results_sha256'],
                    'manifest_sha256': source['manifest_sha256'], 'statistics_sha256': file_digest(statistics),
                    'evaluation_implementation': identity.get('evaluation_implementation'),
                    'evaluation_implementation_status': 'recorded' if 'evaluation_implementation' in identity else 'not-recorded-in-historical-manifest',
                    'reconstructed_records': proof['records']}})
    coverage = [{'architecture': architecture, 'task': task_id,
                 'status': 'record-reconstructed' if any(k[:2] == (architecture, task_id) for k in seen) else 'missing'}
                for architecture in references for task_id in tasks]
    assert all(file_digest(p) == expected for p, expected in request['pins'].items())
    return {'catalogue_schema_version': '0.1.0', 'status': 'development-reference-evidence',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'request_sha256': file_digest(request_path), 'builder_sha256': file_digest(__file__),
            'evaluation': request['evaluation'], 'entries': entries, 'coverage': coverage,
            'missing': [row for row in coverage if row['status'] == 'missing'],
            'scope': 'Precomputed scores for the exact retained development protocols. Source/sample identities, seeds and aggregate metrics reconstructed; encoder inference and source-pixel/geometry verification are not repeated. Paths label historical evidence and are not runtime dependencies. Calibration, numerical qualification, final-partition scores and stable-v1 acceptance remain separate.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists(), 'Choose a fresh catalogue output'
    catalogue = build(args.request)
    write_json(args.output, catalogue)
    print(f"Reconstructed {len(catalogue['entries'])} references; {len(catalogue['missing'])} missing reference/task combinations")
    if args.require_complete and catalogue['missing']:
        raise SystemExit(1)
