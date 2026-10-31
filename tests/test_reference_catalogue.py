from copy import deepcopy
from pathlib import Path
import runpy

import pytest

from bench.storage import digest, file_digest, read_json, write_json


@pytest.fixture
def catalogue():
    return runpy.run_path(str(Path(__file__).parents[1] / 'scripts/build_reference_catalogue.py'))


def test_classification_reference_reconstructs_every_prescribed_seed(catalogue):
    task = {'kind': 'classification', 'classes': 2, 'metric': 'top1', 'probe': {'seeds': [0, 1]}}
    rows = [{'id': 'a', 'target': 0}, {'id': 'b', 'target': 1}]
    records = [{'id': sample['id'], 'label': sample['target'], 'seed': seed,
                'prediction': sample['target'] if seed == 0 else 0} for seed in [0, 1] for sample in rows]
    result = {'sample_count': 2, 'metrics': {'top1': {'value': 75}, 'top1_std': {'value': 25}},
              'seed_results': [{'seed': seed, 'sample_count': 2, 'eligible_count': 2, 'score': score} for seed, score in [(0, 100), (1, 50)]],
              'selection': {'seeds': [{'seed': 0}, {'seed': 1}]}}
    proof = catalogue['reconstruct'](task, result, records, rows)
    assert proof['seed_scores'] == {'0': 100, '1': 50} and proof['records'] == 4
    with pytest.raises(AssertionError):
        catalogue['reconstruct'](task, result, records[:-1], rows)
    with pytest.raises(AssertionError):
        catalogue['reconstruct'](task, result, records + records[:1], rows)
    records[0]['prediction'] = 2
    with pytest.raises(AssertionError):
        catalogue['reconstruct'](task, result, records, rows)


def test_dense_reference_uses_dataset_iou_and_validates_seed_eligibility(catalogue):
    task = {'kind': 'segmentation', 'classes': 2, 'metric': 'miou', 'probe': {'seeds': [0]}}
    rows = [{'id': 'a'}, {'id': 'b'}]
    records = [
        {'id': 'a', 'seed': 0, 'eligible_pixels': 1, 'intersection': [1, 0], 'union': [1, 0]},
        {'id': 'b', 'seed': 0, 'eligible_pixels': 99, 'intersection': [0, 0], 'union': [99, 99]}]
    result = {'sample_count': 2, 'metrics': {'miou': {'value': 0.5}},
              'seed_results': [{'seed': 0, 'sample_count': 2, 'eligible_count': 100, 'score': 0.5}],
              'selection': {'seeds': [{'seed': 0}]}}
    assert catalogue['reconstruct'](task, result, records, rows)['metrics'] == {'miou': 0.5}
    result['seed_results'][0]['eligible_count'] = 101
    with pytest.raises(AssertionError):
        catalogue['reconstruct'](task, result, records, rows)


def test_correspondence_reference_pools_pairs_and_preserves_ceiling(catalogue):
    task = {'kind': 'correspondence', 'metric': 'pck_at_0.1'}
    rows = [{'id': 'a'}, {'id': 'b'}]
    records = [{'id': 'a', 'correct': 0, 'grid_ceiling_correct': 1, 'eligible': 1},
               {'id': 'b', 'correct': 99, 'grid_ceiling_correct': 99, 'eligible': 99}]
    result = {'sample_count': 2, 'metrics': {'pck_at_0.1': {'value': 99}, 'grid_limited_pck_ceiling': {'value': 100}}}
    assert catalogue['reconstruct'](task, result, records, rows)['metrics']['pck_at_0.1'] == 99
    records[0]['correct'] = 2
    with pytest.raises(AssertionError):
        catalogue['reconstruct'](task, result, records, rows)


def test_catalogue_binds_model_protocol_and_source_records(catalogue, clevr_data, tmp_path):
    import bench
    from bench.prepare import prepare
    from test_features import GlobalAdapter

    root, _ = clevr_data
    prepare(names=['clevr-count', 'clevr-distance'], data_root=root)
    model = GlobalAdapter()
    model.info.recipe.update(architecture='fixture', sha256='a' * 64)
    output = tmp_path / 'reference'
    bench.run(model, names=['clevr-count'], data_root=root, output_dir=output,
              execution=bench.Execution(device='cpu', numerics='fp32-v1', workers=0))
    # Old manifests retain full software hashes without a per-task digest field.
    manifest = read_json(output / 'manifest.json')
    manifest['identity']['tasks'][0].pop('evaluation_implementation')
    manifest['identity_digest'] = digest(manifest['identity'])
    write_json(output / 'manifest.json', manifest)
    references = tmp_path / 'references.json'
    write_json(references, {'pretrained_references': [{'architecture': 'fixture', 'sha256': 'a' * 64}]})
    source = {'architecture': 'fixture', 'output': str(output),
              'results_sha256': file_digest(output / 'results.json'), 'manifest_sha256': file_digest(output / 'manifest.json')}
    request = {'reference_models': str(references), 'names': ['clevr-count', 'clevr-distance'],
               'evaluation': 'development', 'data_root': str(root), 'inputs': [source],
               'pins': {str(references): file_digest(references)}}
    path = tmp_path / 'request.json'
    write_json(path, request)
    report = catalogue['build'](path)
    assert len(report['entries']) == 1 and report['missing'] == [{'architecture': 'fixture', 'task': 'clevr-distance-v1', 'status': 'missing'}]
    assert report['status'] == 'development-reference-evidence'
    assert report['entries'][0]['provenance']['evaluation_implementation'] is None
    assert report['entries'][0]['provenance']['evaluation_implementation_status'] == 'not-recorded-in-historical-manifest'
    changed = deepcopy(request)
    changed['inputs'] += changed['inputs']
    write_json(path, changed)
    with pytest.raises(AssertionError, match='one explicit source'):
        catalogue['build'](path)
    write_json(path, request)
    statistics = output / read_json(output / 'results.json')['tasks']['clevr-count-v1']['statistics']['file']
    statistics.write_text('{}')
    with pytest.raises(ValueError, match='corrupt committed statistics'):
        catalogue['build'](path)
