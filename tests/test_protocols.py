from importlib.resources import files

from omegaconf import OmegaConf

from bench import protocols


def test_api_and_cli_default_to_fp32_after_bf16_parity_failures():
    from bench import Execution

    config = OmegaConf.load(files('bench.cli.configs').joinpath('execution/default.yaml'))
    assert Execution().numerics == config.numerics == 'fp32-v1'
    assert Execution(device='cpu').numerics == 'fp32-v1'
    assert Execution(numerics='bf16-v1').numerics == 'bf16-v1'


def test_recipe_calls_return_independent_nested_values():
    original = protocols.recipes()
    changed = protocols.recipes()
    changed['imagenet-1k']['probe']['learning_rates'].append(999.)
    changed['pascal']['standardization']['variance_floor'] = 123.
    assert protocols.recipes() == original


def test_recipe_asset_change_invalidates_parsed_configuration(tmp_path, monkeypatch):
    contents = files('bench.cli.configs.task').joinpath('recipes.yaml').read_text()
    path = tmp_path / 'recipes.yaml'
    path.write_text(contents)
    monkeypatch.setattr(protocols, 'files', lambda package: tmp_path)
    first = protocols.recipes()
    changed = OmegaConf.create(contents)
    changed['tasks']['pascal']['canvas'] = 256
    OmegaConf.save(changed, path)
    second = protocols.recipes()
    assert first['pascal']['canvas'] == 160 and second['pascal']['canvas'] == 256
    assert first['pascal']['digest'] != second['pascal']['digest']
    assert first['imagenet-1k'] == second['imagenet-1k']


def test_recipe_interpolations_resolve_again_without_an_asset_change(tmp_path, monkeypatch):
    config = OmegaConf.load(files('bench.cli.configs.task').joinpath('recipes.yaml'))
    config['tasks']['pascal']['reference'] = '${oc.env:BENCH_TEST_REFERENCE}'
    OmegaConf.save(config, tmp_path / 'recipes.yaml')
    monkeypatch.setattr(protocols, 'files', lambda package: tmp_path)
    monkeypatch.setenv('BENCH_TEST_REFERENCE', 'first')
    first = protocols.recipes()
    monkeypatch.setenv('BENCH_TEST_REFERENCE', 'second')
    second = protocols.recipes()
    assert first['pascal']['reference'] == 'first' and second['pascal']['reference'] == 'second'
    assert first['pascal']['digest'] != second['pascal']['digest']
