from copy import deepcopy
import importlib

import pytest
from PIL import Image

from bench.protocols import recipes
from bench.storage import digest, write_json


@pytest.fixture
def clevr_data(tmp_path, monkeypatch):
    root = tmp_path / "data"
    source = root / "CLEVR_v1.0"
    for split, count in [("train", 12), ("val", 4)]:
        scenes = []
        (source / "images" / split).mkdir(parents=True)
        for index in range(count):
            name = f"CLEVR_{split}_{index:06d}.png"
            Image.new("RGB", (32, 24), (index * 17, 60, 100)).save(source / "images" / split / name)
            objects = [{"pixel_coords": [4 + item * 3, 10, 7.5 + index / 8 + item / 2]} for item in range(3 + index % 8)]
            scenes.append({"image_filename": name, "objects": objects})
        write_json(source / "scenes" / f"CLEVR_{split}_scenes.json", {"scenes": scenes})
    protocol_module = importlib.import_module("bench.protocols")
    prepare_module = importlib.import_module("bench.prepare")
    task_recipes = deepcopy(recipes())
    for name, recipe in task_recipes.items():
        recipe["development_fraction"] = 0.25
        if recipe["source"] == "clevr":
            recipe["probe"] = {**recipe["probe"], "epochs": 2, "batch_size": 4, "learning_rates": [0.01, 0.03], "weight_decays": [0.0], "seeds": [0, 1]}
        recipe["digest"] = digest({key: value for key, value in recipe.items() if key != "digest"})
    from bench.acquisition import source_recipes

    sources = source_recipes()
    sources["clevr"]["counts"] = {"train": 12, "final": 4}
    monkeypatch.setattr(protocol_module, "recipes", lambda: task_recipes)
    monkeypatch.setattr(prepare_module, "source_recipes", lambda: sources)
    return root, source
