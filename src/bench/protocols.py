"""Fixed candidate recipes. Release calibration is tracked independently."""

from importlib.resources import files
from pathlib import Path

from omegaconf import OmegaConf

from .storage import digest, file_digest


def recipes():
    config = OmegaConf.to_container(OmegaConf.load(files("bench.cli.configs.task").joinpath("recipes.yaml")), resolve=True)
    result = {}
    for name, task in config["tasks"].items():
        recipe = {**config["common"], **config[task["kind"]], **task, "name": name}
        if "probe" in recipe:
            recipe["probe"] = {**config["probe_defaults"], **recipe["probe"]}
        recipe["digest"] = digest(recipe)
        result[name] = recipe
    return result


def resolve_names(names):
    available = recipes()
    aliases = {task["id"]: name for name, task in available.items()}
    if not isinstance(names, list) or not names:
        raise ValueError("names must be a nonempty list of benchmark names or exact task identifiers")
    result = []
    for requested in names:
        if not isinstance(requested, str) or requested not in available and requested not in aliases:
            raise ValueError(f"Unknown benchmark {requested!r}; choose from {', '.join(available)}")
        task = available[aliases.get(requested, requested)]
        if any(previous["id"] == task["id"] for previous in result):
            raise ValueError(f"Duplicate resolved task: {task['id']}")
        result.append(task)
    return result


def evaluation_source_digest(task):
    sources = ["tasks/correspondence.py", "geometry.py"] if task["kind"] == "correspondence" else ["probes.py"]
    if task["kind"] == "segmentation":
        sources.extend(["geometry.py", "images.py", "tasks/segmentation.py"])
    root = Path(__file__).parent
    return digest({name: file_digest(root / name) for name in sources})


def prepare_command(names, root):
    import json
    import shlex

    return "(uv run bench prepare " + shlex.quote("names=" + json.dumps(names)) + " " + shlex.quote("data_root=" + json.dumps(str(root))) + ")"
