"""Dispatch setup actions before importing the model/evaluation stack."""

import json
import sys

from hydra import compose, initialize_config_module
from hydra.errors import HydraException
from omegaconf import OmegaConf
from omegaconf.errors import OmegaConfBaseException


def main():
    arguments = sys.argv[1:]
    if not arguments or arguments[0] in ("-h", "--help"):
        print("(uv run bench prepare 'names=[imagenet-100,pascal]' data_root=~/datasets)\n(uv run bench visualize 'names=[imagenet-100,pascal]' data_root=~/datasets)\n(uv run bench prune work_dir=/path/to/training/.bench-work)\nRecipes are implementation candidates pending release calibration.")
        return
    action = arguments.pop(0) if arguments[0] in ("prepare", "visualize", "prune") else "config"
    try:
        with initialize_config_module(config_module="bench.cli.configs", version_base="1.3"):
            configuration = compose(config_name=action, overrides=arguments)
        values = OmegaConf.to_container(configuration, resolve=True, throw_on_missing=True)
        if action == "config":
            unknown = set(values) - {"names", "data_root", "datasets", "model", "execution", "evaluation", "output_dir"}
            if unknown:
                raise ValueError(f"Task overrides and unknown evaluation fields are unsupported: {sorted(unknown)}")
        if action == "prepare":
            from bench.prepare import prepare

            report = prepare(**values)
            print(json.dumps(report, indent=2))
            if not report["ready"]:
                raise SystemExit(1)
        elif action == "visualize":
            from bench.visualize import visualize

            visualize(**values)
        elif action == "prune":
            from bench.workspace import prune

            prune(**values)
        else:
            from hydra.utils import instantiate
            from bench import Execution, run
            from bench.protocols import prepare_command, resolve_names
            from bench.storage import read_json, resolve_path

            for task in resolve_names(values["names"]):
                configured = values["datasets"].get(task["source"])
                if configured:
                    manifest = read_json(resolve_path(values["data_root"]) / ".bench" / "prepared" / task["id"] / "manifest.json")
                    if resolve_path(configured) != resolve_path(manifest["source_root"]):
                        raise ValueError(f"Configured dataset path differs from prepared data; rerun {prepare_command(values['names'], values['data_root'])} with datasets.{task['source']}={configured}")

            model = instantiate(configuration.model, _convert_="all")
            model = model.to(values["execution"]["device"])
            run(model, names=values["names"], evaluation=values["evaluation"], data_root=values["data_root"], output_dir=values["output_dir"], execution=Execution(**values["execution"]))
    except (OSError, ValueError, TypeError, RuntimeError, ImportError, HydraException, OmegaConfBaseException) as error:
        print(f"bench: {error}", file=sys.stderr)
        raise SystemExit(1) from None
