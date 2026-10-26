"""Derive fresh, public-reference-only family plans from preserved source evidence."""

from copy import deepcopy
from pathlib import Path
import argparse
import shutil

from c3_acceptance import ROOT, RUN, read, sha, save, replace, now, public_reference


SCOPE = RUN / "public-reference-scope-20261024"
CORE = SCOPE / "runner/c3_acceptance.py"
PHASES = (
    ("full-imagenet-family-plan.json", "full-imagenet-family-plan.json"),
    ("full-spair-family-plan.json", "full-spair-family-plan.json"),
    ("full-clevr-search-grid-plan.json", "clevr-joint-search-plan.json"),
)
HELPERS = ("validated_calibration_model.py", "verify_imagenet_controls.py", "verify_spair_controls.py",
           "verify_full_probe.py", "calibrate_monitor_probes.py")


def clean_metadata(plan):
    """Inventory actual scoped callers and leave old studies in source evidence."""
    for key in list(plan):
        if key.startswith(("original_", "historical_")) or key in (
            "source_archive", "predecessor_source_archive", "model_config_relocations",
            "motivation", "verification_evidence", "baseline_plan_sha256"):
            plan.pop(key)
    source = Path(plan["package_source_root"])
    plan["durable_caller_sources"] = {str(p.relative_to(source)): {"file": str(p), "sha256": sha(p)}
        for p in sorted((source / "caller").rglob("*")) if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"}
    plan["verification_requirements"] = [
        "Evaluate only the declared official public pretrained checkpoints.",
        "Revalidate pinned environment, model/readout contracts, task/source identities and output ownership before evaluation.",
        "Independently reconstruct retained scores and preserve registered numerical tolerances.",
        "Verify exact source/sample coverage and zero retained model-specific temporary state.",
    ]
    return plan


def revise_metadata():
    """Version queued inputs rather than changing plans already submitted."""
    directory = SCOPE / "family-plan-v2"
    previous = None
    report = []
    for _, name in PHASES:
        old_path = SCOPE / "plans" / name
        path = directory / "plans" / name
        assert not path.exists(), "Inspect existing revision before creating another"
        old = read(old_path)
        new = clean_metadata(replace(deepcopy(old), {str(old_path): str(path)}))
        for job in new["jobs"]:
            public_reference(job)
            assert not Path(job["output"]).exists()
        assert not Path(new["state"]).exists()
        new.pop("c3_predecessor", None)
        if previous is not None:
            new["c3_predecessor"] = {"plan": str(previous), "plan_sha256": sha(previous)}
        new["metadata_revision"] = {"source_plan": str(old_path), "source_plan_sha256": sha(old_path),
            "reason": "Remove obsolete model-study inventories and descriptions; regenerate caller inventory from actual public-only sources. Scientific inputs and fresh output destinations unchanged."}
        new["prepared_at_utc"] = now()
        save(path, new)
        report.append({"plan": str(path), "plan_sha256": sha(path), "previous_plan": str(old_path),
                       "previous_plan_sha256": sha(old_path), "controls": [j["name"] for j in new["jobs"]]})
        previous = path
    save(directory / "preparation.json", {"status": "prepared-awaiting-revalidation", "plans": report})
    for item in report:
        print(item["plan"], item["controls"])


def prepare():
    manifest = ROOT / "docs/reference-models.json"
    approved = {r["sha256"] for r in read(manifest)["pretrained_references"]}
    previous = None
    report = []
    for original_name, name in PHASES:
        original = RUN / "plans" / original_name
        old = read(original)
        selected = [deepcopy(j) for j in old["jobs"] if j["model"]["info"]["recipe"].get("sha256") in approved]
        assert selected, "No eligible public reference jobs"
        for job in selected:
            public_reference(job)
        phase = name.removesuffix("-plan.json")
        plan_path = SCOPE / "plans" / name
        assert not plan_path.exists(), "Inspect existing derived plans before creating another revision"
        source = SCOPE / "sources" / phase
        native_source = Path(old["package_source_root"])
        caller = source / "caller"
        shutil.copytree(native_source / "bench", source / "bench", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        caller.mkdir(parents=True)
        for helper in HELPERS:
            if (native_source / "caller" / helper).exists():
                shutil.copyfile(native_source / "caller" / helper, caller / helper)
        mapping = {str(native_source): str(source), str(original): str(plan_path)}
        for job in selected:
            mapping[job["output"]] = str(RUN / "results/public-references-20261024" / phase / job["name"])
            mapping[job["log"]] = str(RUN / "logs/public-references-20261024" / phase / (job["name"] + ".log"))
        reference_helper = native_source / "caller/c3_reference.py"
        if reference_helper.exists():
            body = reference_helper.read_text().splitlines()
            clean_mapping = {"/data/repos/bench": str(ROOT),
                             "/data/cache/datasets/imagenet": str(ROOT / "outputs/c3-source-overlays/imagenet"),
                             "/data/cache/datasets/pascal-voc-2012/VOCdevkit/VOC2012": str(ROOT / "outputs/c3-source-overlays/pascal-voc-2012/VOCdevkit/VOC2012"),
                             "/data/cache/datasets/ade20k/ADEChallengeData2016": str(ROOT / "outputs/c3-source-overlays/ade20k/ADEChallengeData2016")}
            body = ["MAPPING=" + repr(clean_mapping) if line.startswith("MAPPING=") else line for line in body]
            (caller / "c3_reference.py").write_text("\n".join(body) + "\n")
        import yaml
        for job in selected:
            expected = Path(job["expected_model"])
            destination = Path(replace(str(expected), mapping))
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(expected, destination)
            wrapper = Path(job["model_yaml"])
            cfg = yaml.safe_load(wrapper.read_text())
            assert cfg["_target_"] == "validated_calibration_model.construct"
            base_yaml = Path(cfg["base_yaml"])
            target = Path(replace(str(base_yaml), mapping))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(replace(base_yaml.read_text(), mapping))
            save(Path(replace(str(wrapper), mapping)), replace(cfg, mapping))
        new = replace(deepcopy(old), mapping)
        new["jobs"] = replace(selected, mapping)
        new["package_source_root"] = str(source)
        new["state"] = str(RUN / "states/public-references-20261024" / (phase + ".json"))
        assert not Path(new["state"]).exists()
        for field in ("c3_predecessor", "baseline_verification_command", "baseline_plan"):
            if field in new:
                new["historical_" + field] = new.pop(field)
        if previous is not None:
            new["c3_predecessor"] = {"plan": str(previous), "plan_sha256": sha(previous)}
        for key in ("total_fits", "total_variants", "total_training_epochs", "expected_records", "expected_retained_records"):
            if isinstance(old.get(key), int):
                assert old[key] * len(selected) % len(old["jobs"]) == 0
                new[key] = old[key] * len(selected) // len(old["jobs"])
        if "comparison_output" in new:
            new["comparison_output"] = str(RUN / "results/public-references-20261024" / phase / "comparison.json")
        new.update(status="prepared", scope_revision="public-reference-models-20261024", prepared_at_utc=now(),
                   scope="Official public pretrained references only. Original scientific recipes, model states and tolerances retained; fresh output locations.",
                   runner=str(CORE), runner_sha256=sha(CORE))
        new["relocation"] = {"original_plan": str(original), "original_plan_sha256": sha(original), "mapping": mapping,
                             "scope_revision": new["scope_revision"], "host": "c3"}
        pins = {str(p): sha(p) for p in source.rglob("*") if p.is_file()}
        for path in (manifest, CORE, ROOT / "outputs/environments/setup-receipt.json", Path(selected[0]["command"][0]).resolve(), original):
            pins[str(path)] = sha(path)
        for path, expected in old["pinned_files"].items():
            if Path(path).is_relative_to(RUN / "data"):
                assert sha(path) == expected
                pins[path] = expected
        for job in new["jobs"]:
            job["pythonpath"] = [str(source), str(caller)]
            job["model_yaml_sha256"] = sha(job["model_yaml"])
            job["status"] = "pending"
            assert not Path(job["output"]).exists()
            checkpoint = job["model"]["info"]["recipe"]["checkpoint"]
            pins[checkpoint] = sha(checkpoint)
            if job.get("baseline"):
                for p in Path(job["baseline"]).rglob("*"):
                    if p.is_file() and p.suffix in (".json", ".gz", ".jsonl"):
                        pins[str(p)] = sha(p)
        new["pinned_files"] = pins
        clean_metadata(new)
        save(plan_path, new)
        report.append({"plan": str(plan_path), "plan_sha256": sha(plan_path), "controls": [j["name"] for j in new["jobs"]],
                       "source_pins": len(pins), "source_original_plan_sha256": sha(original)})
        previous = plan_path
    save(SCOPE / "family-plan-preparation.json", {"status": "prepared-awaiting-cpu-contract-validation", "prepared_at_utc": now(),
         "plans": report, "scope": "No GPU evaluation or score claim; full ImageNet source validation remains deferred until its checksum-verified copy."})
    for item in report:
        print(item["plan"], item["controls"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--revise-metadata", action="store_true")
    arguments = parser.parse_args()
    revise_metadata() if arguments.revise_metadata else prepare()
