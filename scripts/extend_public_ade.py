"""Prepare and run two incremental ResNet18 ADE development fits."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from c3_acceptance import ROOT, RUN, read, sha, save, replace, now


SCOPE = RUN / "public-reference-scope-20261024"
DIRECTORY = SCOPE / "ade-extensions/resnet18-lr0p03"
REQUEST = DIRECTORY / "request-v2.json"
SOURCE = SCOPE / "sources/ade-extension-resnet18-lr0p03"
RUNNER = DIRECTORY / "runner-v2"
NAME = "resnet18-pretrained"


def checked(request):
    assert all(sha(path) == expected for path, expected in request["pinned_files"].items()), "Extension input changed"
    import runpy
    core = runpy.run_path(str(RUNNER / "c3_acceptance.py"))
    assert len(request["jobs"]) == 1 and request["jobs"][0]["name"] == NAME
    assert request["rate_multipliers"] == [30] and request["epoch_budgets"] == [40, 80]
    core["public_reference"](request["jobs"][0])
    return core


def predecessor_boundary(request, queue):
    path = Path(request["original_dense_state"])
    state = read(path)
    original = read(request["original_dense_request"])
    assert sha(request["original_dense_request"]) == request["original_dense_request_sha256"]
    assert state["status"] == "needs-search-extension" and state["all_workspaces_pruned"]
    assert state["request_sha256"] == request["original_dense_request_sha256"]
    assert state["declared_controls"] == ["resnet18-pretrained", "vit_b_16-pretrained"]
    assert [job["name"] for job in state["jobs"]] == state["declared_controls"]
    assert NAME in state["unresolved_upper_boundaries"]
    assert [job["name"] for job in original["jobs"]] == state["declared_controls"]
    for job, expected_job in zip(state["jobs"], original["jobs"]):
        output = Path(job["output"])
        assert job["output"] == expected_job["output"]
        assert job["status"] == "complete" and not (output / ".work").exists()
        assert sha(output / "calibration.json") == job["calibration_sha256"]
        assert sha(output / "verification.json") == job["verification_sha256"]
        assert read(output / "calibration.json")["retained_temporary_bytes"] == 0
        assert read(output / "verification.json")["status"] == "verified"
    expected = request["original_dense_task"]
    task = queue["tasks"][str(expected["id"])]
    assert all(task[key] == expected[key] for key in ("created_at", "group", "dependencies", "label"))
    assert task["original_command"] == expected["command"]
    assert task["status"] == {"Done": "Success"}, "Original ADE worker has not succeeded"
    return {"status": "verified", "state_sha256": sha(path), "all_workspaces_pruned": True}


VERIFY_PREVIOUS = """import json,sys
from pathlib import Path
from bench.storage import read_json,file_digest
from verify_ade_calibration import verify
q=read_json(Path(sys.argv[1]));j=q['jobs'][0];p=Path(q['previous_directory'])
v=verify(p,expected_model=j['model'],software=q['software'],feature_shape=j['feature_shape'],baseline=Path(j['baseline_output']),path_mapping=q['path_mapping'],calibration_helper=q['previous_calibration_helper'])
saved=read_json(p/'verification.json')
assert all(saved[k]==value for k,value in v.items() if k!='verifier_sha256')
assert file_digest(p/'verification.json')==q['previous_verification_sha256']
assert v['verified_records']==12126 and any(x['at_upper_rate_boundary'] for x in v['best_rates_by_epoch_budget'].values())
print(json.dumps({'status':'verified','verification_sha256':file_digest(p/'verification.json'),'records':v['verified_records'],'best_rates_by_epoch_budget':v['best_rates_by_epoch_budget'],'source_labels':v['source_labels']}))
"""

VERIFY_EXTENSION = """import json,sys
from pathlib import Path
from bench.storage import read_json,write_json,file_digest
from verify_ade_calibration import verify
q=read_json(Path(sys.argv[1]));j=q['jobs'][0];p=Path(j['output'])
v=verify(p,expected_model=j['model'],software=q['software'],feature_shape=j['feature_shape'],baseline=Path(j['baseline_output']),path_mapping=q['path_mapping'],rate_multipliers=tuple(q['rate_multipliers']),previous=q['previous_directory'],previous_calibration_helper=q['previous_calibration_helper'])
assert v['verified_records']==4042 and v['previous']['verified_records']==12126
assert v['previous']['verification_sha256']==q['previous_verification_sha256']
v['request_sha256']=file_digest(Path(sys.argv[1]));write_json(p/'verification.json',v)
print(json.dumps({'status':v['status'],'best_rates_by_epoch_budget':v['best_rates_by_epoch_budget'],'new_records':v['verified_records'],'prior_records':v['previous']['verified_records']}))
"""


def prepare():
    assert not REQUEST.exists() and not SOURCE.exists() and not RUNNER.exists()
    old_request = SCOPE / "ade-search/request.json"
    old = read(old_request)
    assert all(sha(path) == expected for path, expected in old["pinned_files"].items())
    job = deepcopy(next(item for item in old["jobs"] if item["name"] == NAME))
    previous = Path(job["output"])
    previous_report = read(previous / "calibration.json")
    previous_verification = read(previous / "verification.json")
    assert previous_report["status"] == "complete" and not (previous / ".work").exists()
    assert previous_verification["status"] == "verified" and previous_verification["verified_records"] == 12126
    original = Path(old["package_source_root"])
    shutil.copytree(original / "bench", SOURCE / "bench", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    caller = SOURCE / "caller"
    caller.mkdir()
    for name in ("validated_calibration_model.py", NAME + ".yaml", NAME + "-model.json"):
        shutil.copyfile(original / "caller" / name, caller / name)
    helper = original / "caller/calibrate_monitor_probes.py"
    marker = '        budgets = (2, 4) if stage == "search-extension"'
    text = helper.read_text()
    assert text.count(marker) == 1
    (caller / helper.name).write_text(text.replace(marker,
        '        if stage == "search-boundary":\n            rates = (30,)\n' + marker))
    shutil.copyfile(ROOT / "scripts/verify_ade_calibration.py", caller / "verify_ade_calibration.py")
    wrapper = caller / "configs/model" / (NAME + ".yaml")
    save(wrapper, {"_target_": "validated_calibration_model.construct", "base_yaml": str(caller / (NAME + ".yaml")),
        "expected_model": str(caller / (NAME + "-model.json"))})
    output = DIRECTORY / "results" / NAME
    mapping = {str(original): str(SOURCE), str(previous): str(output)}
    job = replace(job, mapping)
    job.update(output=str(output), log=str(DIRECTORY / "logs" / (NAME + ".log")),
        model_yaml=str(wrapper), model_yaml_sha256=sha(wrapper), status="prepared-incremental-coverage")
    RUNNER.mkdir(parents=True)
    shutil.copyfile(SCOPE / "runner/c3_acceptance.py", RUNNER / "c3_acceptance.py")
    shutil.copyfile(__file__, RUNNER / "extend_public_ade.py")
    pins = dict(old["pinned_files"])
    for path in [old_request, *SOURCE.rglob("*"), *RUNNER.rglob("*"), *previous.rglob("*")]:
        if path.is_file():
            pins[str(path)] = sha(path)
    unchanged = {str(path.relative_to(original)): sha(path) for path in (original / "bench").rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    assert len(unchanged) == 37
    assert all(sha(SOURCE / relative) == checksum for relative, checksum in unchanged.items())
    execution = read(SCOPE / "current-execution.json")
    identities = {item["id"]: item for item in execution["tasks"]}
    request = {**old, "created_at_utc": now(), "status": "prepared", "stage": "incremental-search-boundary",
        "package_source_root": str(SOURCE), "jobs": [job], "pinned_files": pins,
        "rate_multipliers": [30], "learning_rates": [0.03], "epoch_budgets": [40, 80],
        "maximum_variants": 2, "records_per_executed_control": 4042,
        "previous_directory": str(previous), "previous_verification_sha256": sha(previous / "verification.json"),
        "previous_calibration_helper": str(helper), "original_dense_state": str(SCOPE / "ade-search/state.json"),
        "original_dense_request": str(old_request),
        "original_dense_request_sha256": sha(old_request), "original_dense_task": identities[37],
        "estimated_fitting_scoring_seconds": sum(previous_report["results"][f"ade20k-linear-v1-calibration-e{epochs}-s1-lrm10"]["fitting_and_scoring_seconds"] for epochs in (40, 80)),
        "scope": "Two new LR 0.03 fits at the unchanged 40/80-epoch budgets, seed zero, WD 0.01. Reconstruct and retain six prior fits with identical model/source/environment contracts. Does not release the held ImageNet copy or certify unresolved coverage."}
    save(REQUEST, request)
    save(DIRECTORY / "source-audit.json", {"status": "verified", "request_sha256": sha(REQUEST),
        "unchanged_package_files": unchanged, "prior_request_sha256": sha(old_request),
        "prior_helper_sha256": sha(helper), "new_helper_sha256": sha(caller / helper.name),
        "helper_derivation": "Insert only rates=(30,) for search-boundary; preserve all extraction, fitting, scoring and cleanup implementation.",
        "scientific_changes": {"new_learning_rate": 0.03, "epoch_budgets": [40, 80], "probe_seeds": [0]},
        "unchanged": "Model weights/identity, base tasks, data/splits, readouts, package files, execution, WD, selection and tolerances."})
    checked(request)
    print(REQUEST)


def revise_queue():
    previous = DIRECTORY / "request.json"
    old = read(previous)
    assert not REQUEST.exists() and not RUNNER.exists() and not (DIRECTORY / "state.json").exists()
    assert all(not Path(job["output"]).exists() for job in old["jobs"])
    RUNNER.mkdir()
    shutil.copyfile(DIRECTORY / "runner/c3_acceptance.py", RUNNER / "c3_acceptance.py")
    shutil.copyfile(__file__, RUNNER / "extend_public_ade.py")
    request = deepcopy(old)
    request.pop("held_transfer_task")
    request["created_at_utc"] = now()
    request["queue_revision"] = {"previous_request": str(previous), "previous_request_sha256": sha(previous),
        "reason": "Remove a reverse dependency on one held transfer identity. The separate combined transfer boundary requires this extension to complete and clean up before copying. Scientific fits and outputs are unchanged."}
    request["pinned_files"][str(previous)] = sha(previous)
    for path in RUNNER.iterdir():
        request["pinned_files"][str(path)] = sha(path)
    save(REQUEST, request)
    checked(request)
    print(REQUEST)


def validate(request, core):
    job = request["jobs"][0]
    previous = core["probe"]([job["command"][0], "-c", VERIFY_PREVIOUS, str(REQUEST)], core["environment"](job))
    save(DIRECTORY / "previous-check.json", previous)
    core["validate_family"](REQUEST)


def run(request, core):
    assert sha(__file__) == os.environ["PUBLIC_ADE_EXTENSION_RUNNER_SHA256"]
    assert sha(REQUEST) == os.environ["PUBLIC_ADE_EXTENSION_REQUEST_SHA256"]
    assert not (DIRECTORY / "state.json").exists(), "Inspect an existing extension before recovery"
    assert read(REQUEST.with_suffix(".validation.json"))["plan_sha256"] == sha(REQUEST)
    assert read(REQUEST.with_suffix(".validation.json"))["status"] == "passed"
    queue = json.loads(subprocess.check_output(["pueue", "status", "--json"], text=True))
    boundary = predecessor_boundary(request, queue)
    core["boundary_ready"]()
    core["validate_family"](REQUEST, models_only=True)
    job = request["jobs"][0]
    assert Path(job["output"]).resolve() == (DIRECTORY / "results" / NAME).resolve()
    assert not Path(job["output"]).exists()
    assert subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() == [request["gpu_uuid"]]
    assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    assert shutil.disk_usage(DIRECTORY).free > job["estimated_peak_temporary_bytes"] + 10_000_000_000
    state = {"status": "running", "started_at_utc": now(), "request_sha256": sha(REQUEST), "predecessor": boundary, "name": NAME}
    save(DIRECTORY / "state.json", state)
    started = time.monotonic()
    try:
        env = core["environment"](job)
        log = Path(job["log"])
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as stream:
            subprocess.run(job["calibration_command"], env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        subprocess.run([job["command"][0], "-c", VERIFY_EXTENSION, str(REQUEST)], env=env, check=True)
        output = Path(job["output"])
        verification = read(output / "verification.json")
        unresolved = any(value["at_upper_rate_boundary"] for value in verification["best_rates_by_epoch_budget"].values())
        assert not (output / ".work").exists()
        state.update(status="needs-search-extension" if unresolved else "complete", all_workspaces_pruned=True,
            best_rates_by_epoch_budget=verification["best_rates_by_epoch_budget"], calibration_sha256=sha(output / "calibration.json"),
            verification_sha256=sha(output / "verification.json"), finished_at_utc=now(),
            seconds_including_loading_cleanup_and_verification=time.monotonic() - started)
        save(DIRECTORY / "state.json", state)
        print(json.dumps(state), flush=True)
    except BaseException as error:
        state.update(status="failed", error=str(error), failed_at_utc=now())
        save(DIRECTORY / "state.json", state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "revise-queue", "validate", "run"))
    action = parser.parse_args().action
    if action == "prepare":
        prepare()
    elif action == "revise-queue":
        revise_queue()
    else:
        request = read(REQUEST)
        core = checked(request)
        if action == "validate":
            validate(request, core)
        else:
            run(request, core)
