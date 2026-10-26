"""Prepare and run dense acceptance for the declared official public references."""

from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import time
from datetime import datetime, timezone


ROOT = next(p for p in Path(__file__).resolve().parents if (p / "SPEC.md").is_file())
RUN = ROOT / "outputs/c3-acceptance-20261023"
SCOPE = RUN / "public-reference-scope-20261024"
DIRECTORY = SCOPE / "ade-search"
REQUEST = DIRECTORY / "request.json"
CORE = SCOPE / "runner/c3_acceptance.py"
CONTROLS = ("resnet18-pretrained", "vit_b_16-pretrained")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def checked_core(request):
    assert request["status"] == "prepared" and request["stage"] == "search-boundary"
    assert all(sha(p) == expected for p, expected in request["pinned_files"].items()), "Pinned public-reference input changed"
    core = runpy.run_path(str(CORE))
    assert tuple(j["name"] for j in request["jobs"]) == CONTROLS
    for job in request["jobs"]:
        core["public_reference"](job)
    return core


def baseline_boundary(request, core):
    """Bind completed public baselines and actual cleanup without rerunning scores."""
    cancelled = read(request["cancelled_previous_state"])
    assert cancelled["status"] == "cancelled" and cancelled["all_workspaces_pruned"]
    assert all(not (Path(j["output"]) / ".work").exists() for j in cancelled["jobs"])
    records = []
    for job in request["jobs"]:
        core["public_reference"](job)
        output = Path(job["baseline_output"])
        result = read(output / "results.json")
        verification = read(job["baseline_verification"])
        assert result["status"] == "complete" and verification["status"] == "verified"
        assert verification["results_sha256"] == sha(output / "results.json")
        assert verification["records"] == 6063 and verification["selected_seed_count"] == 3
        assert result["storage"]["retained_temporary_bytes"] == result["storage"]["retained_workspace_bytes"] == 0
        assert not (output / ".work").exists()
        actual = core["reidentify"](core["replace"](result["model"], request["path_mapping"]))
        assert actual == job["model"], "Baseline public model identity differs"
        records.append({"name": job["name"], "results_sha256": sha(output / "results.json"),
                        "verification_sha256": sha(job["baseline_verification"])})
    return {"status": "verified", "all_workspaces_pruned": True, "verified_models": len(records),
            "baselines": records, "verified_at_utc": now(),
            "scope": "Existing scientific-verification artifacts and hashes, model identities and actual cleanup; no fresh latency claim."}


def diagnostic_boundary(request, state):
    assert read(DIRECTORY / "state.json") == state, "Diagnostic state changed"
    assert state["status"] == "complete" and state["all_workspaces_pruned"]
    assert state["request_sha256"] == sha(REQUEST)
    assert state["declared_controls"] == list(CONTROLS)
    assert [j["name"] for j in state["jobs"]] == list(CONTROLS)
    assert state["baseline_cleanup_boundary"]["all_workspaces_pruned"]
    for job in state["jobs"]:
        output = Path(job["output"])
        assert not (output / ".work").exists()
        if job["status"] == "skipped-interior-baseline":
            continue
        assert job["status"] == "complete"
        assert sha(output / "calibration.json") == job["calibration_sha256"]
        assert sha(output / "verification.json") == job["verification_sha256"]
        assert read(output / "calibration.json")["retained_temporary_bytes"] == 0
        verified = read(output / "verification.json")
        assert verified["status"] == "verified"
        assert not any(v["at_upper_rate_boundary"] for v in verified["best_rates_by_epoch_budget"].values())
    return {"status": "verified", "all_workspaces_pruned": True, "request_sha256": sha(REQUEST),
            "diagnostic_state_sha256": sha(DIRECTORY / "state.json"), "verified_models": len(CONTROLS)}


def release_command(state, hold, queue):
    diagnostic_boundary({}, state)
    expected = hold["task"]
    task = queue["tasks"][str(expected["id"])]
    assert task["created_at"] == expected["created_at"] and task["group"] == expected["group"]
    assert task["original_command"] == expected["command"] and task["dependencies"] == expected["dependencies"]
    status = task["status"]
    assert status == "Stashed" or (isinstance(status, dict) and set(status) == {"Stashed"}
        and isinstance(status["Stashed"], dict) and status["Stashed"].get("enqueue_at", "unexpected") is None)
    return ["pueue", "enqueue", str(expected["id"])]


VERIFY = """from pathlib import Path
import sys
from bench.storage import read_json,write_json,file_digest
from verify_ade_calibration import verify
p=Path(sys.argv[1]);q=read_json(p);j=next(j for j in q['jobs'] if j['name']==sys.argv[2])
v=verify(Path(j['output']),expected_model=j['model'],software=q['software'],feature_shape=j['feature_shape'],baseline=Path(j['baseline_output']),path_mapping=q['path_mapping'])
assert v['verified_records']==q['records_per_executed_control']
v['request_sha256']=file_digest(p)
write_json(Path(j['output'])/'verification.json',v)
print(j['name'],v['best_rates_by_epoch_budget'],flush=True)
"""


def prepare():
    assert not REQUEST.exists(), "Inspect existing scoped request before deriving another revision"
    declaration = read(SCOPE / "replacement-plan.json")
    previous = Path(declaration["phases"][0]["historical_source_request"])
    assert sha(previous) == declaration["phases"][0]["historical_source_request_sha256"]
    old = read(previous)
    jobs = [deepcopy(j) for j in old["jobs"] if j["name"] in CONTROLS]
    assert tuple(j["name"] for j in jobs) == CONTROLS
    source = SCOPE / "sources/ade"
    caller = source / "caller"
    shutil.copytree(Path(old["package_source_root"]) / "bench", source / "bench", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    caller.mkdir(parents=True)
    for name in ("calibrate_monitor_probes.py", "verify_ade_calibration.py"):
        shutil.copyfile(previous.parent / "caller" / name, caller / name)
    native = Path(old["package_source_root"]) / "caller"
    shutil.copyfile(native / "validated_calibration_model.py", caller / "validated_calibration_model.py")
    for job in jobs:
        name = job["name"]
        shutil.copyfile(native / (name + ".yaml"), caller / (name + ".yaml"))
        shutil.copyfile(job["expected_model"], caller / (name + "-model.json"))
        wrapper = caller / "configs/model" / (name + ".yaml")
        save(wrapper, {"_target_": "validated_calibration_model.construct", "base_yaml": str(caller / (name + ".yaml")), "expected_model": str(caller / (name + "-model.json"))})
        job.update(model_yaml=str(wrapper), model_yaml_sha256=sha(wrapper),
                   expected_model=str(caller / (name + "-model.json")), pythonpath=[str(source), str(caller)],
                   output=str(DIRECTORY / "results" / name), log=str(DIRECTORY / "logs" / (name + ".log")))
        job["command"] = [job["command"][0], "-c", "from bench.cli.main import main; main()",
                          "hydra.searchpath=[file://" + str(caller / "configs") + "]", "model=" + name,
                          "names=[ade20k]", "evaluation=development", "data_root=" + json.dumps(old["data_root"]),
                          "output_dir=" + json.dumps(job["output"]), "execution.batch_size=" + str(job["execution"]["batch_size"]),
                          "execution.device=cuda", "execution.numerics=fp32-v1", "execution.workers=8"]
        job["calibration_command"] = [job["command"][0], str(caller / "calibrate_monitor_probes.py"), "--model", str(wrapper),
                "--data-root", old["data_root"], "--output", job["output"], "--names", "ade20k", "--stage", "search-boundary",
                "--batch-size", str(job["execution"]["batch_size"])]
        job["baseline_verification"] = str(ROOT / "outputs/acceptance/c3-handoff-20261023/ade-recovery-verification/verification.json"
            if name == "resnet18-pretrained" else Path(job["baseline_output"]) / "verification.json")
        job.pop("verification_command", None)
    CORE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "scripts/c3_acceptance.py", CORE)
    shutil.copyfile(__file__, CORE.parent / "public_ade_acceptance.py")
    pins = {str(p): sha(p) for p in source.rglob("*") if p.is_file()}
    for p in [CORE, CORE.parent / "public_ade_acceptance.py", ROOT / "docs/reference-models.json",
              ROOT / "outputs/environments/setup-receipt.json", Path(jobs[0]["command"][0]).resolve(),
              RUN / "states/full-ade-family-queue.json"]:
        pins[str(p)] = sha(p)
    for p, expected in old["pinned_files"].items():
        if Path(p).is_relative_to(RUN / "data"):
            assert sha(p) == expected
            pins[p] = expected
    for job in jobs:
        checkpoint = Path(job["model"]["info"]["recipe"]["checkpoint"])
        pins[str(checkpoint)] = sha(checkpoint)
        for p in [Path(job["baseline_verification"]), *Path(job["baseline_output"]).rglob("*")]:
            if p.is_file() and (p.suffix in (".json", ".gz", ".jsonl")):
                pins[str(p)] = sha(p)
    mapping = {k: v for k, v in old["path_mapping"].items() if k in ("/data/repos/bench", "/data/repos/bench/.venv/bin/python", "/data/repos/bench/outputs/acceptance/data") or k.startswith("/data/cache/datasets/")}
    request = {"status": "prepared", "created_at_utc": now(), "scope_revision": declaration["scope_revision"],
        "source_request_sha256": sha(previous), "package_source_root": str(source), "data_root": old["data_root"],
        "tasks": old["tasks"], "names": ["ade20k"], "software": old["software"], "gpu_uuid": old["gpu_uuid"],
        "path_mapping": mapping, "pinned_files": pins, "jobs": jobs, "stage": "search-boundary",
        "learning_rates": old["learning_rates"], "epoch_budgets": old["epoch_budgets"], "probe_seeds": [0],
        "records_per_executed_control": old["records_per_executed_control"], "maximum_variants": 12,
        "cancelled_previous_state": str(RUN / "states/full-ade-family-queue.json"),
        "scope": "Only official public pretrained reference checkpoints. Fresh outputs, unchanged recipes and native parity tolerances; no release or isolated-timing claim."}
    save(REQUEST, request)
    checked_core(request)
    print(REQUEST)


def execute(action):
    assert sha(__file__) == os.environ["PUBLIC_ADE_RUNNER_SHA256"]
    assert sha(REQUEST) == os.environ["PUBLIC_ADE_REQUEST_SHA256"]
    request = read(REQUEST)
    core = checked_core(request)
    baseline = baseline_boundary(request, core)
    if action == "validate":
        core["validate_family"](REQUEST)
        save(DIRECTORY / "baseline-check.json", baseline)
        return
    validated = read(REQUEST.with_suffix(".validation.json"))
    assert validated["status"] == "passed" and validated["plan_sha256"] == sha(REQUEST)
    state_path = DIRECTORY / "state.json"
    if action == "transfer":
        boundary = diagnostic_boundary(request, read(state_path))
        core["transfer_imagenet"].__globals__["ade_cleanup_boundary"] = lambda: boundary
        core["transfer_imagenet"]()
        return
    assert not state_path.exists(), "Inspect existing public diagnostic state before recovery"
    core["boundary_ready"]()
    core["validate_family"](REQUEST, models_only=True)
    hold_path = DIRECTORY / "transfer-hold.json"
    hold = read(hold_path)
    state = {"status": "running", "request_sha256": sha(REQUEST), "started_at_utc": now(),
        "baseline_cleanup_boundary": baseline, "declared_controls": list(CONTROLS), "hold_sha256": sha(hold_path), "jobs": []}
    save(state_path, state)
    try:
        for job in request["jobs"]:
            checked_core(request)
            output = Path(job["output"])
            assert not output.exists(), "Diagnostic requires fresh output"
            record = {"name": job["name"], "output": str(output)}
            state["jobs"].append(record)
            selected = read(Path(job["baseline_output"]) / "results.json")["tasks"]["ade20k-linear-v1"]["selection"]["seeds"][0]
            upper = max(request["tasks"]["ade20k-linear-v1"]["recipe"]["probe"]["learning_rates"])
            if selected["learning_rate"] < upper:
                record.update(status="skipped-interior-baseline", selected_learning_rate=selected["learning_rate"])
                save(state_path, state)
                continue
            assert selected["learning_rate"] == upper
            assert subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines() == [request["gpu_uuid"]]
            assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
            assert shutil.disk_usage(DIRECTORY).free > job["estimated_peak_temporary_bytes"] + 10_000_000_000
            env = core["environment"](job)
            log = Path(job["log"]); log.parent.mkdir(parents=True, exist_ok=True)
            record.update(status="running", started_at_utc=now(), command=job["calibration_command"])
            save(state_path, state)
            started = time.monotonic()
            print("Starting public ADE diagnostic:", job["name"], flush=True)
            with log.open("w") as stream:
                subprocess.run(job["calibration_command"], env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            subprocess.run([job["command"][0], "-c", VERIFY, str(REQUEST), job["name"]], env=env, check=True)
            verified = read(output / "verification.json")
            record.update(status="complete", finished_at_utc=now(), calibration_sha256=sha(output / "calibration.json"),
                verification_sha256=sha(output / "verification.json"), best_rates_by_epoch_budget=verified["best_rates_by_epoch_budget"],
                seconds_including_loading_cleanup_and_verification=time.monotonic() - started)
            assert not (output / ".work").exists()
            save(state_path, state)
        unresolved = [j["name"] for j in state["jobs"] if any(v["at_upper_rate_boundary"] for v in j.get("best_rates_by_epoch_budget", {}).values())]
        state.update(status="needs-search-extension" if unresolved else "complete", all_workspaces_pruned=True,
                     unresolved_upper_boundaries=unresolved, finished_at_utc=now())
        save(state_path, state)
        if unresolved:
            print("Search extension required; ImageNet copy remains held:", unresolved, flush=True)
            return
        diagnostic_boundary(request, state)
        assert sha(hold_path) == state["hold_sha256"]
        queue = json.loads(subprocess.check_output(["pueue", "status", "--json"], text=True))
        command = release_command(state, hold, queue)
        subprocess.run(command, check=True)
        state["transfer_release"] = {"command": command, "at_utc": now(), "task": hold["task"]}
        save(state_path, state)
    except BaseException as error:
        state.update(status="failed", failed_at_utc=now(), error=str(error))
        save(state_path, state)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["prepare", "validate", "run", "transfer"])
    action = parser.parse_args().action
    prepare() if action == "prepare" else execute(action)
