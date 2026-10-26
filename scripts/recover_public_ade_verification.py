"""Verify completed public ADE data after the frozen checker's float-coherence defect."""

import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess

from c3_acceptance import ROOT, RUN, read, sha, save, now


SCOPE = RUN / "public-reference-scope-20261024"
DIRECTORY = SCOPE / "ade-search/recovery-verification"
REQUEST = DIRECTORY / "request.json"
RUNNER = DIRECTORY / "runner"
CONTROLS = ("resnet18-pretrained", "vit_b_16-pretrained")


def completed_boundary(request, queue):
    expected = request["original_task"]
    task = queue["tasks"][str(expected["id"])]
    assert task["status"] in ({"Done": "Success"}, {"Done": {"Failed": 1}}), "Original worker is not an eligible terminal process"
    assert task["original_command"] == expected["command"]
    assert all(task[key] == expected[key] for key in ("created_at", "group", "label", "dependencies"))
    original = read(request["original_request"])
    assert sha(request["original_request"]) == request["original_request_sha256"]
    state = read(request["original_state"])
    assert state["status"] in ("failed", "needs-search-extension", "complete")
    assert state["request_sha256"] == request["original_request_sha256"]
    assert state["declared_controls"] == list(CONTROLS)
    assert [job["name"] for job in original["jobs"]] == list(CONTROLS)
    reports = []
    for job in original["jobs"]:
        output = Path(job["output"])
        report, plan = read(output / "calibration.json"), read(output / "plan.json")
        assert report["status"] == "complete" and report["retained_temporary_bytes"] == 0
        assert not (output / ".work").exists()
        assert len(report["results"]) == len(plan["variants"]) == 6
        assert report["plan_sha256"] == sha(output / "plan.json")
        assert plan["model"] == job["model"] and plan["software"] == original["software"]
        assert plan["numerics"] == "fp32-v1" and plan["stage"] == "search-boundary"
        reports.append({"name": job["name"], "calibration_sha256": sha(output / "calibration.json")})
    return {"status": "complete-data-awaiting-scientific-reverification", "original_task_status": task["status"],
        "original_state_sha256": sha(request["original_state"]), "reports": reports,
        "all_workspaces_pruned": True, "observed_at_utc": now()}


SCIENCE = """import json,sys
from pathlib import Path
from bench.storage import read_json,write_json,file_digest,lock
from verify_ade_calibration import verify
q=read_json(Path(sys.argv[1]));original=read_json(q['original_request']);checks=[]
import bench
assert Path(bench.__file__).is_relative_to(Path(original['package_source_root'])/'bench')
assert Path(sys.modules['verify_ade_calibration'].__file__)==Path(q['verifier'])
assert file_digest(q['verifier'])==q['verifier_sha256']
for job in original['jobs']:
 p=Path(job['output'])
 with lock(p.parent/('.'+p.name+'.bench-output-lock')):
  actual=verify(p,expected_model=job['model'],software=original['software'],feature_shape=job['feature_shape'],baseline=Path(job['baseline_output']),path_mapping=original['path_mapping'],calibration_helper=q['original_helper'])
  assert actual['verified_records']==12126
  existing=p/'verification.json';was_present=existing.exists()
  if was_present:
   saved=read_json(existing)
   assert saved['request_sha256']==q['original_request_sha256']
   assert all(saved[k]==value for k,value in actual.items() if k!='verifier_sha256')
  else:
   saved={**actual,'request_sha256':q['original_request_sha256'],'verification_recovery_request_sha256':file_digest(Path(sys.argv[1]))}
   write_json(existing,saved)
  proof=Path(q['proof_directory'])/job['name']/'verification.json'
  recovery={**actual,'original_request_sha256':q['original_request_sha256'],'recovery_request_sha256':file_digest(Path(sys.argv[1])),'producer_verification_already_present':was_present,'preserved_or_new_verification_sha256':file_digest(existing)}
  if proof.exists():
   previous=read_json(proof)
   assert all(previous[k]==value for k,value in recovery.items() if k!='producer_verification_already_present')
  else:
   write_json(proof,recovery)
  checks.append({'name':job['name'],'output':str(p),'status':'complete','calibration_sha256':file_digest(p/'calibration.json'),'verification_sha256':file_digest(existing),'recovery_verification_sha256':file_digest(proof),'best_rates_by_epoch_budget':actual['best_rates_by_epoch_budget']})
torch=sys.modules.get('torch');assert torch is None or not torch.cuda.is_initialized()
print(json.dumps({'status':'verified','jobs':checks,'records_verified':24252,'cuda_initialized':False}))
"""


def checked(request):
    assert all(sha(path) == expected for path, expected in request["pinned_files"].items()), "Recovery input changed"
    core = runpy.run_path(str(RUNNER / "c3_acceptance.py"))
    original = read(request["original_request"])
    assert [job["name"] for job in original["jobs"]] == list(CONTROLS)
    for job in original["jobs"]:
        core["public_reference"](job)
    return core, original


def prepare():
    assert not REQUEST.exists() and not RUNNER.exists()
    original_path = SCOPE / "ade-search/request.json"
    original = read(original_path)
    extension_path = SCOPE / "ade-extensions/resnet18-lr0p03/request-v2.json"
    extension = read(extension_path)
    pins = {**original["pinned_files"], **extension["pinned_files"]}
    assert all(sha(path) == expected for path, expected in pins.items())
    observation = SCOPE / "ade-search/roundoff-observation.json"
    issue = read(observation)
    assert issue["coherence_passed"] and issue["largest_absolute_roundoff"] <= 1e-10
    assert issue["registered_score_tolerance_percentage_points"] == 0.1
    RUNNER.mkdir(parents=True)
    shutil.copyfile(SCOPE / "runner/c3_acceptance.py", RUNNER / "c3_acceptance.py")
    shutil.copyfile(__file__, RUNNER / "recover_public_ade_verification.py")
    for path in [original_path, extension_path, observation, *RUNNER.iterdir()]:
        pins[str(path)] = sha(path)
    current = read(SCOPE / "current-execution.json")
    expected = next(task for task in current["tasks"] if task["id"] == 37)
    caller = Path(extension["package_source_root"]) / "caller"
    request = {"status": "prepared", "scope_revision": original["scope_revision"], "created_at_utc": now(),
        "original_request": str(original_path), "original_request_sha256": sha(original_path),
        "original_state": str(SCOPE / "ade-search/state.json"), "original_task": expected,
        "original_helper": str(Path(original["package_source_root"]) / "caller/calibrate_monitor_probes.py"),
        "verifier": str(caller / "verify_ade_calibration.py"), "verifier_sha256": sha(caller / "verify_ade_calibration.py"),
        "proof_directory": str(DIRECTORY / "proofs"), "pinned_files": pins,
        "coherence_tolerance": 1e-10, "registered_score_tolerance_percentage_points": 0.1,
        "scope": "Verification only after exact original worker 37 terminates and both six-variant controls complete and prune. Reconstruct all records using the reviewed floating-coherence check. Preserve the original worker outcome and state. Add a missing independent verification artifact only after scientific checks pass; write a separate effective state. No model inference or fitting."}
    save(REQUEST, request)
    checked(request)
    print(REQUEST)


def runtime_check(core, original):
    job = original["jobs"][0]
    runtime = core["probe"]([job["command"][0], "-c", core["ENV_PROBE"], json.dumps(job["environment_versions"])], core["environment"](job))
    assert runtime["binary_sha256"] == read(ROOT / "outputs/environments/setup-receipt.json")["environments"]["bench"]["python_binary_sha256"]
    return runtime


def execute(action):
    request = read(REQUEST)
    core, original = checked(request)
    if action == "validate":
        runtime = runtime_check(core, original)
        save(DIRECTORY / "configuration-validation.json", {"status": "passed", "request_sha256": sha(REQUEST),
            "runtime": runtime, "checked_at_utc": now(), "pins_verified": len(request["pinned_files"]),
            "scope": "Static inputs and runtime only; original fits remain live and complete-data/scientific guards must pass before verification."})
        return
    assert sha(__file__) == os.environ["PUBLIC_ADE_RECOVERY_RUNNER_SHA256"]
    assert sha(REQUEST) == os.environ["PUBLIC_ADE_RECOVERY_REQUEST_SHA256"]
    validated = read(DIRECTORY / "configuration-validation.json")
    assert validated["status"] == "passed" and validated["request_sha256"] == sha(REQUEST)
    assert not (DIRECTORY / "effective-state.json").exists(), "Inspect existing recovery before another attempt"
    runtime = runtime_check(core, original)
    queue = json.loads(subprocess.check_output(["pueue", "status", "--json"], text=True))
    boundary = completed_boundary(request, queue)
    save(DIRECTORY / "original-boundary.json", boundary)
    preserved = DIRECTORY / "original-state-preserved.json"
    if preserved.exists():
        assert sha(preserved) == boundary["original_state_sha256"]
    else:
        shutil.copyfile(request["original_state"], preserved)
    job = original["jobs"][0]
    env = core["environment"](job)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(request["verifier"]).parent), *job["pythonpath"]])
    verified = core["probe"]([job["command"][0], "-c", SCIENCE, str(REQUEST)], env)
    assert boundary["original_state_sha256"] == sha(request["original_state"])
    unresolved = [item["name"] for item in verified["jobs"]
        if any(value["at_upper_rate_boundary"] for value in item["best_rates_by_epoch_budget"].values())]
    state = {"status": "needs-search-extension" if unresolved else "complete", "request_sha256": request["original_request_sha256"],
        "declared_controls": list(CONTROLS), "jobs": verified["jobs"], "all_workspaces_pruned": True,
        "unresolved_upper_boundaries": unresolved, "verified_records": verified["records_verified"],
        "verification_recovery": {"request_sha256": sha(REQUEST), "original_boundary": boundary,
            "verifier_sha256": request["verifier_sha256"], "runtime": runtime, "cuda_initialized": False},
        "completed_at_utc": now(), "scope": "Scientifically reconstructed completed public diagnostic data; original worker outcome/state remain preserved. Search edges and downstream copy requirements are unchanged."}
    save(DIRECTORY / "effective-state.json", state)
    print(json.dumps({"status": state["status"], "records_verified": state["verified_records"], "unresolved_upper_boundaries": unresolved}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "validate", "run"))
    action = parser.parse_args().action
    prepare() if action == "prepare" else execute(action)
