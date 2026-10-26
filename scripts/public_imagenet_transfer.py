"""Copy missing ImageNet sources after verified public ADE coverage and cleanup."""

import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess

from c3_acceptance import ROOT, RUN, HANDOFF, read, sha, save, now


SCOPE = RUN / "public-reference-scope-20261024"
DIRECTORY = SCOPE / "imagenet-transfer-v2"
REQUEST = DIRECTORY / "request-v3.json"
RUNNER = DIRECTORY / "runner-v3"
VALIDATION = DIRECTORY / "configuration-validation-v3.json"
CONTROLS = ("resnet18-pretrained", "vit_b_16-pretrained")


def proof(job, completed, *, request_sha256, software, records):
    output = Path(job["output"])
    assert completed["status"] == "complete" and completed["name"] == job["name"]
    assert not (output / ".work").exists(), "Dense workspace remains"
    report, plan, verified = (read(output / name) for name in ("calibration.json", "plan.json", "verification.json"))
    assert report["status"] == "complete" and report["retained_temporary_bytes"] == 0
    assert sha(output / "calibration.json") == completed["calibration_sha256"] == verified["calibration_sha256"]
    assert sha(output / "verification.json") == completed["verification_sha256"]
    assert report["plan_sha256"] == verified["plan_sha256"] == sha(output / "plan.json")
    assert plan["model"] == job["model"] and plan["software"] == software and plan["numerics"] == "fp32-v1"
    assert plan["execution"] == {key: job["execution"][key] for key in ("device", "batch_size", "workers")}
    assert verified["request_sha256"] == request_sha256 and verified["status"] == "verified"
    assert verified["verified_records"] == records
    assert verified["source_labels"]["development_images"] == 2021
    assert verified["source_labels"]["verified_source_files"] == 4043
    assert verified["source_labels"]["eligible_pixels"] == 373568791
    assert set(report["results"]) == set(verified["variants"])
    for identifier, result in report["results"].items():
        path = output / result["statistics"]["file"]
        assert path.resolve().is_relative_to(output.resolve())
        assert sha(path) == result["statistics"]["sha256"] == verified["variants"][identifier]["statistics_sha256"]
    assert set(verified["best_rates_by_epoch_budget"]) == {"40", "80"}
    return verified


def combined_boundary(request, queue):
    original = read(request["original_request"])
    extension = read(request["extension_request"])
    original_hash, extension_hash = sha(request["original_request"]), sha(request["extension_request"])
    assert original_hash == request["original_request_sha256"]
    assert extension_hash == request["extension_request_sha256"]
    state = read(request["original_state"])
    extra = read(request["extension_state"])
    assert state["status"] == "needs-search-extension" and state["all_workspaces_pruned"]
    assert state["request_sha256"] == original_hash and state["declared_controls"] == list(CONTROLS)
    assert [job["name"] for job in state["jobs"]] == [job["name"] for job in original["jobs"]] == list(CONTROLS)
    assert extra["status"] == "complete" and extra["all_workspaces_pruned"]
    assert extra["request_sha256"] == extension_hash and extra["name"] == CONTROLS[0]
    for expected in request["predecessor_tasks"]:
        actual = queue["tasks"][str(expected["id"])]
        assert actual["status"] == {"Done": "Success"}, "Dense worker is not successful"
        assert actual["original_command"] == expected["command"]
        assert all(actual[key] == expected[key] for key in ("created_at", "group", "label", "dependencies"))
    original_proofs = []
    for job, completed in zip(original["jobs"], state["jobs"]):
        assert completed["output"] == job["output"]
        original_proofs.append(proof(job, completed, request_sha256=original_hash, software=original["software"], records=12126))
    assert len(extension["jobs"]) == 1 and extension["jobs"][0]["name"] == CONTROLS[0]
    extension_proof = proof(extension["jobs"][0], extra, request_sha256=extension_hash,
        software=original["software"], records=4042)
    assert extension_proof["previous"]["directory"] == original["jobs"][0]["output"]
    assert extension_proof["previous"]["verification_sha256"] == state["jobs"][0]["verification_sha256"]
    assert extension_proof["previous"]["calibration_sha256"] == state["jobs"][0]["calibration_sha256"]
    assert extension_proof["previous"]["verified_records"] == 12126
    assert extension_proof["covered_learning_rates"] == [0.001, 0.003, 0.01, 0.03]
    assert extension_proof["source_labels"] == original_proofs[0]["source_labels"] == original_proofs[1]["source_labels"]
    selected = [extension_proof, original_proofs[1]]
    for item in selected:
        assert all(value["at_upper_rate_boundary"] is False for value in item["best_rates_by_epoch_budget"].values()), "Public ADE search edge remains unresolved"
    return {"status": "verified", "verified_models": 2, "all_workspaces_pruned": True,
        "original_state_sha256": sha(request["original_state"]), "extension_state_sha256": sha(request["extension_state"]),
        "original_records": 24252, "extension_records": 4042,
        "best_rates_by_epoch_budget": dict(zip(CONTROLS, (item["best_rates_by_epoch_budget"] for item in selected))),
        "verified_at_utc": now()}


SCIENTIFIC_CHECK = """import json,sys
from pathlib import Path
from bench.storage import read_json,file_digest
from verify_ade_calibration import verify
q=read_json(Path(sys.argv[1]));original=read_json(q['original_request']);extension=read_json(q['extension_request']);checks=[]
for job in original['jobs']:
 p=Path(job['output']);actual=verify(p,expected_model=job['model'],software=original['software'],feature_shape=job['feature_shape'],baseline=Path(job['baseline_output']),path_mapping=original['path_mapping'],calibration_helper=q['original_helper']);saved=read_json(p/'verification.json')
 assert all(saved[k]==v for k,v in actual.items() if k!='verifier_sha256')
 checks.append({'name':job['name'],'records':actual['verified_records'],'verification_sha256':file_digest(p/'verification.json')})
job=extension['jobs'][0];p=Path(job['output']);actual=verify(p,expected_model=job['model'],software=original['software'],feature_shape=job['feature_shape'],baseline=Path(job['baseline_output']),path_mapping=original['path_mapping'],rate_multipliers=(30,),calibration_helper=q['extension_helper'],previous=extension['previous_directory'],previous_calibration_helper=q['original_helper']);saved=read_json(p/'verification.json')
assert all(saved[k]==v for k,v in actual.items())
checks.append({'name':job['name'],'role':'extension','records':actual['verified_records'],'verification_sha256':file_digest(p/'verification.json')})
print(json.dumps({'status':'verified','checks':checks,'verifier_sha256':file_digest(q['verifier'])}))
"""


def checked(request):
    assert all(sha(path) == expected for path, expected in request["pinned_files"].items()), "Transfer input changed"
    core = runpy.run_path(str(RUNNER / "c3_acceptance.py"))
    original = read(request["original_request"])
    assert [job["name"] for job in original["jobs"]] == list(CONTROLS)
    for job in original["jobs"]:
        core["public_reference"](job)
    return core, original


def prepare():
    assert not REQUEST.exists() and not RUNNER.exists()
    original_path = SCOPE / "ade-search/request.json"
    execution = read(SCOPE / "current-execution.json")
    extension_path = SCOPE / execution["ade_extension_request"]
    original, extension = read(original_path), read(extension_path)
    identities = {item["id"]: item for item in execution["tasks"]}
    pins = {**original["pinned_files"], **extension["pinned_files"]}
    assert all(sha(path) == expected for path, expected in pins.items())
    RUNNER.mkdir(parents=True)
    capacity_plan = SCOPE / "family-plan-v2/plans/full-imagenet-family-plan.json"
    source_core = SCOPE / "runner/c3_acceptance.py"
    text = source_core.read_text()
    marker = 'read(ROOT / "outputs/acceptance/full-imagenet-family-plan.json")["jobs"]'
    assert text.count(marker) == 1
    revised = text.replace(marker, 'read(Path(' + repr(str(capacity_plan)) + '))["jobs"]')
    (RUNNER / "c3_acceptance.py").write_text(revised)
    shutil.copyfile(__file__, RUNNER / "public_imagenet_transfer.py")
    audit = HANDOFF / "dataset-preflight/with-metadata-overlays"
    source_request = read(audit / "plan.json")
    imagenet = next(item for item in source_request["datasets"] if item["name"] == "imagenet")
    checksum_list = (audit / imagenet["list"]).resolve()
    assert sha(checksum_list) == imagenet["list_sha256"]
    for path in [original_path, extension_path, capacity_plan, source_core,
                 audit / "plan.json", audit / "missing-imagenet-size.json",
                 audit / "missing-imagenet-files.txt.gz", checksum_list, *RUNNER.rglob("*")]:
        if path.is_file():
            pins[str(path)] = sha(path)
    caller = Path(extension["package_source_root"]) / "caller"
    request = {"status": "prepared", "scope_revision": original["scope_revision"], "created_at_utc": now(),
        "original_request": str(original_path), "original_request_sha256": sha(original_path),
        "extension_request": str(extension_path), "extension_request_sha256": sha(extension_path),
        "original_state": str(SCOPE / "ade-search/state.json"), "extension_state": str(extension_path.with_name("state.json")),
        "predecessor_tasks": [identities[37], identities[execution["ade_extension_task_id"]]], "pinned_files": pins,
        "capacity_plan": str(capacity_plan), "original_helper": str(Path(original["package_source_root"]) / "caller/calibrate_monitor_probes.py"),
        "extension_helper": str(caller / "calibrate_monitor_probes.py"), "verifier": str(caller / "verify_ade_calibration.py"),
        "missing_images": 1154478, "missing_image_bytes": 132411896262, "source_images_to_checksum": 1281167,
        "scope": "Existing source-copy protocol and full train/development checksum audit, with combined public ADE coverage/cleanup. Uses the public ViT plan's capacity estimate. No final partition is read."}
    save(REQUEST, request)
    save(DIRECTORY / "source-audit-v3.json", {"status": "verified", "request_sha256": sha(REQUEST),
        "original_core_sha256": sha(source_core), "transfer_core_sha256": sha(RUNNER / "c3_acceptance.py"),
        "core_derivation": "Change only the capacity-estimate plan path to the public-reference family plan. Replace the cleanup callback at runtime with the independently verified combined boundary.",
        "missing_list_sha256": sha(audit / "missing-imagenet-files.txt.gz"), "source_checksum_list_sha256": sha(checksum_list)})
    checked(request)
    print(REQUEST)


def runtime_check(core, original):
    job = original["jobs"][0]
    runtime = core["probe"]([job["command"][0], "-c", core["ENV_PROBE"], json.dumps(job["environment_versions"])], core["environment"](job))
    assert runtime["binary_sha256"] == read(ROOT / "outputs/environments/setup-receipt.json")["environments"]["bench"]["python_binary_sha256"]
    return runtime


def validate(request, core, original):
    core["boundary_ready"]()
    runtime = runtime_check(core, original)
    assert not (RUN / "imagenet-transfer/receipt.json").exists()
    save(VALIDATION, {"status": "passed", "request_sha256": sha(REQUEST),
        "runtime": runtime, "pinned_files_verified": len(request["pinned_files"]), "checked_at_utc": now(),
        "scope": "Static source/copy inputs, public checkpoint provenance and verified runtime only. Dense work remains live; combined scientific verification and capacity checks must pass in the held worker before copying."})


def transfer(request, core, original):
    assert sha(__file__) == os.environ["PUBLIC_IMAGENET_RUNNER_SHA256"]
    assert sha(REQUEST) == os.environ["PUBLIC_IMAGENET_REQUEST_SHA256"]
    validated = read(VALIDATION)
    assert validated["status"] == "passed" and validated["request_sha256"] == sha(REQUEST)
    save(DIRECTORY / "execution-runtime.json", runtime_check(core, original))
    queue = json.loads(subprocess.check_output(["pueue", "status", "--json"], text=True))
    boundary = combined_boundary(request, queue)
    job = original["jobs"][0]
    env = core["environment"](job)
    env["PYTHONPATH"] = os.pathsep.join([str(Path(request["verifier"]).parent), *job["pythonpath"]])
    scientific = core["probe"]([job["command"][0], "-c", SCIENTIFIC_CHECK, str(REQUEST)], env)
    save(DIRECTORY / "scientific-preflight.json", scientific)
    assert boundary["original_state_sha256"] == sha(request["original_state"])
    assert boundary["extension_state_sha256"] == sha(request["extension_state"])
    boundary["scientific_preflight_sha256"] = sha(DIRECTORY / "scientific-preflight.json")
    boundary["request_sha256"] = sha(REQUEST)
    save(DIRECTORY / "dense-boundary.json", boundary)
    core["transfer_imagenet"].__globals__["ade_cleanup_boundary"] = lambda: boundary
    core["transfer_imagenet"]()
    receipt = RUN / "imagenet-transfer/receipt.json"
    assert read(receipt)["status"] == "complete" and read(receipt)["source_images_verified"] == 1281167
    save(DIRECTORY / "completion.json", {"status": "complete", "request_sha256": sha(REQUEST),
        "dense_boundary_sha256": sha(DIRECTORY / "dense-boundary.json"), "transfer_receipt_sha256": sha(receipt),
        "final_partitions_read": False, "completed_at_utc": now()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "validate", "transfer"))
    action = parser.parse_args().action
    if action == "prepare":
        prepare()
    else:
        request = read(REQUEST)
        core, original = checked(request)
        if action == "validate":
            validate(request, core, original)
        else:
            transfer(request, core, original)
