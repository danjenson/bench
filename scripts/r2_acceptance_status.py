"""Read the owned r2 queue and validate linked acceptance evidence without CUDA.

This audits the staged workloads, not the separate scientific release gates in
docs/acceptance.md. It never resumes jobs, refits probes or rewrites receipts.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTION = ROOT / "outputs/r2-acceptance-20261026/current-execution.json"
TASK_IDENTITY = ("created_at", "original_command", "group", "label", "dependencies")


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def owned(path, directory):
    path, directory = Path(path).resolve(), Path(directory).resolve()
    require(path != directory and path.is_relative_to(directory), f"Path outside owned directory: {path}")
    return path


def completed_job(job, record, plan_sha256, run_root):
    output = owned(job["output"], run_root / "results")
    require(record["output"] == job["output"], f"Output changed: {job['name']}")
    require(record["command"] == job["command"], f"Command changed: {job['name']}")
    require(not os.path.lexists(output / ".work"), f"Retained workspace: {output / '.work'}")
    artifact = owned(record["artifact"], output)
    require(sha(artifact) == record["artifact_sha256"], f"Artifact changed: {artifact}")
    verification = output / "verification.json"
    require(sha(verification) == record["verification_sha256"], f"Verification changed: {verification}")
    proofs = record["verification_files"]
    require(proofs.get(str(verification)) == record["verification_sha256"], f"Missing primary verification pin: {verification}")
    for filename, expected in proofs.items():
        path = owned(filename, output)
        require(sha(path) == expected, f"Verification changed: {path}")
    require(artifact.name in ("results.json", "calibration.json"), f"Unsupported artifact: {artifact}")
    evidence = read(verification)
    evidence_plan = sha(output / "plan.json") if artifact.name == "calibration.json" else plan_sha256
    require(evidence["status"] == "verified" and evidence["plan_sha256"] == evidence_plan,
            f"Verification belongs to a different plan: {verification}")
    artifact_field = "calibration_sha256" if artifact.name == "calibration.json" else "results_sha256"
    require(evidence[artifact_field] == record["artifact_sha256"], f"Verification belongs to a different artifact: {verification}")
    metadata = read(output / job.get("model_metadata_file", "plan.json"))
    require(metadata["model"] == job["model"] == read(job["expected_model"]), f"Model changed: {job['name']}")
    result = read(artifact)
    require(result["status"] == "complete", f"Incomplete artifact: {artifact}")
    if artifact.name == "results.json":
        require(result["storage"]["retained_temporary_bytes"] == 0
                and result["storage"]["retained_workspace_bytes"] == 0, f"Retained temporary bytes: {artifact}")
        tasks = result["tasks"]
    else:
        require(result["retained_temporary_bytes"] == 0, f"Retained temporary bytes: {artifact}")
        require(result["plan_sha256"] == evidence_plan, f"Calibration plan changed: {artifact}")
        tasks = result["results"]
    # Check compact record bytes too, so an edit cannot leave the rollup green.
    require(bool(tasks), f"Missing task records: {artifact}")
    for task in tasks.values():
        statistics = task["statistics"]
        path = owned(output / statistics["file"], output)
        require(sha(path) == statistics["sha256"], f"Statistics changed: {path}")
    return {"name": job["name"], "status": "verified-commit", "artifact": str(artifact),
            "artifact_sha256": record["artifact_sha256"], "verification_sha256": record["verification_sha256"],
            "workspace_absent": True}


def recovered_job(entry, task, state_path, plan_path, plan, run_root):
    require(task["status"] == {"Done": {"Failed": 1}}, "Recovery requires the preserved original failed task")
    require(len(plan["jobs"]) == 1, "Recovery must identify one original job")
    request_path = owned(entry["request"], run_root)
    effective_path = owned(entry["effective_state"], request_path.parent)
    proof_path = owned(entry["verification"], request_path.parent)
    for path, key in [(request_path, "request_sha256"), (effective_path, "effective_state_sha256"), (proof_path, "verification_sha256")]:
        require(sha(path) == entry[key], f"Recovery evidence changed: {path}")
    request, effective, proof = read(request_path), read(effective_path), read(proof_path)
    plan_sha, state_sha = sha(plan_path), sha(state_path)
    require(request["original_plan"] == str(plan_path) and request["original_plan_sha256"] == plan_sha,
            "Recovery belongs to a different original plan")
    require(request["original_state"] == str(state_path) and request["original_state_sha256"] == state_sha,
            "Recovery belongs to a different original state")
    require(sha(request_path.parent / "original-state.json") == state_sha, "Original recovery state archive changed")
    original_task = request["original_task"]
    require(original_task["id"] == task["id"] and all(original_task[key] == task[key] for key in TASK_IDENTITY)
            and original_task["status"] == task["status"], "Recovery task identity changed")
    for filename, expected in request["pinned_files"].items():
        require(sha(filename) == expected, f"Recovery pinned input changed: {filename}")
    require(effective["status"] == "complete" and effective["all_workspaces_pruned"]
            and effective["recovery_request_sha256"] == entry["request_sha256"]
            and effective["original_plan_sha256"] == plan_sha and effective["original_state_sha256"] == state_sha
            and effective["original_task_status"] == task["status"], "Invalid effective recovery state")
    require(proof["status"] == "verified" and proof["verified_records"] == effective["verified_records"] > 0,
            "Recovery record verification changed")
    job = plan["jobs"][0]
    require(request["output"] == job["output"] and effective["verification_sha256"] == entry["verification_sha256"],
            "Recovery belongs to a different output")
    primary = Path(job["output"]) / "verification.json"
    record = {**job, "status": "complete", "artifact": effective["artifact"],
              "artifact_sha256": effective["artifact_sha256"], "verification_sha256": entry["verification_sha256"],
              "verification_files": {str(primary): entry["verification_sha256"]}}
    result = completed_job(job, record, plan_sha, run_root)
    result.update(status="verified-after-recovery", recovery_request_sha256=entry["request_sha256"],
                  original_queue_status=task["status"], verified_records=effective["verified_records"])
    return result


def audit(execution_path, queue):
    execution_path = Path(execution_path).resolve()
    registry, run_root = read(execution_path), execution_path.parent
    recovery_catalog = run_root / "verification-recoveries.json"
    recoveries = {}
    if recovery_catalog.exists():
        catalog = read(recovery_catalog)
        require(catalog["execution_registry_sha256"] == sha(execution_path), "Recovery catalog belongs to another execution")
        recoveries = catalog["tasks"]
    group = registry["group"]
    require(group in queue["groups"], f"Missing queue group: {group}")
    require(queue["groups"][group]["parallel_tasks"] == 1, "Acceptance group parallelism changed")
    require(len(registry["tasks"]) == len(registry["plans"]) > 0, "Task/plan registry is incomplete")
    ids = {str(task["id"]) for task in registry["tasks"]}
    require(len(ids) == len(registry["tasks"]), "Duplicate registered task ID")
    require(set(recoveries) <= ids, "Recovery catalog contains an unregistered task")
    require({key for key, task in queue["tasks"].items() if task["group"] == group} == ids,
            "Owned queue membership changed; inspect the live task identities")
    reports, pins, failed = [], {}, False
    for task, entry in zip(registry["tasks"], registry["plans"]):
        live = queue["tasks"][str(task["id"])]
        require(all(live[key] == task[key] for key in TASK_IDENTITY), f"Queue task identity changed: {task['id']}")
        plan_path = owned(entry["plan"], run_root / "plans")
        require(sha(plan_path) == entry["plan_sha256"], f"Plan changed: {plan_path}")
        plan = read(plan_path)
        require(plan["gpu_uuid"] == registry["gpu_uuid"], f"GPU pin changed: {plan_path}")
        require(str(plan_path) in task["original_command"], f"Queue command does not name its plan: {task['id']}")
        for filename, expected in plan["pinned_files"].items():
            if filename in pins:
                require(pins[filename] == expected, f"Conflicting pin: {filename}")
            else:
                require(sha(filename) == expected, f"Pinned input changed: {filename}")
                pins[filename] = expected
        # Workers repeat preflight and atomically replace validation receipts.
        # Validate their plan binding, not the historical receipt's byte hash.
        validation = read(plan_path.with_suffix(".validation.json"))
        require(validation["status"] == "passed" and validation["plan_sha256"] == entry["plan_sha256"],
                f"Invalid preflight: {plan_path}")
        state_path = owned(plan["state"], run_root / "states")
        report = {"task_id": task["id"], "label": task["label"], "queue_status": live["status"],
                  "plan": str(plan_path), "plan_sha256": entry["plan_sha256"], "jobs": []}
        state = read(state_path) if state_path.exists() else None
        report["phase_status"] = state["status"] if state else "not-started-or-preflighting"
        phase_failed = False
        if state:
            require(state["plan_sha256"] == entry["plan_sha256"], f"State belongs to a different plan: {state_path}")
            records = state["jobs"]
            require(len(records) <= len(plan["jobs"]), f"Unexpected jobs: {state_path}")
            for job, record in zip(plan["jobs"], records):
                require(job["name"] == record["name"], f"Job order changed: {state_path}")
                if record["status"] == "complete":
                    report["jobs"].append(completed_job(job, record, entry["plan_sha256"], run_root))
                else:
                    report["jobs"].append({"name": job["name"], "status": record["status"]})
            if state["status"] == "complete":
                require(len(records) == len(plan["jobs"]) and state["all_workspaces_pruned"]
                        and all(job["status"] == "verified-commit" for job in report["jobs"]),
                        f"Incomplete completion evidence: {state_path}")
            if state["status"] == "failed":
                report["error"] = state.get("error")
                phase_failed = True
        success = live["status"] == {"Done": "Success"}
        recovered = False
        if str(task["id"]) in recoveries:
            require(state is not None and state["status"] == "failed", "Recovery must retain the original failed phase")
            report["jobs"] = [recovered_job(recoveries[str(task["id"])], live, state_path, plan_path, plan, run_root)]
            report["effective_status"] = "complete-after-recovery"
            recovered = True
        if success:
            require(state is not None and state["status"] == "complete", f"Queue success without completed evidence: {state_path}")
        elif not recovered and isinstance(live["status"], dict) and "Done" in live["status"]:
            failed = True
        failed = failed or (phase_failed and not recovered)
        report["complete"] = success or recovered
        reports.append(report)
    complete = all(report["complete"] for report in reports)
    return {"checked_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed" if failed else "workloads-complete" if complete else "incomplete",
            "execution_registry_sha256": sha(execution_path), "group": queue["groups"][group],
            "unique_static_inputs_verified": len(pins), "workloads_complete": complete,
            "workloads": reports, "release_requirements": "docs/acceptance.md#remaining-release-requirements",
            "scope": "Queue identity, static pins, preflight and completed artifact/verification/statistics hashes and cleanup. Does not replay score verifiers or establish scientific calibration, numerical qualification, monitoring acceptance or v1 readiness."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution", type=Path, default=DEFAULT_EXECUTION)
    parser.add_argument("--require-complete", action="store_true", help="Exit 1 until the staged workloads complete; this is not a v1 release gate")
    args = parser.parse_args()
    try:
        registry = read(args.execution)
        queue = json.loads(subprocess.check_output(["pueue", "-c", registry["pueue_config"], "status", "--json"], text=True))
        report = audit(args.execution, queue)
    except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "invalid", "error": str(error)}))
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return int(report["status"] == "failed" or (args.require_complete and not report["workloads_complete"]))


if __name__ == "__main__":
    raise SystemExit(main())
