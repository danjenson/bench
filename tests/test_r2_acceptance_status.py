from copy import deepcopy
import json
from pathlib import Path
import runpy

import pytest


@pytest.fixture
def evidence(tmp_path):
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/r2_acceptance_status.py"))
    root = tmp_path / "run"

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    source = tmp_path / "frozen.py"
    source.write_text("# pinned source\n")
    model = tmp_path / "model.json"
    write(model, {"digest": "model"})
    plan_path, state_path = root / "plans/family.json", root / "states/family.json"
    output = root / "results/model"
    job = {"name": "model", "output": str(output), "command": ["python", "frozen.py"],
           "model": {"digest": "model"}, "expected_model": str(model), "model_metadata_file": "manifest.json"}
    plan = {"jobs": [job], "state": str(state_path), "gpu_uuid": "gpu", "pinned_files": {str(source): module["sha"](source)}}
    write(plan_path, plan)
    plan_sha = module["sha"](plan_path)
    write(plan_path.with_suffix(".validation.json"), {"status": "passed", "plan_sha256": plan_sha})
    task = {"id": 0, "created_at": "birth", "original_command": f"python worker.py {plan_path}",
            "group": "acceptance", "label": "family", "dependencies": [], "status": "Queued"}
    registry = {"group": "acceptance", "gpu_uuid": "gpu", "tasks": [task],
                "plans": [{"plan": str(plan_path), "plan_sha256": plan_sha}]}
    execution = root / "current-execution.json"
    write(execution, registry)
    queue = {"groups": {"acceptance": {"status": "Running", "parallel_tasks": 1}}, "tasks": {"0": deepcopy(task)}}

    def complete(calibration=False):
        write(output / "manifest.json", {"model": job["model"]})
        statistics = output / "statistics/task.json"
        write(statistics, {"records": [1, 2]})
        tasks = {"task": {"statistics": {"file": "statistics/task.json", "sha256": module["sha"](statistics)}}}
        if calibration:
            write(output / "plan.json", {"model": job["model"], "variants": ["task"]})
            artifact = output / "calibration.json"
            proof_plan = module["sha"](output / "plan.json")
            result = {"status": "complete", "results": tasks, "plan_sha256": proof_plan, "retained_temporary_bytes": 0}
        else:
            artifact = output / "results.json"
            proof_plan = plan_sha
            result = {"status": "complete", "tasks": tasks,
                      "storage": {"retained_temporary_bytes": 0, "retained_workspace_bytes": 0}}
        write(artifact, result)
        verification = output / "verification.json"
        field = "calibration_sha256" if calibration else "results_sha256"
        write(verification, {"status": "verified", "plan_sha256": proof_plan, field: module["sha"](artifact)})
        record = {**job, "status": "complete", "artifact": str(artifact), "artifact_sha256": module["sha"](artifact),
                  "verification_sha256": module["sha"](verification), "verification_files": {str(verification): module["sha"](verification)}}
        state = {"status": "complete", "plan_sha256": plan_sha, "all_workspaces_pruned": True, "jobs": [record]}
        write(state_path, state)
        queue["tasks"]["0"]["status"] = {"Done": "Success"}
        return state

    return {**module, **locals()}


def test_running_preflight_and_refreshed_validation_do_not_claim_completion(evidence):
    e = evidence
    e["queue"]["tasks"]["0"]["status"] = "Running"
    e["write"](e["plan_path"].with_suffix(".validation.json"), {"status": "passed", "plan_sha256": e["plan_sha"], "at_utc": "fresh"})
    report = e["audit"](e["execution"], e["queue"])
    assert report["status"] == "incomplete" and not report["workloads_complete"]
    assert report["workloads"][0]["phase_status"] == "not-started-or-preflighting"


@pytest.mark.parametrize("field,value", [("created_at", "reused-id"), ("dependencies", [7]), ("original_command", "other worker")])
def test_reused_or_changed_queue_identity_is_rejected(evidence, field, value):
    e = evidence
    e["queue"]["tasks"]["0"][field] = value
    with pytest.raises(e["AuditError"], match="identity changed"):
        e["audit"](e["execution"], e["queue"])


def test_unrelated_group_is_ignored_and_unregistered_owned_job_is_rejected(evidence):
    e = evidence
    extra = {**e["task"], "id": 1, "group": "default"}
    e["queue"]["tasks"]["1"] = extra
    assert e["audit"](e["execution"], e["queue"])["status"] == "incomplete"
    extra["group"] = "acceptance"
    with pytest.raises(e["AuditError"], match="membership changed"):
        e["audit"](e["execution"], e["queue"])


def test_pinned_source_change_is_rejected(evidence):
    e = evidence
    e["source"].write_text("changed source")
    with pytest.raises(e["AuditError"], match="Pinned input changed"):
        e["audit"](e["execution"], e["queue"])


def test_scheduler_success_cannot_replace_completion_evidence(evidence):
    e = evidence
    e["queue"]["tasks"]["0"]["status"] = {"Done": "Success"}
    with pytest.raises(e["AuditError"], match="without completed evidence"):
        e["audit"](e["execution"], e["queue"])


@pytest.mark.parametrize("calibration", [False, True])
def test_verified_result_and_calibration_use_their_respective_plan_bindings(evidence, calibration):
    e = evidence
    e["complete"](calibration)
    report = e["audit"](e["execution"], e["queue"])
    assert report["status"] == "workloads-complete" and report["workloads_complete"]
    assert report["workloads"][0]["jobs"][0]["workspace_absent"]
    assert "v1 readiness" in report["scope"]


@pytest.mark.parametrize("calibration", [False, True])
def test_altered_statistics_invalidate_completed_evidence(evidence, calibration):
    e = evidence
    e["complete"](calibration)
    (e["output"] / "statistics/task.json").write_text("changed records")
    with pytest.raises(e["AuditError"], match="Statistics changed"):
        e["audit"](e["execution"], e["queue"])


@pytest.mark.parametrize("dangling", [False, True])
def test_retained_workspace_or_dangling_link_invalidates_completion(evidence, dangling):
    e = evidence
    e["complete"]()
    workspace = e["output"] / ".work"
    if dangling:
        workspace.symlink_to("missing-workspace")
    else:
        workspace.mkdir()
    with pytest.raises(e["AuditError"], match="Retained workspace"):
        e["audit"](e["execution"], e["queue"])


def test_failed_worker_is_reported_without_claiming_completion(evidence):
    e = evidence
    e["write"](e["state_path"], {"status": "failed", "plan_sha256": e["plan_sha"], "jobs": [], "error": "failed preflight"})
    e["queue"]["tasks"]["0"]["status"] = {"Done": {"Failed": 1}}
    report = e["audit"](e["execution"], e["queue"])
    assert report["status"] == "failed" and not report["workloads_complete"]
    assert report["workloads"][0]["error"] == "failed preflight"


def attach_recovery(e):
    completed = e['complete'](calibration=True)
    record = completed['jobs'][0]
    proof = e['read'](e['output'] / 'verification.json')
    proof['verified_records'] = 2
    e['write'](e['output'] / 'verification.json', proof)
    raw = {**completed, 'status': 'failed', 'error': 'original exact checker failed',
           'jobs': [{**record, 'status': 'running'}]}
    e['write'](e['state_path'], raw)
    e['queue']['tasks']['0']['status'] = {'Done': {'Failed': 1}}
    directory = e['root'] / 'recovery'
    e['write'](directory / 'original-state.json', raw)
    e['write'](directory / 'verification.json', proof)
    request = directory / 'request.json'
    e['write'](request, {'original_plan': str(e['plan_path']), 'original_plan_sha256': e['plan_sha'],
        'original_state': str(e['state_path']), 'original_state_sha256': e['sha'](e['state_path']),
        'original_task': deepcopy(e['queue']['tasks']['0']), 'output': str(e['output']),
        'pinned_files': {str(e['source']): e['sha'](e['source'])}})
    effective = directory / 'effective-state.json'
    e['write'](effective, {'status': 'complete', 'all_workspaces_pruned': True,
        'recovery_request_sha256': e['sha'](request), 'original_plan_sha256': e['plan_sha'],
        'original_state_sha256': e['sha'](e['state_path']), 'original_task_status': {'Done': {'Failed': 1}},
        'artifact': record['artifact'], 'artifact_sha256': record['artifact_sha256'],
        'verification_sha256': e['sha'](directory / 'verification.json'), 'verified_records': 2})
    catalog = {'execution_registry_sha256': e['sha'](e['execution']), 'tasks': {'0': {
        'request': str(request), 'request_sha256': e['sha'](request),
        'effective_state': str(effective), 'effective_state_sha256': e['sha'](effective),
        'verification': str(directory / 'verification.json'), 'verification_sha256': e['sha'](directory / 'verification.json')}}}
    e['write'](e['root'] / 'verification-recoveries.json', catalog)
    return catalog


def test_linked_recovery_completes_evidence_while_preserving_the_original_failure(evidence):
    e = evidence
    attach_recovery(e)
    before = e['state_path'].read_bytes()
    report = e['audit'](e['execution'], e['queue'])
    assert report['workloads_complete'] and report['status'] == 'workloads-complete'
    row = report['workloads'][0]
    assert row['queue_status'] == {'Done': {'Failed': 1}} and row['phase_status'] == 'failed'
    assert row['effective_status'] == 'complete-after-recovery'
    assert row['jobs'][0]['status'] == 'verified-after-recovery'
    assert e['state_path'].read_bytes() == before


@pytest.mark.parametrize('change', ['state', 'archive', 'statistics', 'effective-binding'])
def test_recovery_cannot_hide_changed_original_or_result_evidence(evidence, change):
    e = evidence
    catalog = attach_recovery(e)
    if change == 'state':
        e['state_path'].write_text('{}')
    elif change == 'archive':
        (e['root'] / 'recovery/original-state.json').write_text('{}')
    elif change == 'statistics':
        (e['output'] / 'statistics/task.json').write_text('{}')
    else:
        path = e['root'] / 'recovery/effective-state.json'
        value = e['read'](path)
        value['original_state_sha256'] = 'wrong'
        e['write'](path, value)
        catalog['tasks']['0']['effective_state_sha256'] = e['sha'](path)
        e['write'](e['root'] / 'verification-recoveries.json', catalog)
    with pytest.raises((e['AuditError'], KeyError)):
        e['audit'](e['execution'], e['queue'])
