from copy import deepcopy
from pathlib import Path
import runpy

import pytest


@pytest.fixture
def recovery_boundary(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/recover_public_ade_verification.py"))
    save, sha = (module[key] for key in ("save", "sha"))
    jobs = []
    for name in module["CONTROLS"]:
        output = tmp_path / name
        model = {"identity": name}
        save(output / "plan.json", {"model": model, "software": {"same": "source"},
            "numerics": "fp32-v1", "stage": "search-boundary", "variants": list(range(6))})
        save(output / "calibration.json", {"status": "complete", "retained_temporary_bytes": 0,
            "plan_sha256": sha(output / "plan.json"), "results": {str(i): {} for i in range(6)}})
        jobs.append({"name": name, "output": str(output), "model": model})
    original = tmp_path / "request.json"
    save(original, {"jobs": jobs, "software": {"same": "source"}})
    state = tmp_path / "state.json"
    save(state, {"status": "failed", "request_sha256": sha(original), "declared_controls": list(module["CONTROLS"])})
    expected = {"id": 37, "command": "frozen-original-worker", "created_at": "original-birth",
        "group": "acceptance", "dependencies": [], "label": "original-ade"}
    request = {"original_request": str(original), "original_request_sha256": sha(original),
        "original_state": str(state), "original_task": expected}
    queue = {"tasks": {"37": {**expected, "original_command": expected["command"], "status": {"Done": {"Failed": 1}}}}}
    return module, request, queue, jobs


def test_recovery_requires_terminal_complete_data_before_independent_science(recovery_boundary):
    module, request, queue, _ = recovery_boundary
    for status in ({"Done": {"Failed": 1}}, {"Done": "Success"}):
        queue["tasks"]["37"]["status"] = status
        result = module["completed_boundary"](request, queue)
        assert result["all_workspaces_pruned"] and len(result["reports"]) == 2
        assert result["original_task_status"] == status
    for mutation in ({"status": "Running"}, {"status": {"Stashed": {"enqueue_at": None}}},
                     {"status": {"Done": "Killed"}}, {"status": {"Done": {"Failed": 2}}},
                     {"created_at": "reused-id"}, {"original_command": "different-worker"}):
        altered = deepcopy(queue)
        altered["tasks"]["37"].update(mutation)
        with pytest.raises(AssertionError):
            module["completed_boundary"](request, altered)


@pytest.mark.parametrize("mutation", ["retained-work", "incomplete-fit", "missing-variant", "changed-plan", "changed-request", "live-state"])
def test_recovery_cannot_turn_missing_or_changed_fit_evidence_into_completion(recovery_boundary, mutation):
    module, request, queue, jobs = recovery_boundary
    read, save = (module[key] for key in ("read", "save"))
    output = Path(jobs[1]["output"])
    if mutation == "retained-work":
        (output / ".work").mkdir()
    elif mutation == "changed-plan":
        plan = read(output / "plan.json")
        plan["model"] = {"identity": "other"}
        save(output / "plan.json", plan)
    elif mutation in ("changed-request", "live-state"):
        state = read(request["original_state"])
        state.update({"request_sha256": "changed"} if mutation == "changed-request" else {"status": "running"})
        save(request["original_state"], state)
    else:
        report = read(output / "calibration.json")
        if mutation == "incomplete-fit":
            report["status"] = "running"
        else:
            report["results"].pop("5")
        save(output / "calibration.json", report)
    with pytest.raises(AssertionError):
        module["completed_boundary"](request, queue)
