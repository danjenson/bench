from copy import deepcopy
from pathlib import Path
import runpy

import pytest


@pytest.fixture
def controller(tmp_path):
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/public_ade_acceptance.py"))
    c = module["release_command"].__globals__
    c.update(DIRECTORY=tmp_path, REQUEST=tmp_path / "request.json")
    c["save"](c["REQUEST"], {"scope": "public-reference-fixture"})
    jobs = []
    for name in c["CONTROLS"]:
        output = tmp_path / name
        c["save"](output / "calibration.json", {"retained_temporary_bytes": 0})
        c["save"](output / "verification.json", {"status": "verified", "best_rates_by_epoch_budget": {"40": {"at_upper_rate_boundary": False}}})
        jobs.append({"name": name, "output": str(output), "status": "complete",
            "calibration_sha256": c["sha"](output / "calibration.json"),
            "verification_sha256": c["sha"](output / "verification.json")})
    state = {"status": "complete", "all_workspaces_pruned": True,
        "baseline_cleanup_boundary": {"all_workspaces_pruned": True},
        "declared_controls": list(c["CONTROLS"]), "jobs": jobs, "request_sha256": c["sha"](c["REQUEST"])}
    task = {"id": 38, "created_at": "unique-creation", "group": "acceptance", "command": "exact-transfer-command", "dependencies": [37]}
    queue = {"tasks": {"38": {**task, "original_command": task["command"], "status": {"Stashed": {"enqueue_at": None}}}}}
    c["save"](tmp_path / "state.json", state)
    return c, state, {"task": task}, queue


@pytest.mark.parametrize("status", ["Stashed", {"Stashed": {"enqueue_at": None}}])
def test_copy_requires_exact_held_queue_identity_and_complete_dense_evidence(controller, status):
    c, state, hold, queue = controller
    queue["tasks"]["38"]["status"] = status
    assert c["release_command"](state, hold, queue) == ["pueue", "enqueue", "38"]
    for change in ({"created_at": "reused-id"}, {"original_command": "different-command"},
                   {"dependencies": []}, {"group": "other"}, {"status": "Queued"},
                   {"status": {"Stashed": {"enqueue_at": "tomorrow"}}}):
        altered = deepcopy(queue)
        altered["tasks"]["38"].update(change)
        with pytest.raises(AssertionError):
            c["release_command"](state, hold, altered)


def test_unresolved_search_or_retained_work_cannot_release_image_copy(controller):
    c, state, hold, queue = controller
    output = Path(state["jobs"][0]["output"])
    c["save"](output / "verification.json", {"status": "verified", "best_rates_by_epoch_budget": {"80": {"at_upper_rate_boundary": True}}})
    state["jobs"][0]["verification_sha256"] = c["sha"](output / "verification.json")
    c["save"](c["DIRECTORY"] / "state.json", state)
    with pytest.raises(AssertionError):
        c["release_command"](state, hold, queue)
    c["save"](output / "verification.json", {"status": "verified", "best_rates_by_epoch_budget": {"80": {"at_upper_rate_boundary": False}}})
    state["jobs"][0]["verification_sha256"] = c["sha"](output / "verification.json")
    c["save"](c["DIRECTORY"] / "state.json", state)
    (output / ".work").mkdir()
    with pytest.raises(AssertionError):
        c["release_command"](state, hold, queue)


def test_changed_artifact_or_uncommitted_state_cannot_release_image_copy(controller):
    c, state, hold, queue = controller
    altered = {**state, "status": "needs-search-extension"}
    with pytest.raises(AssertionError, match="state changed"):
        c["release_command"](altered, hold, queue)
    c["save"](Path(state["jobs"][0]["output"]) / "calibration.json", {"retained_temporary_bytes": 1})
    with pytest.raises(AssertionError):
        c["release_command"](state, hold, queue)
