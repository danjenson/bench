from copy import deepcopy
from pathlib import Path
import runpy

import pytest


def test_extension_waits_for_exact_original_worker_and_cleanup(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    controller = runpy.run_path(str(Path(__file__).parents[1] / "scripts/extend_public_ade.py"))
    save, sha, boundary = (controller[key] for key in ("save", "sha", "predecessor_boundary"))
    names = ["resnet18-pretrained", "vit_b_16-pretrained"]
    jobs = []
    for name in names:
        output = tmp_path / name
        save(output / "calibration.json", {"retained_temporary_bytes": 0})
        save(output / "verification.json", {"status": "verified"})
        jobs.append({"name": name, "output": str(output), "status": "complete",
            "calibration_sha256": sha(output / "calibration.json"),
            "verification_sha256": sha(output / "verification.json")})
    original_request = tmp_path / "original-request.json"
    save(original_request, {"jobs": jobs})
    state_path = tmp_path / "original-state.json"
    state = {"status": "needs-search-extension", "all_workspaces_pruned": True,
        "declared_controls": names, "jobs": jobs, "unresolved_upper_boundaries": names,
        "request_sha256": sha(original_request)}
    save(state_path, state)
    original = {"id": 37, "created_at": "original-creation", "group": "acceptance", "dependencies": [],
        "label": "original", "command": "original-command"}
    held = {"id": 38, "created_at": "held-creation", "group": "acceptance", "dependencies": [37],
        "label": "transfer", "command": "transfer-command"}
    request = {"original_dense_state": str(state_path), "original_dense_request": str(original_request),
        "original_dense_request_sha256": sha(original_request), "original_dense_task": original,
        "held_transfer_task": held}
    queue = {"tasks": {"37": {**original, "original_command": original["command"], "status": {"Done": "Success"}},
        "38": {**held, "original_command": held["command"], "status": {"Stashed": {"enqueue_at": None}}}}}
    assert boundary(request, queue)["all_workspaces_pruned"]
    request.pop("held_transfer_task")
    queue["tasks"].pop("38")
    assert boundary(request, queue)["all_workspaces_pruned"]
    for task_id, change in [("37", {"status": "Running"}), ("37", {"status": {"Done": {"Failed": 1}}}),
                           ("37", {"created_at": "reused-id"}), ("37", {"dependencies": [999]})]:
        altered = deepcopy(queue)
        altered["tasks"][task_id].update(change)
        with pytest.raises(AssertionError):
            boundary(request, altered)
    altered = {**state, "status": "running"}
    save(state_path, altered)
    with pytest.raises(AssertionError):
        boundary(request, queue)
    save(state_path, state)
    workspace = Path(jobs[0]["output"]) / ".work"
    workspace.mkdir()
    with pytest.raises(AssertionError):
        boundary(request, queue)
    workspace.rmdir()
    save(Path(jobs[0]["output"]) / "verification.json", {"status": "different"})
    with pytest.raises(AssertionError):
        boundary(request, queue)
