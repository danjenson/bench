from copy import deepcopy
from pathlib import Path
import runpy

import pytest


@pytest.fixture
def transfer_boundary(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/public_imagenet_transfer.py"))
    save, read, sha = (module[key] for key in ("save", "read", "sha"))
    software = {"package.py": "unchanged"}
    jobs = [{"name": name, "output": str(tmp_path / name), "model": {"identity": name},
        "execution": {"device": "cuda", "batch_size": 32, "workers": 8}} for name in module["CONTROLS"]]
    new_job = {**jobs[0], "output": str(tmp_path / "extension")}
    original_request, extension_request = tmp_path / "original.json", tmp_path / "extension.json"
    save(original_request, {"jobs": jobs, "software": software})
    save(extension_request, {"jobs": [new_job]})

    def make_proof(job, records, request_hash):
        output = Path(job["output"])
        save(output / "plan.json", {"model": job["model"], "software": software,
            "numerics": "fp32-v1", "execution": job["execution"]})
        results, variants = {}, {}
        for index in range(records // 2021):
            relative = f"statistics/{index}.json"
            save(output / relative, {"record": index})
            results[str(index)] = {"statistics": {"file": relative, "sha256": sha(output / relative)}}
            variants[str(index)] = {"statistics_sha256": sha(output / relative)}
        save(output / "calibration.json", {"status": "complete", "retained_temporary_bytes": 0,
            "plan_sha256": sha(output / "plan.json"), "results": results})
        save(output / "verification.json", {"status": "verified", "request_sha256": request_hash,
            "plan_sha256": sha(output / "plan.json"), "calibration_sha256": sha(output / "calibration.json"),
            "verified_records": records, "variants": variants,
            "source_labels": {"development_images": 2021, "verified_source_files": 4043, "eligible_pixels": 373568791},
            "best_rates_by_epoch_budget": {budget: {"selected_learning_rate": 0.003, "score": 1.0,
                "at_upper_rate_boundary": False} for budget in ("40", "80")}})
        return {"status": "complete", "name": job["name"], "output": job["output"],
            "calibration_sha256": sha(output / "calibration.json"), "verification_sha256": sha(output / "verification.json")}

    completed = [make_proof(job, 12126, sha(original_request)) for job in jobs]
    extra = make_proof(new_job, 4042, sha(extension_request))
    extra_proof = Path(new_job["output"]) / "verification.json"
    verified = read(extra_proof)
    verified.update(previous={"directory": jobs[0]["output"], "calibration_sha256": completed[0]["calibration_sha256"],
        "verification_sha256": completed[0]["verification_sha256"], "verified_records": 12126},
        covered_learning_rates=[0.001, 0.003, 0.01, 0.03])
    save(extra_proof, verified)
    extra.update(verification_sha256=sha(extra_proof), all_workspaces_pruned=True, request_sha256=sha(extension_request))
    original_state, extension_state = tmp_path / "original-state.json", tmp_path / "extension-state.json"
    save(original_state, {"status": "needs-search-extension", "all_workspaces_pruned": True,
        "request_sha256": sha(original_request), "declared_controls": list(module["CONTROLS"]), "jobs": completed})
    save(extension_state, extra)
    predecessors = [{"id": identifier, "created_at": str(identifier) + "-original-birth", "command": str(identifier) + "-command",
        "group": "acceptance", "label": str(identifier), "dependencies": [] if identifier == 37 else [37]} for identifier in (37, 44)]
    queue = {"tasks": {str(task["id"]): {**task, "original_command": task["command"], "status": {"Done": "Success"}} for task in predecessors}}
    request = {"original_request": str(original_request), "original_request_sha256": sha(original_request),
        "extension_request": str(extension_request), "extension_request_sha256": sha(extension_request),
        "original_state": str(original_state), "extension_state": str(extension_state), "predecessor_tasks": predecessors}
    return module, request, queue, jobs, new_job


def test_combined_copy_boundary_preserves_prior_evidence_and_checks_both_models(transfer_boundary):
    module, request, queue, _, _ = transfer_boundary
    result = module["combined_boundary"](request, queue)
    assert result["all_workspaces_pruned"] and result["verified_models"] == 2
    assert result["original_records"] == 24252 and result["extension_records"] == 4042
    for task_id, mutation in [("37", {"status": "Running"}), ("44", {"status": {"Done": {"Failed": 1}}}),
                              ("44", {"created_at": "reused-id"}), ("44", {"original_command": "unrelated-worker"})]:
        altered = deepcopy(queue)
        altered["tasks"][task_id].update(mutation)
        with pytest.raises(AssertionError):
            module["combined_boundary"](request, altered)


@pytest.mark.parametrize("mutation", ["vit-upper", "extension-upper", "retained-vit", "retained-extension",
    "wrong-prior-proof", "changed-statistics", "changed-model", "changed-original-request", "unfinished-extension"])
def test_combined_copy_boundary_rejects_unresolved_or_changed_evidence(transfer_boundary, mutation):
    module, request, queue, jobs, new_job = transfer_boundary
    read, save, sha = (module[key] for key in ("read", "save", "sha"))
    if mutation in ("retained-vit", "retained-extension"):
        output = Path(jobs[1]["output"] if mutation == "retained-vit" else new_job["output"])
        (output / ".work").mkdir()
    elif mutation == "changed-statistics":
        save(Path(new_job["output"]) / "statistics/0.json", {"changed": True})
    elif mutation == "changed-model":
        path = Path(new_job["output"]) / "plan.json"
        plan = read(path)
        plan["model"] = {"identity": "another-model"}
        save(path, plan)
    elif mutation == "changed-original-request":
        value = read(request["original_request"])
        save(request["original_request"], {**value, "changed": True})
    elif mutation == "unfinished-extension":
        value = read(request["extension_state"])
        save(request["extension_state"], {**value, "status": "running"})
    else:
        output = Path(jobs[1]["output"] if mutation == "vit-upper" else new_job["output"])
        value = read(output / "verification.json")
        if mutation == "wrong-prior-proof":
            value["previous"]["verification_sha256"] = "changed-prior"
        else:
            value["best_rates_by_epoch_budget"]["80"]["at_upper_rate_boundary"] = True
        save(output / "verification.json", value)
        state_path = request["original_state"] if mutation == "vit-upper" else request["extension_state"]
        state = read(state_path)
        target = state["jobs"][1] if mutation == "vit-upper" else state
        target["verification_sha256"] = sha(output / "verification.json")
        save(state_path, state)
    with pytest.raises(AssertionError):
        module["combined_boundary"](request, queue)
