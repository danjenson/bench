from copy import deepcopy
from pathlib import Path
import runpy

import pytest


@pytest.fixture
def continuation(tmp_path):
    module = runpy.run_path(str(Path(__file__).parents[1] / "scripts/c3_acceptance.py"))
    namespace = module["boundary_ready"].__globals__
    namespace.update(ROOT=tmp_path, HANDOFF=tmp_path / "handoff", RUN=tmp_path / "run")
    return namespace


def test_original_failed_boundary_requires_revalidated_evidence(continuation):
    c = continuation
    h = c["HANDOFF"]
    c["save"](h / "ghost-boundary.json", {"action": "queue-ended-before-stop", "task": {"status": {"Done": {"Failed": 1}}}})
    c["save"](h / "transfer-receipt.json", {"status": "complete"})
    with pytest.raises(FileNotFoundError):
        c["boundary_ready"]()


def test_revalidated_boundary_checks_linked_bytes(continuation):
    c = continuation
    h, root = c["HANDOFF"], c["ROOT"]
    c["save"](h / "ghost-boundary.json", {"action": "queue-ended-before-stop", "task": {"status": {"Done": {"Failed": 1}}}})
    c["save"](h / "ghost-ade-failed-queue-state.json", {"status": "failed"})
    result = root / "outputs/acceptance/full-probes/ade-family/resnet18-pretrained/results.json"
    c["save"](result, {"status": "complete"})
    c["save"](h / "verification.json", {"status": "verified", "records": 6063, "pinned_inputs_verified": 82})
    corrected = {"status": "complete", "action": "model-verified-after-resumption-accounting-fix", "evaluation_exit_code": 0,
                 "remaining_models_started_on_ghost": False, "verification": "verification.json",
                 "verification_sha256": c["sha"](h / "verification.json"), "results_sha256": c["sha"](result),
                 "original_boundary_receipt_sha256": c["sha"](h / "ghost-boundary.json"),
                 "original_queue_state_sha256": c["sha"](h / "ghost-ade-failed-queue-state.json")}
    c["save"](h / "ghost-boundary-revalidated.json", corrected)
    c["save"](h / "transfer-receipt.json", {"status": "complete", "boundary_sha256": c["sha"](h / "ghost-boundary-revalidated.json")})
    c["boundary_ready"]()
    result.write_text('{"status":"altered"}')
    with pytest.raises(AssertionError):
        c["boundary_ready"]()


@pytest.mark.parametrize("status,pruned", [("running", False), ("failed", False), ("complete", False)])
def test_imagenet_copy_refuses_incomplete_ade(continuation, status, pruned):
    c = continuation
    plan = c["RUN"] / "plans/full-ade-family-plan.json"
    state = c["RUN"] / "states/ade.json"
    c["save"](plan, {"state": str(state), "jobs": []})
    c["save"](state, {"status": status, "all_workspaces_pruned": pruned})
    with pytest.raises(AssertionError):
        c["ade_cleanup_boundary"]()


def test_path_relocation_preserves_state_and_changes_only_declared_identity(continuation):
    c = continuation
    original = {"state_digest": "same-weight-bytes", "info": {"recipe": {"checkpoint": "/ghost/checkpoint.pt"}},
                "numerics": "fp32-v1", "digest": "unused"}
    original = c["reidentify"](original)
    relocated = c["reidentify"](c["replace"](original, {"/ghost": "/c3"}))
    expected = deepcopy(original)
    expected["info"]["recipe"]["checkpoint"] = "/c3/checkpoint.pt"
    expected["digest"] = c["digest"]({k: v for k, v in expected.items() if k != "digest"})
    assert relocated == expected and relocated["digest"] != original["digest"]
    assert original["info"]["recipe"]["checkpoint"] == "/ghost/checkpoint.pt"


def test_output_mapping_does_not_capture_similarly_named_historical_artifacts(continuation):
    transform = continuation["replace"]
    mapping = {"/ghost/seam": "/c3/new-seam"}
    assert transform("/ghost/seam/acceptance.json", mapping) == "/c3/new-seam/acceptance.json"
    assert transform("/ghost/seam-cpu-check/acceptance.json", mapping) == "/ghost/seam-cpu-check/acceptance.json"
    assert transform('output_dir="/ghost/seam"', mapping) == 'output_dir="/c3/new-seam"'


def test_prospective_comparisons_cannot_write_into_historical_evidence(continuation):
    c = continuation
    run = c["RUN"]
    plan = {"state": str(run / "states/queue.json"),
            "comparison_output": str(c["ROOT"] / "outputs/acceptance/comparison.json"),
            "jobs": [{"name": "model", "output": str(run / "results/model"),
                      "log": str(run / "logs/model.log")}]}
    with pytest.raises(AssertionError, match="comparison_output"):
        c["validate_execution_outputs"](plan)
    plan["comparison_output"] = str(run / "results/comparison.json")
    c["validate_execution_outputs"](plan)
    historical = c["ROOT"] / "outputs/acceptance"
    historical.mkdir(parents=True)
    (run / "results").mkdir(parents=True)
    (run / "results/escape").symlink_to(historical, target_is_directory=True)
    plan["comparison_output"] = str(run / "results/escape/comparison.json")
    with pytest.raises(AssertionError, match="comparison_output"):
        c["validate_execution_outputs"](plan)


def test_imagenet_copy_refuses_a_retained_ade_workspace(continuation):
    c = continuation
    plan_path = c["RUN"] / "plans/full-ade-family-plan.json"
    state_path = c["RUN"] / "states/ade.json"
    jobs = [{"name": str(i), "output": str(c["RUN"] / "results" / str(i))} for i in range(5)]
    c["save"](plan_path, {"state": str(state_path), "jobs": jobs})
    c["save"](state_path, {"status": "complete", "all_workspaces_pruned": True,
        "plan_sha256": c["sha"](plan_path), "jobs": [{"name": j["name"], "status": "complete"} for j in jobs]})
    (Path(jobs[0]["output"]) / ".work").mkdir(parents=True)
    with pytest.raises(AssertionError):
        c["ade_cleanup_boundary"]()


def test_reference_evaluation_requires_the_declared_public_checkpoint(continuation):
    c = continuation
    checkpoint = c["ROOT"] / "weights.pt"
    checkpoint.write_bytes(b"official-fixture-weights")
    reference = {"architecture": "resnet18", "sha256": c["sha"](checkpoint),
                 "url": "https://download.pytorch.org/models/fixture.pth"}
    c["save"](c["ROOT"] / "docs/reference-models.json", {"pretrained_references": [reference]})
    job = {"model": {"info": {"recipe": {**reference, "checkpoint": str(checkpoint)}}}}
    assert c["public_reference"](job) == reference
    for change in ({"architecture": "prototype"}, {"url": "https://example.org/private.pt"}):
        altered = deepcopy(job)
        altered["model"]["info"]["recipe"].update(change)
        with pytest.raises(AssertionError, match="supported public reference"):
            c["public_reference"](altered)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(AssertionError, match="checkpoint differs"):
        c["public_reference"](job)


def test_unqualified_reference_is_rejected_before_model_construction(continuation, monkeypatch):
    c = continuation
    c["save"](c["ROOT"] / "docs/reference-models.json", {"pretrained_references": []})
    plan = c["RUN"] / "plans/prototype.json"
    c["save"](plan, {"pinned_files": {}, "jobs": [{"model": {"info": {"recipe": {"architecture": "prototype"}}}}]})
    monkeypatch.setitem(c, "probe", lambda *args: pytest.fail("An excluded model must never be constructed"))
    with pytest.raises(AssertionError, match="supported public reference"):
        c["validate_family"](plan, models_only=True)
