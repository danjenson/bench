"""Run with the copied verified Python on r2; validate and stage a paused queue."""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import subprocess
import sys
import time

ROOT = Path("/home/scratch/danson/repos/bench")
MIGRATION = ROOT / "outputs/r2-migration-20261026"
ORIGIN = ROOT / "outputs/c3-acceptance-20261023"
SCOPE = ORIGIN / "public-reference-scope-20261024"
RUN = ROOT / "outputs/r2-acceptance-20261026"
NAMES = ("full-imagenet-family-plan.json", "full-spair-family-plan.json", "clevr-joint-search-plan.json")
GROUP = "bench-r2-acceptance"


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def save(p, value):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_suffix(p.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(p)


def emit(stage, **kwargs):
    print(json.dumps({"stage": stage, "at_utc": now(), **kwargs}), flush=True)


def main():
    # The bootstrap is a separate live SSH job; do not touch its incomplete data.
    while True:
        if (MIGRATION / "bootstrap-failure.json").exists():
            raise RuntimeError("Bootstrap failed: " + str(read(MIGRATION / "bootstrap-failure.json")))
        if (MIGRATION / "source-audit.json").exists():
            progress = read(MIGRATION / "source-audit.json")
            if progress["status"] == "complete":
                break
            assert progress["status"] != "failed", "Source audit has failures; inspect before proceeding"
        emit("waiting-for-source-copy-and-audit", copied=(MIGRATION / "copy-receipt.json").exists())
        time.sleep(30)
    request = read(MIGRATION / "request.json")
    sources = read(MIGRATION / "source-audit.json")
    assert sources["status"] == "complete" and sources["request_sha256"] == sha(MIGRATION / "request.json")
    assert sources["final_partitions_read"] is False
    assert subprocess.check_output(["hostname"], text=True).strip() == request["destination_host"]
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
    assert gpu == [request["r2_gpu_uuid"]]
    assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
    assert not RUN.exists(), "Inspect an existing r2 run rather than overwriting it"
    RUN.mkdir()
    sys.path.insert(0, str(SCOPE / "imagenet-transfer-v2/runner-v4"))
    transfer = runpy.run_path(str(SCOPE / "imagenet-transfer-v2/runner-v4/public_imagenet_transfer.py"))
    transfer_request = read(SCOPE / "imagenet-transfer-v2/request-v4.json")
    origin_core, original = transfer["checked"](transfer_request)
    runtime = transfer["runtime_check"](origin_core, original)
    assert runtime["binary_sha256"] == read(ROOT / "outputs/environments/setup-receipt.json")["environments"]["bench"]["python_binary_sha256"]
    gpu_probe = """import json,torch
assert torch.cuda.is_available() and torch.cuda.get_device_capability()==(8,9)
assert torch.cuda.is_bf16_supported()
for dtype in (torch.float32,torch.bfloat16):
 a=torch.ones((32,32),device='cuda',dtype=dtype)
 assert torch.all(a@a==32).item()
torch.cuda.synchronize()
p=torch.cuda.get_device_properties(0)
print(json.dumps({'status':'passed','device':p.name,'capability':list(torch.cuda.get_device_capability()),'total_memory':p.total_memory,'fp32_and_bf16_matrix_smoke':True,'scope':'CUDA runtime smoke only, not numerical-profile or performance qualification.'}))
"""
    gpu_runtime = origin_core["probe"]([original["jobs"][0]["command"][0], "-c", gpu_probe], origin_core["environment"](original["jobs"][0]))
    save(RUN / "gpu-runtime-probe.json", gpu_runtime)
    origin_queue = read(MIGRATION / "origin-queue.json")
    assert origin_queue["origin_host"] == request["origin_host"]
    boundary = transfer["combined_boundary"](transfer_request, origin_queue)
    env = origin_core["environment"](original["jobs"][0])
    env["PYTHONPATH"] = os.pathsep.join([str(Path(transfer_request["verifier"]).parent), *original["jobs"][0]["pythonpath"]])
    scientific = origin_core["probe"]([original["jobs"][0]["command"][0], "-c", transfer["SCIENTIFIC_CHECK"],
                                        str(SCOPE / "imagenet-transfer-v2/request-v4.json")], env)
    save(RUN / "origin-dense-verification.json", {"status": "verified", "at_utc": now(), "origin_host": request["origin_host"],
        "origin_queue_sha256": sha(MIGRATION / "origin-queue.json"), "boundary": boundary, "scientific": scientific,
        "runtime": runtime, "scope": "Archived completed c3 task identities plus fresh r2 reconstruction of all 28,294 records. No c3 task is represented as an r2 task."})
    emit("origin-science-and-environment-verified", records=28294)

    image_root = str(ROOT / "outputs/c3-source-overlays/imagenet")
    imagenet = next(s for s in sources["sources"] if s["root"] == image_root)
    assert imagenet["files_verified"] == 1_281_167 and not imagenet["mismatches"]
    save(RUN / "imagenet-source-audit/receipt.json", {"status": "complete", "completed_at_utc": now(),
        "mode": "existing-r2-source-checksum-verification", "source_images_verified": 1_281_167,
        "source_audit_sha256": sha(MIGRATION / "source-audit.json"), "images_transferred": 0,
        "existing_dataset": request["existing_r2_imagenet"], "logical_source_root": image_root,
        "final_partitions_read": False})
    (RUN / "data").symlink_to(ORIGIN / "data", target_is_directory=True)
    runner_dir = RUN / "runner"
    runner_dir.mkdir()
    parent_runner = SCOPE / "runner/c3_acceptance.py"
    text = parent_runner.read_text()
    before = 'RUN = ROOT / "outputs/c3-acceptance-20261023"'
    assert text.count(before) == 1
    text = text.replace(before, 'RUN = ROOT / "outputs/r2-acceptance-20261026"')
    marker = 'RUN / "imagenet-transfer/receipt.json"'
    assert text.count(marker) == 1
    text = text.replace(marker, 'RUN / "imagenet-source-audit/receipt.json"')
    runner = runner_dir / "r2_acceptance.py"
    runner.write_text(text)
    compile(text, str(runner), "exec")
    core = runpy.run_path(str(runner))
    core["boundary_ready"]()
    core_hash = sha(runner)
    source_plans = [SCOPE / "family-plan-v2/plans" / n for n in NAMES]
    plans = [RUN / "plans" / n for n in NAMES]
    mapping = {str(ORIGIN / "data"): str(RUN / "data"),
               str(ORIGIN / "states/public-references-20261024"): str(RUN / "states"),
               str(ORIGIN / "results/public-references-20261024"): str(RUN / "results"),
               str(ORIGIN / "logs/public-references-20261024"): str(RUN / "logs"),
               str(parent_runner): str(runner)}
    for src, dst in zip(source_plans, plans, strict=True):
        mapping[str(src)] = str(dst)
        mapping[read(src)["package_source_root"]] = str(RUN / "sources" / Path(read(src)["package_source_root"]).name)
    snapshots = []
    for src, dst in zip(source_plans, plans, strict=True):
        original_plan = read(src)
        source_root = Path(original_plan["package_source_root"])
        new_source = Path(mapping[str(source_root)])
        shutil.copytree(source_root, new_source, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        plan = core["replace"](deepcopy(original_plan), mapping)
        plan.update(gpu_uuid=request["r2_gpu_uuid"], runner=str(runner), runner_sha256=core_hash,
                    host_relocation={"origin_host": request["origin_host"], "host": request["destination_host"],
                        "parent_plan": str(src), "parent_plan_sha256": sha(src), "created_at_utc": now()},
                    scope="Public reference development acceptance relocated to r2. Scientific recipes, models, budgets and tolerances are unchanged; hardware and execution paths are recorded separately.")
        if "c3_predecessor" in plan:
            previous = Path(plan["c3_predecessor"]["plan"])
            plan["c3_predecessor"]["plan_sha256"] = sha(previous)
        pins = {}
        for name, expected in plan["pinned_files"].items():
            path = Path(name)
            actual = sha(path)
            assert actual == (core_hash if path == runner else expected), name
            pins[name] = actual
        plan["pinned_files"] = pins
        for evidence in (MIGRATION / "source-audit.json", MIGRATION / "origin-queue.json",
                         RUN / "origin-dense-verification.json", RUN / "imagenet-source-audit/receipt.json",
                         RUN / "gpu-runtime-probe.json"):
            plan["pinned_files"][str(evidence)] = sha(evidence)
        for job in plan["jobs"]:
            assert not Path(job["output"]).exists()
            assert not Path(job["log"]).exists()
        assert not Path(plan["state"]).exists()
        core["validate_execution_outputs"](plan)
        save(dst, plan)
        core["validate_family"](dst)
        validation = read(dst.with_suffix(".validation.json"))
        assert validation["status"] == "passed" and validation["plan_sha256"] == sha(dst)
        snapshots.append({"plan": str(dst), "plan_sha256": sha(dst), "parent_plan_sha256": sha(src),
                          "validation_sha256": sha(dst.with_suffix(".validation.json")), "jobs": len(plan["jobs"]),
                          "static_pins": len(pins), "source_files_unchanged": True})
        emit("family-preflight-passed", plan=dst.name, jobs=len(plan["jobs"]))
    save(RUN / "relocation-validation.json", {"status": "passed", "at_utc": now(), "plans": snapshots,
        "runner": str(runner), "runner_sha256": core_hash, "parent_runner_sha256": sha(parent_runner),
        "runner_changes": ["r2-owned execution root", "verified existing ImageNet availability receipt path"],
        "source_audit_sha256": sha(MIGRATION / "source-audit.json"), "origin_dense_verification_sha256": sha(RUN / "origin-dense-verification.json"),
        "scope": "Fresh r2 CPU model/readout/environment contracts and complete train/dev source checks. No GPU evaluation or new performance/parity qualification."})

    # Use a private daemon so an unrelated stopped default queue cannot wake up.
    config = RUN / "pueue/config.yml"
    config.parent.mkdir(parents=True)
    state_dir, runtime_dir = config.parent / "state", config.parent / "runtime"
    state_dir.mkdir()
    runtime_dir.mkdir()
    config.write_text("shared:\n  pueue_directory: " + str(state_dir) + "\n  runtime_directory: " + str(runtime_dir)
                      + "\n  unix_socket_path: " + str(runtime_dir / "pueue.socket")
                      + "\n  pid_path: " + str(runtime_dir / "pueued.pid") + "\n  use_unix_socket: true\n")
    with (config.parent / "daemon.log").open("a") as log:
        subprocess.run(["pueued", "-c", str(config), "-d"], check=True,
                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    base = ["pueue", "-c", str(config)]
    for _ in range(20):
        ready = subprocess.run([*base, "status", "--json"], capture_output=True, text=True)
        if ready.returncode == 0:
            break
        time.sleep(0.25)
    assert ready.returncode == 0, ready.stderr
    assert not json.loads(ready.stdout)["tasks"]
    subprocess.run([*base, "group", "add", GROUP], check=True)
    subprocess.run([*base, "pause", "--group", GROUP], check=True)
    ids = []
    for plan in plans:
        command = shlex.join(["env", "PYTHONDONTWRITEBYTECODE=1", "C3_RUNNER_SHA256=" + core_hash,
                              str(ROOT / "outputs/environments/bench/bin/python"), "-u", str(runner), "phase", str(plan)])
        add = [*base, "add", "--print-task-id", "--working-directory", str(ROOT), "--group", GROUP,
               "--label", "r2-public-" + plan.stem]
        if ids:
            add += ["--after", str(ids[-1])]
        add += ["--", command]
        ids.append(int(subprocess.check_output(add, text=True).strip()))
    queue = json.loads(subprocess.check_output([*base, "status", "--json"], text=True))
    assert queue["groups"][GROUP]["status"] == "Paused" and queue["groups"][GROUP]["parallel_tasks"] == 1
    assert all(queue["tasks"][str(i)]["status"] == "Queued" and queue["tasks"][str(i)]["start"] is None for i in ids)
    tasks = [{"id": i, **{k: queue["tasks"][str(i)][k] for k in
              ("created_at", "original_command", "group", "label", "dependencies", "status", "start", "end")}} for i in ids]
    save(RUN / "current-execution.json", {"status": "ready-paused", "host": request["destination_host"],
        "checked_at_utc": now(), "gpu_uuid": request["r2_gpu_uuid"], "pueue_config": str(config),
        "group": GROUP, "tasks": tasks, "plans": snapshots,
        "resume_command": shlex.join([*base, "start", "--group", GROUP]),
        "scope": "Remaining benchmark evaluations migrated and validated; private queue remains paused. c3 waiting tasks must be archived/removed after this receipt is checked."})
    emit("r2-ready-paused", ids=ids, config=str(config), evaluations_started=False)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if RUN.exists():
            save(RUN / "migration-failure.json", {"status": "failed", "at_utc": now(), "error": str(error)})
        raise
