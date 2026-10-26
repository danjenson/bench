"""Install the handoff, detach the owned paused daemon, and export receipts."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path("/home/scratch/danson/repos/bench")
RUN = ROOT / "outputs/r2-acceptance-20261026"
MIGRATION = ROOT / "outputs/r2-migration-20261026"
GROUP = "bench-r2-acceptance"


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    payload = json.load(sys.stdin)
    config = RUN / "pueue/config.yml"
    base = ["pueue", "-c", str(config)]
    old = json.loads(subprocess.check_output([*base, "status", "--json"], text=True))
    assert old["groups"][GROUP]["status"] == "Paused"
    assert set(old["tasks"]) == {"0", "1", "2"}
    assert all(t["status"] == "Queued" and t["start"] is None for t in old["tasks"].values())
    for name, record in payload["files"].items():
        p = Path(name)
        assert not p.is_absolute() and ".." not in p.parts
        assert p.parts[0] in ("docs", "scripts", "outputs")
        assert hashlib.sha256(record["text"].encode()).hexdigest() == record["sha256"]
        path = ROOT / p
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(record["text"])
        assert sha(path) == record["sha256"]
    # The first daemon inherited the staging SSH channel. No task has started;
    # restart only this private daemon with explicit detached log descriptors.
    subprocess.run([*base, "shutdown"], input="y\n", check=True, capture_output=True, text=True)
    for _ in range(40):
        probe = subprocess.run([*base, "status", "--json"], capture_output=True, text=True)
        if probe.returncode:
            break
        time.sleep(0.25)
    assert probe.returncode != 0, "Private daemon has not stopped"
    with (config.parent / "daemon.log").open("a") as log:
        subprocess.run(["pueued", "-c", str(config), "-d"], check=True,
                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(40):
        probe = subprocess.run([*base, "status", "--json"], capture_output=True, text=True)
        if probe.returncode == 0:
            break
        time.sleep(0.25)
    assert probe.returncode == 0, probe.stderr
    current = json.loads(probe.stdout)
    assert current["groups"][GROUP] == {"status": "Paused", "parallel_tasks": 1}
    keys = ("created_at", "original_command", "group", "label", "dependencies", "status", "start", "end")
    for task_id, task in old["tasks"].items():
        assert all(current["tasks"][task_id][key] == task[key] for key in keys)
    registry_path = RUN / "current-execution.json"
    registry = json.loads(registry_path.read_text())
    for task in registry["tasks"]:
        assert all(current["tasks"][str(task["id"])][key] == task[key] for key in keys)
    registry["c3_waiting_tasks_archived_and_removed"] = [56, 57, 58, 59]
    registry["c3_archive_sha256"] = payload["c3_archive_sha256"]
    registry["scope"] = "Remaining public-reference evaluations are migrated and validated on r2; the isolated group is paused, no benchmark task has started, and superseded c3 waiting tasks are archived and removed."
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    final = {"status": "complete-paused", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "queue_identity_preserved": True, "evaluations_started": False,
        "c3_waiting_tasks_removed": [56, 57, 58, 59], "private_daemon_stdio_detached": True,
        "documents": {n: v["sha256"] for n, v in payload["files"].items() if n.startswith("docs/")}}
    (RUN / "migration-final-check.json").write_text(json.dumps(final, indent=2, sort_keys=True) + "\n")
    files = [RUN / "current-execution.json", RUN / "migration-final-check.json", RUN / "relocation-validation.json",
             RUN / "origin-dense-verification.json", RUN / "gpu-runtime-probe.json", RUN / "imagenet-source-audit/receipt.json",
             MIGRATION / "copy-receipt.json", MIGRATION / "source-audit.json",
             *(p.with_suffix(".validation.json") for p in sorted((RUN / "plans").glob("*-plan.json")))]
    receipts = {str(p.relative_to(ROOT)): {"sha256": sha(p), "data": json.loads(p.read_text())} for p in files}
    for entry in registry["plans"]:
        p = Path(entry["plan"])
        assert sha(p) == entry["plan_sha256"]
        plan = json.loads(p.read_text())
        assert all(sha(Path(n)) == h for n, h in plan["pinned_files"].items())
    print(json.dumps({"status": "complete-paused", "host": "clpc35.cs.ox.ac.uk", "receipts": receipts,
                      "queue": {"groups": {GROUP: current["groups"][GROUP]}, "tasks": registry["tasks"]}}), flush=True)


if __name__ == "__main__":
    main()
