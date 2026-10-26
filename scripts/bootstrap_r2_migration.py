"""Run on r2: copy the frozen transfer set and checksum all train/dev inputs."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import gzip
import hashlib
import json
import os
import shutil
import subprocess

ROOT = Path("/home/scratch/danson/repos/bench")
RUN = ROOT / "outputs/r2-migration-20261026"
ID = "bench-r2-migration-20261026"


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def save(name, value):
    path = RUN / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def emit(stage, **kwargs):
    print(json.dumps({"stage": stage, "at_utc": now(), **kwargs}), flush=True)


def main():
    owner = ROOT / ".r2-migration-owner.json"
    if ROOT.exists():
        assert owner.exists() and json.loads(owner.read_text())["id"] == ID, "Existing r2 checkout is not owned by this migration"
    else:
        ROOT.mkdir(parents=True)
        owner.write_text(json.dumps({"id": ID, "created_at_utc": now()}) + "\n")
    RUN.mkdir(parents=True, exist_ok=True)
    assert shutil.disk_usage(ROOT).free > 75_000_000_000
    host = subprocess.check_output(["hostname"], text=True).strip()
    assert host == "clpc35.cs.ox.ac.uk"
    assert subprocess.check_output(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "c3", "hostname"], text=True).strip() == "clpc122.cs.ox.ac.uk"
    emit("copy-started", root=str(ROOT))
    ssh = "ssh -o BatchMode=yes -o ConnectTimeout=12"
    subprocess.run(["rsync", "-a", "-e", ssh, "c3:" + str(RUN) + "/files-from.txt", str(RUN / "files-from.txt")], check=True)
    with (RUN / "rsync.log").open("a") as log:
        subprocess.run(["rsync", "-ar", "--partial", "--stats", "--exclude=.work/", "--exclude=__pycache__/",
                        "--files-from=" + str(RUN / "files-from.txt"), "-e", ssh, "c3:" + str(ROOT) + "/", str(ROOT) + "/"],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    request = json.loads((RUN / "request.json").read_text())
    assert sha(RUN / "files-from.txt") == request["files_from_sha256"]
    assert sha(RUN / "expected-source-files.json.gz") == request["expected_sources_sha256"]
    assert all(sha(p) == h for p, h in request["static_pins"].items())
    save("copy-receipt.json", {"status": "complete", "completed_at_utc": now(), "static_pins": len(request["static_pins"]),
                               "request_sha256": sha(RUN / "request.json"), "rsync_log_sha256": sha(RUN / "rsync.log")})
    emit("copy-complete", static_pins=len(request["static_pins"]))
    overlay = ROOT / "outputs/c3-source-overlays/imagenet"
    original = Path(request["existing_r2_imagenet"])
    assert original.is_dir() and (original / "train").is_dir()
    if overlay.exists() or overlay.is_symlink():
        assert overlay.is_symlink() and overlay.resolve() == original.resolve()
    else:
        overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay.symlink_to(original, target_is_directory=True)
    with gzip.open(RUN / "expected-source-files.json.gz", "rt") as f:
        sources = json.load(f)
    summaries = []
    for root, group in sources.items():
        digest = hashlib.sha256()
        count = total_bytes = 0
        errors = []
        entries = sorted(group["files"].items())
        emit("source-check-started", root=root, files=len(entries))
        def verify(item):
            relative, expected = item
            path = Path(root) / relative
            try:
                actual = sha(path)
                return relative, expected, actual, path.stat().st_size, None
            except OSError as error:
                return relative, expected, None, 0, str(error)
        with ThreadPoolExecutor(max_workers=8) as pool:
            # Consume bounded chunks so Executor.map cannot queue 1.3M futures.
            for offset in range(0, len(entries), 2048):
                for relative, expected, actual, size, error in pool.map(verify, entries[offset:offset + 2048]):
                    count += 1
                    total_bytes += size
                    digest.update((relative + "\0" + expected + "\n").encode())
                    if error or actual != expected:
                        errors.append({"file": relative, "expected": expected, "actual": actual, "error": error})
                if offset and offset % (2048 * 25) == 0:
                    emit("source-check-progress", root=root, checked=count, total=len(entries), errors=len(errors))
        record = {"root": root, "files_verified": count, "bytes_read": total_bytes, "expected_records_sha256": digest.hexdigest(),
                  "mismatches": errors, "tasks": group["tasks"]}
        summaries.append(record)
        save("source-audit.json", {"status": "failed" if errors else "running", "checked_at_utc": now(), "sources": summaries, "final_partitions_read": False})
        assert not errors, f"Source checksum failures under {root}: {len(errors)}; existing data was not modified"
        emit("source-check-complete", root=root, files=count, bytes_read=total_bytes)
    save("source-audit.json", {"status": "complete", "completed_at_utc": now(), "sources": summaries,
                               "request_sha256": sha(RUN / "request.json"), "final_partitions_read": False,
                               "existing_imagenet_reused_readonly": str(original)})
    emit("sources-verified", unique_files=sum(s["files_verified"] for s in summaries), evaluations_started=False)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if RUN.exists():
            save("bootstrap-failure.json", {"status": "failed", "at_utc": now(), "error": str(error)})
        raise
