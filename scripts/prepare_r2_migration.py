"""Freeze a bounded, model-independent transfer list for the r2 handoff."""

from pathlib import Path
from datetime import datetime, timezone
import gzip
import hashlib
import json
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/c3-acceptance-20261023"
SCOPE = SOURCE / "public-reference-scope-20261024"
DEST = ROOT / "outputs/r2-migration-20261026"


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def prepare():
    DEST.mkdir(exist_ok=True)
    paths = set()
    tracked = subprocess.check_output(["git", "ls-files", "-co", "--exclude-standard", "-z"], cwd=ROOT).split(b"\0")
    for raw in tracked:
        if not raw:
            continue
        p = Path(raw.decode())
        if (ROOT / p).is_file() and not any(x in p.parts for x in (".aws", ".codex", ".agents", ".work", "outputs", ".venv")):
            paths.add(p.as_posix())
    paths.update((".git", "outputs/environments/bench", "outputs/environments/python",
                  "outputs/environments/setup-receipt.json", str(SCOPE.relative_to(ROOT)),
                  "outputs/c3-acceptance-20261023/data", "outputs/acceptance/c3-handoff-20261023"))
    request_paths = [SCOPE / "ade-search/request.json", SCOPE / "ade-search/recovery-verification/request.json",
                     SCOPE / "ade-extensions/resnet18-lr0p03/request-v3.json", SCOPE / "imagenet-transfer-v2/request-v4.json",
                     *(SCOPE / "family-plan-v2/plans" / n for n in ("full-imagenet-family-plan.json", "full-spair-family-plan.json", "clevr-joint-search-plan.json"))]
    pins = {}
    for p in request_paths:
        q = read(p)
        for filename, expected in q["pinned_files"].items():
            file = Path(filename)
            assert file.is_relative_to(ROOT), filename
            assert sha(file) == expected, filename
            assert filename not in pins or pins[filename] == expected
            pins[filename] = expected
            paths.add(file.relative_to(ROOT).as_posix())
        for job in q.get("jobs", []):
            if job.get("baseline_output"):
                p = Path(job["baseline_output"])
                assert p.is_relative_to(ROOT) and not (p / ".work").exists()
                paths.add(p.relative_to(ROOT).as_posix())
    # Keep the full native baseline directories needed by the independent verifiers.
    paths.update(("outputs/acceptance/full-probes/ade-family/resnet18-pretrained",
                  "outputs/c3-acceptance-20261023/results/full-ade-family/vit_b_16-pretrained",
                  "outputs/acceptance/full-probes/clevr-family/vit_b_16-pretrained"))
    data = {}
    tasks = ("imagenet1k-linear-v1", "imagenet100-linear-v1", "ade20k-linear-v1",
             "voc2012-linear-v1", "spair71k-match-v1", "clevr-count-v1", "clevr-distance-v1")
    for name in tasks:
        directory = SOURCE / "data/.bench/prepared" / name
        manifest = read(directory / "manifest.json")
        root = Path(manifest["source_root"])
        assert root.is_relative_to(ROOT)
        group = data.setdefault(str(root), {"files": {}, "tasks": []})
        group["tasks"].append({"task": name, "digest": manifest["digest"], "counts": {s: manifest["splits"][s]["count"] for s in ("train", "development")}})
        for split in ("train", "development"):
            for row in read(directory / manifest["splits"][split]["file"]):
                for relative, expected in row["files"].items():
                    p = Path(relative)
                    assert not p.is_absolute() and ".." not in p.parts
                    assert relative not in group["files"] or group["files"][relative] == expected
                    group["files"][relative] = expected
                    if root.name != "imagenet":
                        paths.add((root / p).relative_to(ROOT).as_posix())
    imagenet = str(ROOT / "outputs/c3-source-overlays/imagenet")
    assert len(data[imagenet]["files"]) == 1_281_167
    assert all(Path(p).parts[0] == "train" for p in data[imagenet]["files"])
    # The existing r2 ImageNet is reused in place, without writes to that dataset.
    assert not any(p.startswith("outputs/c3-source-overlays/imagenet/") for p in paths)
    with gzip.open(DEST / "expected-source-files.json.gz", "wt") as f:
        json.dump(data, f, sort_keys=True, separators=(",", ":"))
    paths.update(("outputs/r2-migration-20261026/expected-source-files.json.gz",
                  "outputs/r2-migration-20261026/request.json", "outputs/r2-migration-20261026/origin-queue.json",
                  "outputs/r2-migration-20261026/files-from.txt"))
    for path in paths:
        assert not Path(path).is_absolute() and ".." not in Path(path).parts
    (DEST / "files-from.txt").write_text("\n".join(sorted(paths)) + "\n")
    request = {"status": "prepared", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "origin_host": "clpc122.cs.ox.ac.uk", "destination_host": "clpc35.cs.ox.ac.uk", "root": str(ROOT),
        "r2_gpu_uuid": "GPU-e82ac849-9237-5670-9b20-5354929b6d05",
        "existing_r2_imagenet": "/home/scratch/danson/datasets/imagenet",
        "files_from_sha256": sha(DEST / "files-from.txt"), "expected_sources_sha256": sha(DEST / "expected-source-files.json.gz"),
        "static_pins": pins, "source_manifests": [{"path": str(p), "sha256": sha(p)} for p in request_paths],
        "source_file_counts": {r: len(g["files"]) for r, g in data.items()},
        "scope": "Copy checkout, exact verified main environment, frozen public-model evidence and training/development source inputs; reuse and checksum r2 ImageNet. Derive fresh r2 plans and keep evaluations held. No completed fit is rerun and final partition contents are not copied or evaluated."}
    (DEST / "request.json").write_text(json.dumps(request, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": request["status"], "transfer_entries": len(paths), "source_file_counts": request["source_file_counts"]}))


if __name__ == "__main__":
    prepare()
