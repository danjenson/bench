"""Relocate and run the c3 acceptance continuation without editing Ghost evidence.

Each phase records its own inputs and fails closed before GPU work. Historical
views change only execution locations, retaining the original model identities.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import gzip
import json
import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time


ROOT = next(path for path in Path(__file__).resolve().parents if (path / "SPEC.md").is_file())
OLD = "/data/repos/bench"
HANDOFF = ROOT / "outputs/acceptance/c3-handoff-20261023"
RUN = ROOT / "outputs/c3-acceptance-20261023"
ORIGINALS = Path("/tmp/bench-c3-original-sources-20261023")
BENCH = ROOT / "outputs/environments/bench/bin/python"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def replace(value, mapping):
    if isinstance(value, str):
        for before, after in sorted(mapping.items(), key=lambda pair: -len(pair[0])):
            value = re.sub(re.escape(before) + r"(?=$|[^A-Za-z0-9_.-])", lambda match: str(after), value)
        return value
    if isinstance(value, list):
        return [replace(item, mapping) for item in value]
    if isinstance(value, dict):
        return {replace(key, mapping): replace(item, mapping) for key, item in value.items()}
    return value


def base_mapping():
    return {
        OLD + "/.venv/bin/python": str(BENCH),
        "/data/cache/datasets/imagenet": str(ROOT / "outputs/c3-source-overlays/imagenet"),
        "/data/cache/datasets/ade20k/ADEChallengeData2016": str(ROOT / "outputs/c3-source-overlays/ade20k/ADEChallengeData2016"),
        "/data/cache/datasets/pascal-voc-2012/VOCdevkit/VOC2012": str(ROOT / "outputs/c3-source-overlays/pascal-voc-2012/VOCdevkit/VOC2012"),
        OLD: str(ROOT),
    }


def original_location(name):
    if name.startswith("/tmp/bench-"):
        return ORIGINALS / Path(name).relative_to("/tmp")
    if name == read(HANDOFF / "manifest.json")["python_binary_pin_requiring_relocation"]:
        return BENCH.resolve()
    return Path(replace(name, base_mapping()))


def archive_sources():
    manifest = read(HANDOFF / "manifest.json")
    archive = HANDOFF / manifest["source_archive"]
    assert sha(archive) == manifest["source_archive_sha256"], "Source archive differs"
    with tarfile.open(archive) as source:
        assert all(not Path(member.name).is_absolute() and ".." not in Path(member.name).parts
                   and (member.isfile() or member.isdir()) for member in source.getmembers())
        source.extractall(ORIGINALS, filter="data")
    for name, expected in manifest["source_files"].items():
        assert sha(original_location(name)) == expected, name
    for plan in manifest["plans"]:
        assert sha(original_location(plan["plan"])) == plan["plan_sha256"], plan["plan"]
    save(RUN / "archive-verification.json", {
        "status": "verified", "at_utc": now(), "archive_sha256": sha(archive),
        "source_files": len(manifest["source_files"]), "plans": len(manifest["plans"]),
        "manifest_sha256": sha(HANDOFF / "manifest.json"),
    })


def prepared_data():
    source = ROOT / "outputs/acceptance/data/.bench"
    target = RUN / "data/.bench"
    if (RUN / "prepared-relocation.json").exists():
        return
    changes = []
    for path in source.rglob("*.json"):
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if path.name == "manifest.json" or relative.parts[0] == "sources":
            old = read(path)
            new = deepcopy(old)
            field = "source_root" if path.name == "manifest.json" else "root"
            new[field] = replace(old[field], base_mapping())
            if path.name == "manifest.json":
                assert digest({k: v for k, v in old.items() if k not in ("digest", "source_root")}) == old["digest"]
                assert digest({k: v for k, v in new.items() if k not in ("digest", "source_root")}) == new["digest"]
            save(destination, new)
            changes.append({"original": str(path), "original_sha256": sha(path),
                            "relocated": str(destination), "relocated_sha256": sha(destination),
                            "field": field, "before": old[field], "after": new[field]})
        elif not destination.exists():
            os.link(path, destination)
    save(RUN / "prepared-relocation.json", {"status": "relocated", "at_utc": now(), "changes": changes,
         "scope": "Only source roots change; split files are hardlinked with identical bytes, labels and order. Final partitions are not evaluated."})


def reidentify(value):
    if isinstance(value, list):
        return [reidentify(item) for item in value]
    if isinstance(value, dict):
        result = {key: reidentify(item) for key, item in value.items()}
        if {"state_digest", "info", "numerics", "digest"} <= result.keys():
            result["digest"] = digest({key: item for key, item in result.items() if key != "digest"})
        return result
    return value


def historical_view(name, source, mapping, *, preserve_models=True):
    """A location-only plan view, never a rewritten result or model identity."""
    old = read(source)
    new = replace(old, mapping)
    if preserve_models:
        if "model" in old:
            new["model"] = old["model"]
        before_jobs, after_jobs = old.get("jobs", []), new.get("jobs", [])
        if isinstance(before_jobs, dict):
            before_jobs, after_jobs = list(before_jobs.values()), list(after_jobs.values())
        for before, after in zip(before_jobs, after_jobs):
            for field in ("model", "model_identity"):
                if field in before:
                    after[field] = before[field]
    else:
        new = reidentify(new)
    destination = RUN / "historical-views" / name
    save(destination, new)
    save(destination.with_suffix(".provenance.json"), {
        "original": str(source), "original_sha256": sha(source),
        "view_sha256": sha(destination), "mapping": mapping,
        "scope": "Explicit location view of a historical plan; historical result files are unchanged.",
    })
    return destination


def relocate_family(name):
    archive_sources()
    prepared_data()
    manifest = read(HANDOFF / "manifest.json")
    entry = next(item for item in manifest["plans"] if Path(item["plan"]).name == name)
    original_path = original_location(entry["plan"])
    old = read(original_path)
    plan_path = RUN / "plans" / name
    if plan_path.exists():
        return plan_path
    for path, expected in old["pinned_files"].items():
        assert sha(original_location(path)) == expected, f"Original pin mismatch: {path}"
    source = RUN / "sources" / Path(entry["source_root"]).name
    if source.exists():
        abandoned = RUN / "drafts" / (source.name + "-" + str(time.time_ns()))
        abandoned.parent.mkdir(parents=True, exist_ok=True)
        source.rename(abandoned)
    shutil.copytree(original_location(entry["source_root"]), source)
    mapping = base_mapping()
    for previous in (RUN / "plans").glob("*-plan.json"):
        mapping.update(read(previous)["relocation"]["mapping"])
    mapping[entry["source_root"]] = str(source)
    mapping[OLD + "/outputs/acceptance/data"] = str(RUN / "data")
    mapping[entry["plan"]] = str(plan_path)
    mapping[old["state"]] = str(RUN / "states" / Path(old["state"]).name)
    for job in old["jobs"]:
        mapping[job["output"]] = str(RUN / "results" / name.removesuffix("-plan.json") / job["name"])
        mapping[job["log"]] = str(RUN / "logs" / name.removesuffix("-plan.json") / (job["name"] + ".log"))
    if "comparison_output" in old:
        mapping[old["comparison_output"]] = str(RUN / "results" / name.removesuffix("-plan.json") / "comparison.json")
    # Evaluation plans outside the temporary caller tree need their own copies.
    extra_configs = []
    if name == "public-cnn-batch16-plan.json":
        original = ROOT / "outputs/acceptance/public-cnn-batch16-evaluation-plan.json"
        destination = RUN / "configs" / original.name
        mapping[OLD + "/outputs/acceptance/" + original.name] = str(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, destination)
        save(destination.with_suffix(".provenance.json"), {"original": str(original), "original_sha256": sha(original)})
        extra_configs.append(destination)
        baseline = read(original)["baseline_plan"]
        view = historical_view("public-checkpoint-monitoring-source-label-plan.json", original_location(baseline),
            {**base_mapping(), OLD + "/outputs/acceptance/data": str(RUN / "data")})
        mapping[baseline] = str(view)
    if name == "monitor-fuller-selection-plan.json":
        baseline = old["baseline_plan"]
        view = historical_view("fuller-convergence-plan.json", original_location(baseline),
            {**base_mapping(), OLD + "/outputs/acceptance/data": str(RUN / "data")}, preserve_models=False)
        mapping[baseline] = str(view)
    if name == "full-clevr-search-grid-plan.json":
        baseline = old["baseline_plan"]
        view = historical_view("full-clevr-family-plan.json", original_location(baseline),
            {**base_mapping(), OLD + "/outputs/acceptance/data": str(RUN / "data")})
        mapping[baseline] = str(view)
    for relative, entry_source in old.get("durable_caller_sources", {}).items():
        candidate = source / relative
        if not candidate.exists():
            candidate = source / "caller" / relative
        if candidate.is_file():
            mapping[entry_source["file"]] = str(candidate)
    file_changes = []
    for path in sorted(list(source.rglob("*")) + extra_configs):
        if path.is_relative_to(source / "bench"):
            continue
        if path.suffix not in (".json", ".yaml", ".yml") or not path.is_file():
            continue
        before = sha(path)
        if path.suffix == ".json":
            save(path, reidentify(replace(read(path), mapping)))
        else:
            path.write_text(replace(path.read_text(), mapping))
        if before != sha(path):
            file_changes.append({"file": str(path), "original_sha256": before, "relocated_sha256": sha(path)})
    identity_map = {}
    for job in old["jobs"]:
        before = read(original_location(job["expected_model"]))
        after = read(replace(job["expected_model"], mapping))
        identity_map[before["digest"]] = after["digest"]
    for path in [*(p for p in source.rglob("*.json") if not p.is_relative_to(source / "bench")), *extra_configs]:
        save(path, replace(read(path), identity_map))
    new = reidentify(replace(old, mapping))
    new = replace(new, identity_map)
    new["gpu_uuid"] = manifest["c3_gpu_uuid"]
    new.setdefault("data_root", str(RUN / "data"))
    new["status"] = "prepared"
    for field in ("cpu_identity_checks", "cpu_identity_verification", "validation"):
        if field in new:
            new["original_" + field] = new.pop(field)
    new["relocation"] = {"original_plan": str(original_path), "original_plan_sha256": entry["plan_sha256"],
        "mapping": mapping, "report": str(plan_path.with_suffix(".relocation.json")),
        "host": "c3", "timing_scope": "Fresh c3 evaluation on RTX 4090; independent of Ghost timing."}
    new.pop("predecessor", None)
    new.pop("serialization_dependency", None)
    index = manifest["plans"].index(entry)
    if index:
        previous = RUN / "plans" / Path(manifest["plans"][index - 1]["plan"]).name
        new["c3_predecessor"] = {"plan": str(previous), "plan_sha256": sha(previous)}
    if name == "full-ade-family-plan.json":
        new["completed_on_ghost"] = old["jobs"][0]
        new["jobs"] = new["jobs"][1:]
        new["total_fits"] = 25
        new["expected_retained_records"] = 30315
        view_mapping = {**base_mapping(), OLD + "/outputs/acceptance/data": str(RUN / "data")}
        view = historical_view("full-clevr-family-plan.json", ROOT / "outputs/acceptance/full-clevr-family-plan.json", view_mapping)
        new["baseline_verification_command"] = replace(old["predecessor"]["verification_command"], mapping)
        new["baseline_verification_command"][-1] = str(view)
    new["pinned_files"] = {str(Path(replace(path, mapping))): expected for path, expected in old["pinned_files"].items()}
    # Configuration hashes refer to relocated files, while checkpoint/source
    # checksums retain their original values. No result identity is rewritten.
    configs = [*(p for p in source.rglob("*.json") if not p.is_relative_to(source / "bench")), *extra_configs]
    for _ in range(3):
        for path in configs:
            config = read(path)
            refresh_config_pins(config, mapping)
            if "hardware" in config:
                config["original_hardware"] = config.get("original_hardware", config["hardware"])
                config["hardware"] = hardware()
            save(path, config)
    adapt_reference_verifiers(source)
    for path in configs:
        config = read(path)
        refresh_config_pins(config, mapping)
        save(path, config)
    pins = {}
    pin_changes = []
    for path, expected in old["pinned_files"].items():
        target = Path(replace(path, mapping))
        if path == manifest["python_binary_pin_requiring_relocation"]:
            target = BENCH.resolve()
        actual = sha(target)
        pins[str(target)] = actual
        if actual != expected:
            if target.is_relative_to(RUN / "plans"):
                assert read(target)["relocation"]["original_plan_sha256"] == expected
            elif target.is_relative_to(RUN / "historical-views") or target.is_relative_to(RUN / "configs"):
                assert read(target.with_suffix(".provenance.json"))["original_sha256"] == expected
            elif target.is_relative_to(RUN / "sources") and not target.is_relative_to(source):
                assert any(read(previous)["pinned_files"].get(str(target)) == actual for previous in (RUN / "plans").glob("*-plan.json")), f"Unverified relocated predecessor input: {path}"
            else:
                assert target.is_relative_to(source) or target.is_relative_to(RUN / "data") or target in extra_configs, f"Undeclared changed pin: {path}"
            pin_changes.append({"original": path, "original_sha256": expected, "relocated": str(target), "relocated_sha256": actual})
    new["pinned_files"] = pins
    if (source / "caller/c3_reference.py").exists():
        new["pinned_files"][str(source / "caller/c3_reference.py")] = sha(source / "caller/c3_reference.py")
    refresh_config_pins(new, mapping)
    for job in new["jobs"]:
        job["model_yaml_sha256"] = sha(job["model_yaml"])
        expected = read(job["expected_model"])
        if "model" in job:
            assert expected == job["model"], job["name"]
        else:
            job["model"] = expected
        job["status"] = "pending"
    save(plan_path.with_suffix(".relocation.json"), {"status": "relocated-awaiting-validation", "at_utc": now(),
        "original_plan_sha256": entry["plan_sha256"], "mapping": mapping,
        "changed_files": file_changes, "changed_pins": pin_changes,
        "package_sources_unchanged": True, "excluded_ghost_models": ["resnet18-pretrained"] if name == "full-ade-family-plan.json" else []})
    save(plan_path, new)
    return plan_path


def hardware():
    # This is also checked again by monitoring workers at evaluation time.
    if (RUN / "hardware.json").exists():
        return read(RUN / "hardware.json")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader,nounits"], text=True).strip()
    import platform
    return {"gpu_query": gpu, "host": platform.node(), "platform": platform.platform(),
            "logical_cpus": os.cpu_count(), "workspace_device_id": RUN.stat().st_dev}


def refresh_config_pins(value, mapping):
    if isinstance(value, list):
        for item in value:
            refresh_config_pins(item, mapping)
    if not isinstance(value, dict):
        return
    if "pinned_files" in value:
        pins = {}
        for path, expected in value["pinned_files"].items():
            location = Path(replace(path, mapping))
            if location.exists():
                pins[str(location)] = sha(location)
            else:
                pins[str(location)] = expected
        value["pinned_files"] = pins
    for path_field, hash_field in [("model_yaml", "model_yaml_sha256"), ("baseline_plan", "baseline_plan_sha256")]:
        if path_field in value and hash_field in value and Path(value[path_field]).exists():
            value[hash_field] = sha(value[path_field])
    if "durable_caller_sources" in value:
        for entry in value["durable_caller_sources"].values():
            if Path(entry["file"]).exists():
                entry["sha256"] = sha(entry["file"])
    if isinstance(value.get("file"), str) and "sha256" in value and Path(value["file"]).is_file():
        value["sha256"] = sha(value["file"])
    for item in value.values():
        if isinstance(item, (dict, list)):
            refresh_config_pins(item, mapping)


def adapt_reference_verifiers(source):
    """Keep historic bytes intact; compare declared location changes explicitly."""
    helper = source / "caller/c3_reference.py"
    reference = {**base_mapping()}
    helper.write_text('''import hashlib,json\nfrom pathlib import Path\nMAPPING=''' + repr(reference) + '''\ndef location(v):
 if isinstance(v,str):
  for a,b in sorted(MAPPING.items(),key=lambda p:-len(p[0])):v=v.replace(a,b)
  return v
 if isinstance(v,list):return [location(x) for x in v]
 if isinstance(v,dict):return {k:location(x) for k,x in v.items()}
 return v
def relocated_model(v):
 from bench.storage import digest
 assert v['digest']==digest({k:x for k,x in v.items() if k!='digest'})
 r=location(v);r['digest']=digest({k:x for k,x in r.items() if k!='digest'});return r
def original_plan_digest(p):
 p=Path(p);v=p.with_suffix('.provenance.json')
 if v.exists():return json.loads(v.read_text())['original_sha256']
 with p.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
''')
    path = source / "caller/verify_full_probe.py"
    if path.exists():
        text = path.read_text().replace("assert plan['model'] == baseline['model']", "from c3_reference import relocated_model\n    assert plan['model'] == relocated_model(baseline['model'])")
        path.write_text(text)
    path = source / "caller/verify_extraction_batch.py"
    if path.exists():
        text = path.read_text()
        text = text.replace("from public_checkpoint_monitoring import verify as verify_monitor", "from public_checkpoint_monitoring import verify as verify_monitor\nfrom c3_reference import location, relocated_model, original_plan_digest")
        text = text.replace("assert results[0]['model'] == results[1]['model']", "assert relocated_model(results[0]['model']) == results[1]['model']")
        text = text.replace("assert execution == expected", "assert location(execution) == expected")
        text = text.replace("file_digest(ppath)", "original_plan_digest(ppath)")
        text = text.replace("only extraction batch size changes", "declared extraction batch and host/environment locations change; historical timings retain their original host")
        path.write_text(text)


def environment(job):
    return dict(os.environ, PYTHONPATH=os.pathsep.join(job["pythonpath"]), PYTHONUNBUFFERED="1", MLFLOW_DISABLE_AGENT_HINT="1")


def public_reference(job):
    """Require a pinned official checkpoint before reference evaluation."""
    references = read(ROOT / "docs/reference-models.json")["pretrained_references"]
    recipe = job["model"]["info"]["recipe"]
    matches = [entry for entry in references if entry["architecture"] == recipe.get("architecture")
               and entry["sha256"] == recipe.get("sha256") and entry["url"] == recipe.get("url")]
    assert len(matches) == 1, "Model is outside the supported public reference set"
    checkpoint = Path(recipe["checkpoint"])
    assert sha(checkpoint) == matches[0]["sha256"], "Official reference checkpoint differs"
    return matches[0]


ENV_PROBE = """import importlib.metadata as m,json,sys,hashlib,torch
from pathlib import Path
required=json.loads(sys.argv[1]);actual={}
for name in required:
 try: actual[name]=m.version(name)
 except m.PackageNotFoundError: actual[name]=None
assert actual==required,(actual,required)
assert sys.version_info[:3]==(3,13,13)
assert torch.__version__=='2.14.1+cu130' and torch.version.cuda=='13.0'
assert not torch.cuda.is_initialized()
print(json.dumps({'packages':actual,'python':sys.version,'binary_sha256':hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),'cuda_initialized':False}))
"""


CPU_PROBE = """import json,sys,torch
from pathlib import Path
from hydra import compose,initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf
import bench
from bench.features import model_identity,ImageDataset,collate_images,extract,storage_estimate
from bench.prepare import load_prepared
from bench.protocols import resolve_names
p=json.loads(Path(sys.argv[1]).read_text());j=next(j for j in p['jobs'] if j['name']==sys.argv[2]);torch.set_num_threads(8)
torch.random.default_generator.manual_seed(937);state=torch.random.get_rng_state().clone()
if j['command'][1:3]==['-c','from bench.cli.main import main; main()']:
 with initialize_config_module(config_module='bench.cli.configs',version_base='1.3'):
  cfg=compose(config_name='config',overrides=j['command'][3:])
  model=instantiate(cfg.model,_convert_='all')
else:
 with torch.random.fork_rng(devices=[]):
  torch.random.default_generator.manual_seed(0)
  model=instantiate(OmegaConf.load(j['model_yaml']),_convert_='all')
assert torch.equal(state,torch.random.get_rng_state())
profile=j['model']['numerics']
actual=model_identity(model,bench.Execution(device='cpu',numerics=profile))
assert actual==j['model'],(actual,j['model'])
names=p.get('names')
if not names and isinstance(p.get('tasks'),dict):names=[v['recipe']['name'] for v in p['tasks'].values()]
if not names:names=['imagenet-100','pascal']
task=resolve_names(names)[0]
prepared,rows=load_prepared(Path(p['data_root']),task,'development',verify=False)
if task['kind']=='correspondence':
 from bench.tasks.correspondence import distinct_images
 rows=distinct_images(rows)
index=next(i for i,r in enumerate(rows) if 'image' in r and (Path(prepared['source_root'])/r['image']).is_file())
batch=bench.ImageBatch(*collate_images([ImageDataset(Path(prepared['source_root']),rows,task)[index]]))
tensors=extract(model,batch,task['method'],bench.Execution(device='cpu',numerics='fp32-v1',workers=0))
shape=list(tensors['values'].shape[1:])
expected_shape=j.get('feature_shape') or ([j['feature_width']] if 'feature_width' in j else p.get('expected_shapes',{}).get(task['id']))
if expected_shape is not None:assert shape==expected_shape
if 'grid_to_canvas' in j:
 matrix=torch.linalg.solve(batch.image_to_original[0],tensors['grid_to_original'][0])
 torch.testing.assert_close(matrix,torch.tensor(j['grid_to_canvas']),rtol=1e-6,atol=1e-7)
assert model_identity(model,bench.Execution(device='cpu',numerics=profile))==actual
assert torch.equal(state,torch.random.get_rng_state()) and not torch.cuda.is_initialized()
assert Path(bench.__file__).is_relative_to(Path(p['package_source_root'])/'bench')
print(json.dumps({'name':j['name'],'model':actual,'feature_shape':shape,'cuda_initialized':False,'cpu_rng_preserved':True,'bench_source':bench.__file__}))
"""


def validate_family(plan_path, *, models_only=False):
    plan = read(plan_path)
    assert all(sha(path) == expected for path, expected in plan["pinned_files"].items()), "Relocated pins changed before validation"
    for job in plan["jobs"]:
        public_reference(job)
    checks = []
    for job in plan["jobs"]:
        env = environment(job)
        runtime = probe([job["command"][0], "-c", ENV_PROBE, json.dumps(job["environment_versions"])], env)
        assert runtime["binary_sha256"] == read(ROOT / "outputs/environments/setup-receipt.json")["environments"]["bench"]["python_binary_sha256"]
        identity = probe([job["command"][0], "-c", CPU_PROBE, str(plan_path), job["name"]], env)
        checks.append({"runtime": runtime, **identity})
        print("CPU identity and spatial contract verified:", job["name"], flush=True)
    if models_only:
        save(plan_path.with_suffix(".models-validation.json"), {"status": "passed", "plan_sha256": sha(plan_path), "at_utc": now(), "checks": checks,
             "scope": "CPU model/environment contracts only; data and scientific predecessors must still pass before evaluation."})
        return
    env = environment(plan["jobs"][0])
    script = """import json,sys
from pathlib import Path
from bench.prepare import load_prepared
from bench.protocols import resolve_names
p=json.loads(Path(sys.argv[1]).read_text());out=[]
for name in sys.argv[2:]:
 try:t=resolve_names([name])[0]
 except ValueError:t=json.loads((Path(p['data_root'])/'.bench/prepared'/name/'manifest.json').read_text())['task']
 for split in ['train','development']:
  m,rows=load_prepared(Path(p['data_root']),t,split,verify=True);out.append({'task':t['id'],'split':split,'rows':len(rows),'preparation_digest':m['digest']})
print(json.dumps(out))
"""
    names = plan.get("names")
    if not names and isinstance(plan.get("tasks"), dict):
        names = [item["recipe"]["name"] for item in plan["tasks"].values()]
    if not names:
        names = ["imagenet-100", "pascal"]
    if plan_path.name == "monitor-fuller-selection-plan.json":
        names = [*names, "imagenet100-linear-v1-fuller-bank-calibration-v1", "voc2012-linear-v1-fuller-bank-calibration-v1"]
    data = json.loads(subprocess.check_output([str(BENCH), "-c", script, str(plan_path), *names], env=env, text=True))
    if plan.get("baseline_verification_command"):
        subprocess.run(plan["baseline_verification_command"], env=env, check=True)
    save(plan_path.with_suffix(".validation.json"), {"status": "passed", "plan_sha256": sha(plan_path), "at_utc": now(), "checks": checks, "data": data})


def probe(command, env):
    result = subprocess.run(command, env=env, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("CPU preflight failed:\n" + result.stderr[-8000:])
    return json.loads(result.stdout)


def boundary_ready():
    boundary = read(HANDOFF / "ghost-boundary.json")
    transfer = read(HANDOFF / "transfer-receipt.json")
    assert transfer["status"] == "complete"
    if boundary["action"] == "stop-after-verified-model":
        assert "Done" in boundary["final_task_status"]
        assert boundary["jobs"][0]["status"] == "complete" and boundary["jobs"][0]["exit_code"] == 0
    else:
        corrected_path = HANDOFF / "ghost-boundary-revalidated.json"
        corrected = read(corrected_path)
        assert transfer["boundary_sha256"] == sha(corrected_path)
        assert corrected["status"] == "complete" and corrected["evaluation_exit_code"] == 0
        assert corrected["action"] == "model-verified-after-resumption-accounting-fix"
        assert not corrected["remaining_models_started_on_ghost"]
        assert corrected["original_boundary_receipt_sha256"] == sha(HANDOFF / "ghost-boundary.json")
        assert corrected["original_queue_state_sha256"] == sha(HANDOFF / "ghost-ade-failed-queue-state.json")
        assert "Done" in boundary["task"]["status"]
        assert sha(HANDOFF / corrected["verification"]) == corrected["verification_sha256"]
        verified = read(HANDOFF / corrected["verification"])
        assert verified["status"] == "verified" and verified["records"] == 6063
        assert verified["pinned_inputs_verified"] == 82
        assert sha(ROOT / "outputs/acceptance/full-probes/ade-family/resnet18-pretrained/results.json") == corrected["results_sha256"]
        assert not (ROOT / "outputs/acceptance/full-probes/ade-family/resnet18-pretrained/.work").exists()


def validate_execution_outputs(plan):
    """Keep prospective writes under the c3 execution directories."""
    targets = [("state", plan["state"], RUN / "states")]
    if "comparison_output" in plan:
        targets.append(("comparison_output", plan["comparison_output"], RUN / "results"))
    for job in plan["jobs"]:
        targets.extend([(job["name"] + ":output", job["output"], RUN / "results"),
                        (job["name"] + ":log", job["log"], RUN / "logs")])
    for field, value, directory in targets:
        path, owner = Path(value).resolve(), directory.resolve()
        assert path.is_relative_to(owner) and path != owner, f"Output escapes c3 ownership: {field}: {value}"


def run_family(plan_path):
    validate_execution_outputs(read(plan_path))
    boundary_ready()
    plan = read(plan_path)
    for job in plan["jobs"]:
        public_reference(job)
    assert read(plan_path.with_suffix(".validation.json"))["plan_sha256"] == sha(plan_path)
    if "c3_predecessor" in plan:
        previous = plan["c3_predecessor"]
        assert sha(previous["plan"]) == previous["plan_sha256"]
        verify_family(Path(previous["plan"]))
    if plan_path.name == "full-imagenet-family-plan.json":
        receipt = read(RUN / "imagenet-transfer/receipt.json")
        assert receipt["status"] == "complete" and receipt["source_images_verified"] == 1_281_167
    state_path = Path(plan["state"])
    assert not state_path.exists(), "Inspect an existing state before attempting recovery"
    state = {"status": "running", "plan_sha256": sha(plan_path), "started_at_utc": now(), "jobs": []}
    save(state_path, state)
    try:
        for job in plan["jobs"]:
            assert all(sha(path) == expected for path, expected in plan["pinned_files"].items()), "Pinned inputs changed"
            assert sha(plan_path) == state["plan_sha256"]
            gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True).splitlines()
            assert gpu == [plan["gpu_uuid"]]
            assert not subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip(), "GPU has another compute process"
            required = job.get("estimated_peak_temporary_bytes", 0)
            assert shutil.disk_usage(RUN).free > required + 10_000_000_000, "Insufficient space for the declared run and margin"
            assert not Path(job["output"]).exists(), "Evaluation needs a fresh output"
            log = Path(job["log"])
            log.parent.mkdir(parents=True, exist_ok=True)
            record = {"name": job["name"], "status": "running", "started_at_utc": now(), "command": job["command"], "output": job["output"]}
            state["jobs"].append(record)
            save(state_path, state)
            started = time.monotonic()
            print("Starting", job["name"], flush=True)
            with log.open("w") as stream:
                subprocess.run(job["command"], env=environment(job), stdout=stream, stderr=subprocess.STDOUT, check=True)
            assert read(Path(job["output"]) / job.get("model_metadata_file", "plan.json"))["model"] == read(job["expected_model"])
            subprocess.run(job["verification_command"], env=environment(job), check=True)
            assert not (Path(job["output"]) / ".work").exists()
            artifact = next(Path(job["output"]) / name for name in ("results.json", "calibration.json", "acceptance.json") if (Path(job["output"]) / name).exists())
            record.update(status="complete", finished_at_utc=now(), wall_seconds_including_loading_cleanup_and_verification=time.monotonic() - started,
                          artifact=str(artifact), artifact_sha256=sha(artifact),
                          verification_sha256=sha(Path(job["output"]) / "verification.json"),
                          verification_files={str(p): sha(p) for p in Path(job["output"]).glob("*verification.json")})
            if artifact.name == "results.json":
                record["results_sha256"] = sha(artifact)
            save(state_path, state)
        state.update(status="complete", finished_at_utc=now(), all_workspaces_pruned=True)
        save(state_path, state)
    except BaseException as error:
        state.update(status="failed", error=str(error), failed_at_utc=now())
        save(state_path, state)
        raise


def verify_family(plan_path):
    plan = read(plan_path)
    state = read(plan["state"])
    assert state["status"] == "complete" and state["plan_sha256"] == sha(plan_path)
    assert len(state["jobs"]) == len(plan["jobs"])
    for job, completed in zip(plan["jobs"], state["jobs"]):
        assert job["name"] == completed["name"] and completed["status"] == "complete"
        assert sha(completed["artifact"]) == completed["artifact_sha256"]
        assert sha(Path(job["output"]) / "verification.json") == completed["verification_sha256"]
        assert all(sha(path) == expected for path, expected in completed["verification_files"].items())
        assert not (Path(job["output"]) / ".work").exists()
        subprocess.run(job["verification_command"], env=environment(job), check=True)


def verify_ghost_model():
    plan = read(RUN / "plans/full-ade-family-plan.json")
    mapping = {**base_mapping(), OLD + "/outputs/acceptance/data": str(RUN / "data")}
    view = historical_view("full-ade-family-plan.json", ROOT / "outputs/acceptance/full-ade-family-plan.json", mapping)
    corrected = HANDOFF / "ade-recovery-verification"
    request = read(corrected / "request.json")
    assert sha(corrected / "caller/verify_ade_controls.py") == request["verifier_sha256"]
    original = read(ROOT / "outputs/acceptance/full-ade-family-plan.json")
    for path, expected in original["pinned_files"].items():
        assert sha(original_location(path)) == expected, path
    script = """import json,sys
from pathlib import Path
from verify_ade_controls import verify
from bench.storage import read_json,file_digest
actual=verify(Path(sys.argv[1]),'resnet18-pretrained',reused_bank=True)
saved=read_json(sys.argv[2])
assert all(value==saved[key] for key,value in actual.items() if key!='plan_sha256')
print(json.dumps(actual))
"""
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(corrected / "caller"), plan["package_source_root"]]))
    actual = probe([str(BENCH), "-c", script, str(view), str(corrected / "verification.json")], env)
    save(RUN / "ghost-model-verification.json", {"status": "verified", "at_utc": now(), "evidence": actual,
         "recovery_request_sha256": sha(corrected / "request.json"), "historical_plan_sha256": request["original_plan_sha256"]})


def ade_cleanup_boundary():
    plan_path = RUN / "plans/full-ade-family-plan.json"
    plan = read(plan_path)
    state = read(plan["state"])
    assert state["status"] == "complete" and state["all_workspaces_pruned"]
    assert state["plan_sha256"] == sha(plan_path)
    assert len(plan["jobs"]) == len(state["jobs"]) == 5
    for job, completed in zip(plan["jobs"], state["jobs"]):
        assert completed["status"] == "complete" and completed["name"] == job["name"]
        output = Path(job["output"])
        assert not (output / ".work").exists()
        result = read(output / "results.json")
        assert result["status"] == "complete"
        assert result["storage"]["retained_temporary_bytes"] == result["storage"]["retained_workspace_bytes"] == 0
        assert sha(output / "results.json") == completed["results_sha256"]
        assert sha(output / "verification.json") == completed["verification_sha256"]
        subprocess.run(job["verification_command"], env=environment(job), check=True)
    return {"plan_sha256": sha(plan_path), "state_sha256": sha(plan["state"]), "verified_models": 5,
            "all_workspaces_pruned": True, "verified_at_utc": now()}


def transfer_imagenet():
    boundary_ready()
    boundary = ade_cleanup_boundary()
    location = RUN / "imagenet-transfer"
    location.mkdir(parents=True, exist_ok=True)
    receipt = location / "receipt.json"
    assert not receipt.exists(), "Inspect the existing transfer receipt"
    audit = HANDOFF / "dataset-preflight/with-metadata-overlays"
    sizes = read(audit / "missing-imagenet-size.json")
    compressed = audit / "missing-imagenet-files.txt.gz"
    assert sizes["missing_images"] == 1_154_478 and sizes["missing_image_bytes"] == 132_411_896_262
    required_margin = 10_000_000_000
    next_job_space = max(job["estimated_peak_temporary_bytes"] for job in read(ROOT / "outputs/acceptance/full-imagenet-family-plan.json")["jobs"])
    free = shutil.disk_usage(RUN).free
    assert free > sizes["missing_image_bytes"] + next_job_space + required_margin, "Insufficient space after ADE cleanup"
    files = location / "missing-files.txt"
    with gzip.open(compressed, "rt") as source, files.open("w") as destination:
        count = 0
        for line in source:
            relative = Path(line.rstrip("\n"))
            assert not relative.is_absolute() and ".." not in relative.parts and relative.parts[0] == "train"
            destination.write(relative.as_posix() + "\n")
            count += 1
    assert count == sizes["missing_images"]
    root = ROOT / "outputs/c3-source-overlays/imagenet"
    command = ["rsync", "-a", "--partial", "--stats", "--files-from=" + str(files), "ghost:/data/cache/datasets/imagenet/", str(root) + "/"]
    request = {"status": "prepared", "created_at_utc": now(), "ade_cleanup_boundary": boundary,
               "missing_list_sha256": sha(compressed), "files_from_sha256": sha(files), "missing_images": count,
               "missing_image_bytes": sizes["missing_image_bytes"], "free_bytes_before": free,
               "next_family_peak_estimate_bytes": next_job_space, "margin_bytes": required_margin, "command": command}
    save(location / "request.json", request)
    with (location / "rsync.log").open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    original_request = read(audit / "plan.json")
    entry = next(item for item in original_request["datasets"] if item["name"] == "imagenet")
    checksum_source = (audit / entry["list"]).resolve()
    assert sha(checksum_source) == entry["list_sha256"]
    checksums = location / "imagenet.sha256"
    with gzip.open(checksum_source, "rt") as source, checksums.open("w") as destination:
        count = 0
        for line in source:
            expected, name = line.rstrip("\n").split("  ", 1)
            relative = Path(name).relative_to(entry["root"])
            assert ".." not in relative.parts and relative.parts[0] == "train"
            destination.write(expected + "  " + str(root / relative) + "\n")
            count += 1
    assert count == entry["unique_files"] == 1_281_167
    audit_request = {"original_request": str(audit / "plan.json"), "original_request_sha256": sha(audit / "plan.json"),
        "original_checksum_list_sha256": sha(checksum_source), "relocated_checksum_list_sha256": sha(checksums),
        "original_root": entry["root"], "root": str(root), "unique_files": count,
        "final_partitions_read": False, "transfer_request_sha256": sha(location / "request.json")}
    save(location / "audit-request.json", audit_request)
    with (location / "checksum-audit.log").open("w") as stream:
        subprocess.run(["sha256sum", "--strict", "--quiet", "--check", str(checksums)], stdout=stream, stderr=subprocess.STDOUT, check=True)
    save(receipt, {"status": "complete", "completed_at_utc": now(), "source_images_verified": count,
        "audit_request_sha256": sha(location / "audit-request.json"), "checksum_log_sha256": sha(location / "checksum-audit.log"),
        "final_partitions_read": False, "free_bytes_after": shutil.disk_usage(RUN).free})


def main():
    parser = argparse.ArgumentParser()
    if os.environ.get("C3_RUNNER_SHA256"):
        assert sha(__file__) == os.environ["C3_RUNNER_SHA256"], "Frozen runner changed"
    parser.add_argument("action", choices=["relocate", "validate", "validate-models", "run", "phase", "verify", "verify-ghost", "transfer-imagenet"])
    parser.add_argument("plan", nargs="?")
    args = parser.parse_args()
    if args.action == "relocate":
        print(relocate_family(args.plan))
    elif args.action == "validate":
        validate_family(Path(args.plan))
    elif args.action == "validate-models":
        validate_family(Path(args.plan), models_only=True)
    elif args.action == "run":
        run_family(Path(args.plan))
    elif args.action == "phase":
        path = Path(args.plan)
        validate_execution_outputs(read(path))
        validate_family(path)
        run_family(path)
    elif args.action == "verify":
        verify_family(Path(args.plan))
    elif args.action == "verify-ghost":
        verify_ghost_model()
    else:
        transfer_imagenet()


if __name__ == "__main__":
    main()
