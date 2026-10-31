# bench

Frozen visual representation benchmarks with one model adapter interface. Preparation, inspection, extraction, linear probes, correspondence scoring, compact results, and temporary-workspace pruning are implemented in Python 3.13.

## Use during model training

Install bench as a dependency of your training project in a Python 3.13+ environment. For a local checkout of this release:

```bash
uv add /path/to/bench
uv run bench prepare 'names=[imagenet-100,pascal]' data_root=/datasets/bench
```

Prepare the datasets once, then evaluate saved checkpoints during training. Implement an adapter in your model repository using [the public API](docs/api.md): `encode_global` returns classification features and `encode_spatial` returns spatial features with their geometry. The example below assumes your `EncoderAdapter` implements both methods and loads your checkpoint; bench supplies the datasets, frozen-feature probes and metrics.

```python
from pathlib import Path
import bench
from my_model.benchmark import EncoderAdapter


def benchmark_checkpoint(checkpoint, step):
    # Load an immutable snapshot in your evaluation worker.
    adapter = EncoderAdapter.from_checkpoint(checkpoint).to("cuda")
    result = bench.run(
        adapter,
        names=["imagenet-100", "pascal"],
        evaluation="development",
        data_root=Path("/datasets/bench"),
        output_dir=Path("outputs/bench") / f"step-{step}-development",
        execution=bench.Execution(
            device="cuda",
            batch_size=64,
            work_dir=Path("outputs/bench-work"),
        ),
    )
    return result.metrics


# Call after saving a checkpoint, at a cadence chosen by your training code.
metrics = benchmark_checkpoint("checkpoints/step-10000.pt", step=10000)
print(metrics)  # Send these metrics to your existing training logger.
```

Run evaluation in a separate process when training continues concurrently; never evaluate the same live module while its weights are changing. A synchronous callback waits for `bench.run` to finish. Choose the GPU and cadence for your setup and include evaluation time and contention when measuring training overhead. A global-only adapter can request `names=["imagenet-100"]`. FP32 is the default, and bench fits probe heads without training your encoder.

For broader checkpoint comparisons, prepare and add `imagenet-1k`, `ade20k`, `spair71k`, `clevr-count` and `clevr-distance`. After selecting your final checkpoint using development results, run the chosen task list with `evaluation="final"` and a fresh output directory, recording checkpoint-selection provenance in the adapter recipe. A shared `work_dir` reuses an available development probe selection for that same checkpoint and is closed after final evaluation. If training ends without a final evaluation, close its idle workspace with:

```bash
uv run bench prune work_dir=outputs/bench-work
```

The fourteen pretrained reference scores ship with the package. Comparing against them requires no reference-model training or additional baseline-artifact downloads; you supply only the datasets and your model.

The 1.0 interface evaluates frozen encoders under seven versioned task recipes. Package/API, task and result-schema versions are independent. See [SPEC.md](SPEC.md) for the contract and [docs/acceptance.md](docs/acceptance.md) for measured acceptance and its scope.

## Install and prepare

```nu
(uv sync --locked)
(uv run bench prepare 'names=[imagenet-100,pascal]' 'data_root=~/datasets')
(uv run bench visualize 'names=[imagenet-100,pascal]' 'data_root=~/datasets')
```

Python 3.13 is retained. `uv.lock` pins the tested environment. The base stack contains PyTorch, torchvision, NumPy, Pillow, safetensors, and Hydra/OmegaConf; DINOv3 adds its adapter dependencies with `(uv sync --extra dinov3)`.

Preparation indexes compatible local data in place. ImageNet requires an officially obtained tree with `train/<synset>/*.JPEG` and `val/<synset>/*.JPEG`. VOC2012, the ADE20K scene parsing challenge, SPair-71k, and CLEVR have archive acquisition recipes. Downloads are checksummed, extraction rejects links and escaping paths, and interrupted downloads can resume. Setup prints required space before acquisition and reports each requested task as ready or incomplete. An incomplete request exits with status 1 while retaining independent completed preparation.

Dataset paths can be set explicitly during preparation:

```nu
(uv run bench prepare 'names=[pascal,ade20k]' data_root=/data/bench-data datasets.voc2012=/data/datasets/VOCdevkit/VOC2012 datasets.ade20k=/data/datasets/ADEChallengeData2016)
```

The prepared manifests record those source locations, which evaluation then uses. `~` and relative paths are resolved against the caller's environment. No source images are copied into task manifests. Setup and visualization require neither a model nor CUDA and do not import PyTorch. The visualization command writes a standalone `index.html`, local annotated PNGs, and a selection manifest, using only development examples.

## Supported reference models

Supported benchmark references must have an established public checkpoint or model release with documented upstream validation, loading and preprocessing. The current reference architectures are torchvision ResNet18 and ViT-B/16 with official ImageNet weights. The DINOv3 adapter targets official public releases; each exact local checkpoint still requires qualification. Private or experimental model studies are outside the supported set and release requirements. See [the reference model policy](docs/reference-models.md).

Caller-owned models use the same adapter interface. Seeded untrained controls exercise calibration and correctness; the published pretrained references are official ResNet18 and ViT-B/16.

Runtime inputs are the datasets and your model, supplied through an adapter or local checkpoint. Recipes, fixed class lists, pretrained reference scores and monitoring interval tables ship with the package. `bench prepare` generates deterministic sample manifests locally. Probes are fitted on your model's frozen features; the encoder is never trained by `bench.run`. You can compare against the included reference scores without retraining reference models or downloading acceptance checkpoints, fitted heads or feature banks. Reproducing a reference evaluation requires that reference's original public weights.

## Included reference scores

`reference-scores.json` contains fourteen development references: the two official pretrained architectures on all seven tasks. Each entry retains its checkpoint, model-state, numerical profile, task recipe, source/preparation binding, metrics and original measurement provenance. Full searched probes include the prescribed three-seed mean and standard deviation; monitoring uses its single declared seed. These are measured scores under this package's protocols. Inspect evaluation exposure before describing a comparison as held out.

```python
from importlib.resources import files
import json

references = json.loads(files("bench").joinpath("reference-scores.json").read_text())
for entry in references["entries"]:
    print(entry["architecture"], entry["task"], entry["metrics"])
```

The same resource directory includes `monitoring-evidence.json` and `monitoring-intervals.csv`, preserving all 873 interval occurrences from six named reports, including uncertain directions and failed prefixes. Historical pending flags in the embedded source reports describe those reports' original status. The publication envelope identifies their scope. Reading these packaged tables requires no datasets, model weights or historical artifact downloads. The JSON schemas allow additive 1.x fields; readers must reject unsupported major versions. The monitoring JSON binds the CSV by checksum.

## Evaluate an adapter

```python
from pathlib import Path
import bench
from my_model.benchmark import EncoderAdapter

# The model repository implements and loads this adapter.
model = EncoderAdapter.from_checkpoint("outputs/checkpoints/best.pt").to("cuda")
result = bench.run(
    model,
    names=["imagenet-100", "pascal"],
    evaluation="development",
    data_root=Path("~/datasets"),
    output_dir=Path("outputs/bench/checkpoint-development"),
    execution=bench.Execution(device="cuda", batch_size=64, numerics="fp32-v1"),
)
print(result.metrics)
```

The public types are `VisionModel`, `ModelInfo`, `ImageBatch`, `Extraction`, `SpatialFeatures`, `Execution`, and `RunResult`. Their fields follow [the API contract](docs/api.md) and [SPEC.md](SPEC.md). Classification requests only `encode_global`; spatial tasks request only `encode_spatial`. The adapter owns normalization, readout construction, and reproducible per-example sampling. Inputs contain RGB pixels, padding validity, affine geometry, and opaque IDs; annotations stay in the task implementation.

Existing models do not need to implement this API themselves. Put a `VisionModel` wrapper in the model repository, register the inference modules on it, and implement whichever readouts the architecture supports. Start with `encode_global` when only classification is available. A spatial adapter must describe its actual native grid centers and full represented extent; [the torchvision adapters](src/bench/adapters/torchvision.py) show both convolutional and patch-token geometry. The runner discovers feature widths from the returned tensors.

Include model-library versions in the adapter recipe alongside its source revision. The reference wrappers record their libraries automatically, including torchvision and Transformers where applicable. The runner records PyTorch, NumPy, Pillow and image-processing source identity so changed dependencies invalidate affected results and probes.

Evaluation identities also include the runner and relevant scoring source. After changing that code, use a fresh output directory. Selected probes are reused only when their fitting, normalization and scoring implementations still match; these checks preserve the model's extraction identity.

FP32 is the default numerical profile. CPU execution uses `Execution(device="cpu")`. The explicit `bf16-v1` option requires a compatible CUDA device and qualified model/task parity for reference comparisons. Extraction batch size does not change the probe's effective batch, seed, or training budget.

Use an explicit device such as `Execution(device="cuda:1")` to select a GPU, and place the adapter on that device. An unindexed `cuda` means the caller's current GPU. Evaluation restores the caller's current device afterward.

Monitoring uses `fp32-v1` throughout. ImageNet-100 has 10,000 training images, 7,500 development images and 400 probe updates. PASCAL uses 1,024 training images, 256 development images, a 160-pixel canvas and 12,800 updates. Each monitoring task uses seed zero and final-update scoring. Full ImageNet-1K, ADE20K and CLEVR tasks use the documented development search, budget checks and three reporting seeds. SPair uses cosine matching without a fitted head. Monitoring and full tasks retain distinct identities and scores.

Scientific acceptance retains all paired intervals and half-widths, with no absolute width cap. Meaningful fuller-reference changes are two top-1 points or one mIoU point with an interval excluding zero. At least 90% of qualifying directions must also be confidently detected by monitoring, with at least three qualifying pairs covering CNN and ViT. Protected development and six prospectively reserved checkpoints pass the declared tests; all five fresh qualifying directions agree for each task. These findings describe the declared adapters and checkpoints.

Measured standalone two-task calls take 112.49–116.27 seconds for the declared ModelZoo CIFAR ResNet18 and LAION CLIP ViT-B/14 adapters. The qualified official torchvision CNN implementation takes 79.77–91.39 seconds over three fresh calls; one official ViT call takes 116.58 seconds. Earlier failed candidates remain in the acceptance ledger. Actual matched training overhead is 3.67% for the declared CNN setup and 3.45% for ViT, with the specified optimizer-step cadences. Those timing and overhead measurements retain their source, adapter, data, hardware and cadence scope; measure your own training integration.

BF16 qualification applies only to the measured model, task, recipe, source and hardware. Seven historical ImageNet-100 monitoring comparisons and an earlier untrained-ViT classification control exceed the unchanged 0.1-point tolerance; those failures remain restricted to FP32. Earlier passing comparisons do not qualify the new classification bank or every full task. See [numerical profiles](docs/numerical-profiles.md) for the retained failures and [the verification summary](docs/acceptance.md) for measured scope.

Reference adapters accept torchvision ResNet/ViT state dictionaries and local Hugging Face DINOv3 checkpoint directories. Their normalization and readouts are recorded in the extraction recipe. For example:

```nu
(uv run bench model=resnet18 model.checkpoint=/models/resnet18.pth 'names=[imagenet-100,pascal]' execution.numerics=fp32-v1 data_root=/data/bench-data output_dir=outputs/bench/resnet18-development)
(uv run --extra dinov3 bench model=dinov3 model.checkpoint=/models/dinov3-hf 'names=[imagenet-100]' execution.numerics=fp32-v1 data_root=/data/bench-data output_dir=outputs/bench/dinov3-development)
```

Model configurations use Hydra `_target_` factories, which receive ordinary Python containers including nested provenance. Python callers do not use Hydra. Hydra never changes the working directory. Task overrides and duplicate resolved benchmark identities are rejected.

## Results and workspace lifetime

Results retain `resolved.yaml`, `manifest.json`, `results.json`, and `statistics/`. Each completed task commits its metrics and compact statistics before the next starts. A later failure preserves completed tasks and identifies unfinished work. Accuracy, PCK, and mIoU are percentages; seed variability and evaluation partitions remain explicit.

Temporary feature shards are verified before reuse. Each tensor shard is written atomically, followed by its atomic index entry; recovery reuses every verified committed entry. Completed-result reuse invokes no encoder. A standalone successful run removes its `.work` directory. An explicit training workspace retains selected probes for the most recent model identity, allowing final evaluation to reuse development selection. New model identities replace obsolete workspace state. Final evaluation closes the workspace. To end or abandon training explicitly:

```nu
(uv run bench prune work_dir=/path/to/training/.bench-work)
```

Pruning takes the same exclusive lock, refuses active or unowned state, and preserves unrelated files. Source data, prepared manifests, reports, checkpoints, and results are outside the disposable feature/probe directories.

Feature reads use bounded memory mappings. Dense banks retain at most 256 MiB and 1,024 shards. Global banks can expand that cache to one eighth of Linux-reported available host memory, capped at 4 GiB and 16,384 shards; the baseline limit applies when memory information is unavailable or too low to expand it. These are cache limits, not limits on total process memory.

Training integrations should evaluate immutable saved checkpoints in a separate process, choose their own benchmark lists and cadence, and log completed local metrics through their existing tracker. The package does not schedule jobs, select training checkpoints, or own tracker runs.

When monitoring and broader evaluations coincide, combine their names into one request for that checkpoint. Use a checkpoint/list/partition-specific output directory and one shared workspace per training job. Repeating a completed request returns its committed results; concurrent requests sharing an output or workspace report `Busy` for the caller to retry. Evaluation frequency remains an uncalibrated caller choice.

For example, the caller can consume saved-checkpoint notifications synchronously. Here each notification contains `(epoch, checkpoint, evaluation, selection_rule)`, where `evaluation` is `development` or `final`, and `selection_rule` describes how that checkpoint was chosen. The DINOv3 example expects immutable local Hugging Face checkpoint directories. The caller supplies the final selected checkpoint; the loop does not select one.

```python
import hashlib
import json
from pathlib import Path
import subprocess


def evaluate_checkpoints(notifications, *, data_root, output_root, work_dir):
    completed = set()
    work_dir = str(Path(work_dir).expanduser().resolve())
    for epoch, checkpoint, evaluation, selection_rule in notifications:
        names = ["imagenet-100", "pascal"]
        if epoch % 25 == 0 or evaluation == "final":
            names += ["imagenet-1k", "ade20k", "spair71k", "clevr-count", "clevr-distance"]
        checkpoint = str(Path(checkpoint).expanduser().resolve())
        request = json.dumps([checkpoint, names, evaluation])
        if request in completed:
            continue
        output = Path(output_root).expanduser().resolve() / hashlib.sha256(request.encode()).hexdigest()
        subprocess.run([
            "bench", "model=dinov3",
            "model.checkpoint=" + json.dumps(checkpoint),
            f"model.provenance={{training_epoch:{epoch},model_selection:{json.dumps(selection_rule)}}}",
            "names=" + json.dumps(names), "evaluation=" + evaluation,
            "data_root=" + json.dumps(str(Path(data_root).expanduser().resolve())),
            "output_dir=" + json.dumps(str(output)),
            "execution.work_dir=" + json.dumps(work_dir),
        ], check=True)
        completed.add(request)
        if evaluation == "final":
            return  # Successful final evaluation already closed the workspace.
    subprocess.run(["bench", "prune", "work_dir=" + json.dumps(work_dir)], check=True)
```

An epoch-25 notification produces one seven-task request. Duplicate notifications within this invocation launch nothing; after a caller restart, the same output paths let the runner reuse committed results or resume verified unfinished work. Evaluation blocks this caller, so there is no background evaluation queue in the example and its waiting time counts toward training overhead. A failed command leaves unfinished work available for retry; use the pruning command above if abandoning it. Ending the notification stream without final evaluation prunes the idle workspace. The interval of 25 epochs illustrates list selection, not a validated cadence.

Before advertising a cadence, measure the same fixed interval of optimizer steps with evaluation disabled and enabled. Restore the same training checkpoint, optimizer, scheduler, RNG state and input position for both trials. Keep the hardware allocation, input batches, warmup procedure and checkpoint-save schedule identical. Synchronize training work at the timing boundaries and include evaluator startup, result collection and cleanup in the enabled trial. If training overlaps evaluation, wait for all evaluations triggered within the interval before stopping its timer.

Record the two wall times as `T_off` and `T_on`, and report overhead as `100 × (T_on − T_off) / T_off` percent. Retain the step interval, checkpoint identity, allocation, adapter recipe, numerical profile, cadence and evaluation start/completion times with the measurements. Every result must arrive before the next scheduled evaluation; report separate evaluation GPU time when using another GPU. The acceptance limit is 5%. The single-checkpoint monitoring harness supplies evaluation latency, while this comparison requires measurements from the training caller.

## Validation and model comparisons

```nu
(uv run pytest -m 'not gpu')
(uv run pytest -m gpu)
```

To run the complete CPU suite in two isolated workers with a 30-second target,
use `(uv run python scripts/test_cpu.py --jobs 2 --output outputs/cpu-tests)`.
Choose a fresh output directory for each run; it retains both test reports and
a combined timing and coverage receipt. The serial command remains available.

CPU tests use small fixtures, separate from scientific recipes. The GPU job checks BF16 geometry on a CUDA host. Full source acquisition, real galleries, checkpoint evaluations, numerical parity, and calibration are explicit acceptance work.

[examples/compare_torchvision.py](examples/compare_torchvision.py) runs pretrained and seed-zero untrained ResNet18 or ViT-B/16 controls on ImageNet-100 and VOC development data by default; use `--names` to select other task recipes. It accepts a JSON list of weight records with `architecture`, `checkpoint`, `sha256`, `pretraining`, and `model_selection` fields; checkpoint files remain caller-owned. The local acceptance weight manifest records the official torchvision download URLs and selection exposure. For example:

```nu
(uv run python examples/compare_torchvision.py --architecture resnet18 --weights-manifest outputs/acceptance/reference-checkpoints/manifest.json --data-root outputs/acceptance/data --output outputs/reference-controls/resnet18 --batch-size 32)
```

[examples/accept_spair_reference.py](examples/accept_spair_reference.py) verifies the matching kernel against checksum-pinned Probe3D functions with explicitly recorded convention adjustments. It requires the two upstream source files and CUDA, and does not install Probe3D's training framework.

[examples/accept_regeneration.py](examples/accept_regeneration.py) checks a completed standalone run against a fresh output using its recorded numerical profile and extraction batch. Its CLI constructs torchvision reference models; model integrations can call `verify_regeneration` with their own constructed CUDA adapter. It rejects encoder calls during result reuse, records fresh extraction calls after pruning, and compares task scores and compact statistics. Use the matching package source and dependency environment recorded by the baseline.

[examples/calibrate_monitor_probes.py](examples/calibrate_monitor_probes.py) and the acceptance scripts are development diagnostics for probe budgets, feature scaling and search boundaries. Their historical calibration identities remain distinct from the released task recipes. The [verification summary](docs/acceptance.md) records the measured scope and retained qualifications; end users do not rerun these diagnostics to use the included reference scores.

[examples/measure_dense_probe.py](examples/measure_dense_probe.py) measures feature reads, full-resolution loss/backward, accumulation parity, memory and cleanup using the SPEC's ADE-sized tensors. It separates first-batch phase timing from five repeated updates after warmup. Its synthetic inputs isolate kernel behavior; it does not report benchmark scores. Run it on a CUDA GPU with at least 10 GiB free:

```nu
(uv run python examples/measure_dense_probe.py --output outputs/dense-probe-kernels.json)
```

The internal `bench.analysis` module reconstructs paired differences from retained statistics. It groups repeated probe seeds by source image and stratifies ImageNet resampling by class. SPair deliberately refuses pair-independent confidence intervals.

Its `numerical_parity(fp32_output, bf16_output, task_name)` check compares matched FP32/BF16 results against the registered task tolerance. It verifies model/adapter/software identities, data partitions, feature shapes and complete sample/seed coverage, then reconstructs scores from committed statistics. The report preserves both profiles, checks individual probe seeds as well as aggregate metrics, and returns `passed=False` for excessive differences. Changed recipes, unmatched inputs or inconsistent reported scores raise an error. Passing this check does not merge profiles or enable cross-profile paired bootstrap comparisons.

[examples/monitoring_pair_gates.py](examples/monitoring_pair_gates.py) applies the monitoring precision, directional-agreement and doubled-budget reversal rules to verified paired differences. Its `summarize` function takes every predeclared pair's `family`, `first` and `second` checkpoint IDs plus `monitoring`, `reference` and `doubled` comparison reports. Use `None` for missing measurements, and supply the predeclared `minimum_qualifying_pairs` and detectable change `delta`. The caller must verify matched data/readouts and reference-budget adequacy first. Missing evidence, an undeclared minimum, or insufficient qualifying pairs across the supported CNN/ViT families cannot pass sensitivity. The output preserves every pair and reversal; latency, training overhead, reserved-checkpoint provenance and checkpoint score stability remain separate requirements.

The monitoring policy publishes every paired 95% interval and its half-width, including uncertain directions and nonqualifying pairs. It applies no absolute interval-width cap. At least 90% of qualifying fuller-reference pairs must have the same confidently detected direction in monitoring, with at least three qualifying pairs spanning CNN and ViT. Meaningful reference changes remain two top-1 points or one mIoU point, with a reference interval excluding zero. An interval that touches zero cannot establish confident direction. The [verification summary](docs/acceptance.md) records the agreed policy and measured scope.

References and dataset-source provenance are in [docs/references.md](docs/references.md) and the packaged task/acquisition manifests.
