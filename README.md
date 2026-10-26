# bench

Frozen visual representation benchmarks with one model adapter interface. Preparation, inspection, extraction, linear probes, correspondence scoring, compact results, and temporary-workspace pruning are implemented in Python 3.13.

**This is a 0.1 development implementation. The recipe identifiers in `SPEC.md` are candidates, not calibrated v1 releases.** Monitoring sample counts and update budgets are provisional. The release gates and current verification boundaries are recorded in [docs/acceptance.md](docs/acceptance.md).

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

The generic adapter API remains available for caller-owned models. Internal synthetic and seeded-untrained test controls exercise correctness and are not advertised pretrained benchmark references. Dataset tasks remain unchanged; task recipes and numerical profiles retain their recorded acceptance status.

## Evaluate an adapter

```python
from pathlib import Path
import bench

# The model repository implements and loads this adapter.
model = EncoderAdapter.from_checkpoint("outputs/checkpoints/best.pt").to("cuda")
result = bench.run(
    model,
    names=["imagenet-100", "pascal"],
    evaluation="development",
    data_root=Path("~/datasets"),
    output_dir=Path("outputs/bench/checkpoint-development"),
    execution=bench.Execution(device="cuda", batch_size=64),
)
print(result.metrics)
```

The public types are `VisionModel`, `ModelInfo`, `ImageBatch`, `Extraction`, `SpatialFeatures`, `Execution`, and `RunResult`. Their fields follow [SPEC.md](SPEC.md). Classification requests only `encode_global`; spatial tasks request only `encode_spatial`. The adapter owns normalization, readout construction, and reproducible per-example sampling. Inputs contain RGB pixels, padding validity, affine geometry, and opaque IDs; annotations stay in the task implementation.

Existing models do not need to implement this API themselves. Put a `VisionModel` wrapper in the model repository, register the inference modules on it, and implement whichever readouts the architecture supports. Start with `encode_global` when only classification is available. A spatial adapter must describe its actual native grid centers and full represented extent; [the torchvision adapters](src/bench/adapters/torchvision.py) show both convolutional and patch-token geometry. The runner discovers feature widths from the returned tensors.

Include model-library versions in the adapter recipe alongside its source revision. The reference wrappers record their libraries automatically, including torchvision and Transformers where applicable. The runner records PyTorch, NumPy, Pillow and image-processing source identity so changed dependencies invalidate affected results and probes.

Evaluation identities also include the runner and relevant scoring source. After changing that code, use a fresh output directory. Selected probes are reused only when their fitting, normalization and scoring implementations still match; these checks preserve the model's extraction identity.

CPU execution requires `Execution(device="cpu", numerics="fp32-v1")`. The default `bf16-v1` profile requires a compatible CUDA device. Extraction batch size does not change the probe's effective batch, seed, or training budget.

Use an explicit device such as `Execution(device="cuda:1")` to select a GPU, and place the adapter on that device. An unindexed `cuda` means the caller's current GPU. Evaluation restores the caller's current device afterward.

Numerical-profile acceptance is still provisional: the untrained ViT reference control exceeded the candidate score tolerance on ImageNet-100. Use `fp32-v1` for comparisons involving that control until calibration resolves this failure; the measured differences are in [the acceptance ledger](docs/acceptance.md).

Reference adapters accept torchvision ResNet/ViT state dictionaries and local Hugging Face DINOv3 checkpoint directories. Their normalization and readouts are recorded in the extraction recipe. For example:

```nu
(uv run bench model=resnet18 model.checkpoint=/models/resnet18.pth 'names=[imagenet-100,pascal]' data_root=/data/bench-data output_dir=outputs/bench/resnet18-development)
(uv run --extra dinov3 bench model=dinov3 model.checkpoint=/models/dinov3-hf 'names=[imagenet-100]' data_root=/data/bench-data output_dir=outputs/bench/dinov3-development)
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

Record the two wall times as `T_off` and `T_on`, and report overhead as `100 × (T_on − T_off) / T_off` percent. Retain the step interval, checkpoint identity, allocation, adapter recipe, numerical profile, cadence and evaluation start/completion times with the measurements. Every result must arrive before the next scheduled evaluation; report separate evaluation GPU time when using another GPU. The proposed acceptance limit is 5%. The single-checkpoint monitoring harness supplies evaluation latency, while this comparison requires measurements from the training caller.

## Validation and model comparisons

```nu
(uv run pytest)
(uv run pytest -m gpu)
```

CPU tests use small fixtures, separate from scientific recipes. The GPU job checks BF16 geometry on a CUDA host. Full source acquisition, real galleries, checkpoint evaluations, numerical parity, and calibration are explicit acceptance work.

[examples/compare_torchvision.py](examples/compare_torchvision.py) runs pretrained and seed-zero untrained ResNet18 or ViT-B/16 controls on ImageNet-100 and VOC development data by default; use `--names` to select other complete candidate recipes. It accepts a JSON list of weight records with `architecture`, `checkpoint`, `sha256`, `pretraining`, and `model_selection` fields; checkpoint files remain caller-owned. The local acceptance weight manifest records the official torchvision download URLs and selection exposure. For example:

```nu
(uv run python examples/compare_torchvision.py --architecture resnet18 --weights-manifest outputs/acceptance/reference-checkpoints/manifest.json --data-root outputs/acceptance/data --output outputs/reference-controls/resnet18 --batch-size 32)
```

[examples/accept_spair_reference.py](examples/accept_spair_reference.py) verifies the matching kernel against checksum-pinned Probe3D functions with explicitly recorded convention adjustments. It requires the two upstream source files and CUDA, and does not install Probe3D's training framework.

[examples/accept_regeneration.py](examples/accept_regeneration.py) checks a completed standalone run against a fresh output using its recorded numerical profile and extraction batch. Its CLI constructs torchvision reference models; model integrations can call `verify_regeneration` with their own constructed CUDA adapter. It rejects encoder calls during result reuse, records fresh extraction calls after pruning, and compares task scores and compact statistics. Use the matching package source and dependency environment recorded by the baseline.

[examples/calibrate_monitor_probes.py](examples/calibrate_monitor_probes.py) measures doubled probe budgets and feature scaling on the two monitoring sample banks using a Hydra adapter-constructor YAML. Its `--stage learning-rate` check compares fixed learning-rate multipliers of 1, 3, 10 and 30 at both 100 and 200 updates. It declares separate calibration recipes before fitting, retains compact statistics and removes temporary banks and heads. The reference controls show that the current 100-update budgets need further calibration; these diagnostic results do not establish release acceptance or a recommended cadence.

The `--stage convergence` diagnostic continues those four learning-rate settings at four and eight times the named monitoring budget (currently 400 and 800 updates). It keeps the same sample banks and final-update scoring, and reports the doubled-budget difference against the same adequacy tolerance. Longer diagnostic budgets do not change the published candidates or establish monitoring latency.

The `--stage variance-floor` diagnostic repeats the original budget and scale checks with an absolute training-variance floor of 1e-8. Each variant has a separate identity and records the named recipe's original floor. This development diagnostic tests normalization sensitivity on supported public reference representations before changing a named recipe.

The `--stage fuller-reference` diagnostic uses those same learning-rate and update settings with every official training image outside the fixed monitoring development bank. It preserves the ImageNet-100 classes, labels, preprocessing, readout and development/final partitions. Its separate calibration identities and shared preparation manifests leave the named monitoring recipes unchanged. First use prepares and verifies the larger banks, recording preparation time separately; later evaluations reuse them without reading final examples. This supplies the larger-bank budget check required for a same-task reference. Its training budget still needs validation, and checkpoint-pair agreement and training overhead still need suitable training runs.

The `--stage fuller-convergence` diagnostic uses those larger banks and four learning rates at 50/100 complete training passes for ImageNet-100 and 40/80 for VOC. It converts these budgets to fixed update counts using the actual training-bank size and the prescribed effective batch, retaining final-update scoring and seed zero. On the prepared full banks this gives 6,100/12,200 and 3,040/6,080 updates respectively. These are longer development candidates for resolving undertraining; adequate convergence and useful predictions still require measurement before choosing a reference recipe.

The `--stage fuller-selection` diagnostic uses the same larger banks, learning rates and complete training budgets, but scores each epoch and selects the best development score, breaking ties by the earlier epoch. Separate `-bestdev` identities record this selection change. CPU checks verify selection and cleanup; real-data budget stability and checkpoint-pair agreement remain unvalidated. The named monitoring recipes retain final-update selection.

For full probes, `--stage search-boundary` uses 1, 3 and 10 times the named recipe's largest learning rate at the original and doubled epoch budgets. It preserves the weight-decay grid and uses seed zero for development selection. Select the full task with `--names`; every variant retains all candidate scores. Full-probe comparisons report score changes without applying the monitoring acceptance thresholds, and leave the named recipes unchanged.

The `--stage search-extension` diagnostic tests multipliers 1, 3, 10, 30 and 100 at twice and four times the named epoch budget. It retains the earlier doubled-budget variants with identical recipe identities so their records and selected heads can be compared directly. For CLEVR this means rates 0.01 through 1.0 at 100 and 200 epochs; wider-search measurements remain pending.

The `--stage search-weight-decay` diagnostic extends a searched weight-decay grid with ten times its upper value, preserving all named learning rates and testing the original and doubled epoch budgets with seed zero. The current classification grid becomes 0, 0.1 and 1.0. It rejects fixed weight-decay recipes, records separate calibration identities, and leaves named recipes unchanged. A winner at the original upper boundary requires this coverage check before release; the diagnostic itself does not establish an adequate search range.

The `--stage search-grid` diagnostic extends both axes together when their upper boundaries coincide. It adds three and ten times the largest named learning rate and ten times the largest weight decay, retaining the original grid and epoch budget plus a doubled-budget search. For full ImageNet-1K this gives rates 0.001, 0.003, 0.01, 0.03 and 0.1; weight decays 0, 0.1 and 1.0; and 50/100 epochs. Each budget searches all 15 combinations with seed zero under separate calibration identities. Overlapping candidates can be checked against the original search before interpreting the new combinations.

[examples/measure_dense_probe.py](examples/measure_dense_probe.py) measures feature reads, full-resolution loss/backward, accumulation parity, memory and cleanup using the SPEC's ADE-sized tensors. It separates first-batch phase timing from five repeated updates after warmup. Its synthetic inputs isolate kernel behavior; it does not report benchmark scores. Run it on a CUDA GPU with at least 10 GiB free:

```nu
(uv run python examples/measure_dense_probe.py --output outputs/dense-probe-kernels.json)
```

The internal `bench.analysis` module reconstructs paired differences from retained statistics. It groups repeated probe seeds by source image and stratifies ImageNet resampling by class. SPair deliberately refuses pair-independent confidence intervals.

Its `numerical_parity(fp32_output, bf16_output, task_name)` check compares matched FP32/BF16 results against the registered task tolerance. It verifies model/adapter/software identities, data partitions, feature shapes and complete sample/seed coverage, then reconstructs scores from committed statistics. The report preserves both profiles, checks individual probe seeds as well as aggregate metrics, and returns `passed=False` for excessive differences. Changed recipes, unmatched inputs or inconsistent reported scores raise an error. Passing this check does not merge profiles or enable cross-profile paired bootstrap comparisons.

[examples/monitoring_pair_gates.py](examples/monitoring_pair_gates.py) applies the monitoring precision, directional-agreement and doubled-budget reversal rules to verified paired differences. Its `summarize` function takes every predeclared pair's `family`, `first` and `second` checkpoint IDs plus `monitoring`, `reference` and `doubled` comparison reports. Use `None` for missing measurements, and supply the predeclared `minimum_qualifying_pairs` and detectable change `delta`. The caller must verify matched data/readouts and reference-budget adequacy first. Missing evidence, an undeclared minimum, or insufficient qualifying pairs across the supported CNN/ViT families cannot pass sensitivity. The output preserves every pair and reversal; latency, training overhead, reserved-checkpoint provenance and checkpoint score stability remain separate requirements.

References and dataset-source provenance are in [docs/references.md](docs/references.md) and the packaged task/acquisition manifests.
