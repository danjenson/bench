# Minimal vision benchmark package specification

Status: proposed implementation contract. The Python examples describe the intended package, not functionality already implemented. Target repository: `bench`.

## Purpose and scope

Build a small PyTorch package that evaluates newly trained visual representations through a stable model interface. A researcher should write one adapter for a new model, then reuse the same datasets, probe training, metrics, and result format across checkpoints.

The first release evaluates frozen global and spatial image representations, with glimpse-based SSL models as a primary use case. A model-specific adapter exposes separate calls for a single classification vector and a spatial grid of tokens, and owns the procedure that produces each. The benchmark measures recognition, scene parsing, correspondence, counting, and distance information separately, without combining them into an overall model score. Scores describe information accessible through the specified readout, not all information a representation might contain.

The implementation should reuse established dataset definitions and evaluation algorithms while owning a small, explicit protocol. It should not import several complete training frameworks into one environment. Runtime optimization must preserve the declared experiment.

Aggressively prune derived state. Retain source datasets, compact deterministic preparation recipes, sample manifests, and evaluation results; regenerate task inputs and model-specific features when needed. Only preparation independent of the evaluated model and reusable across all models under the same task protocol may persist as shared prepared data. Model-specific working state must not become an archive of training checkpoints.

Out of scope for the first release: SSL pretraining, backbone fine-tuning, detection training, language evaluation, video, model-specific acquisition or policy engines, distributed scheduling, compilation support, custom task overrides, a dashboard, and a general plugin system. NYUv2 and Something-Something-v2 are possible later extensions.

## Benchmark selection

Callers select an explicit list of named benchmarks. Each name resolves to one fixed, versioned task recipe.

| Name | Task identifier | Dataset | Readout and primary metric |
| --- | --- | --- | --- |
| `imagenet-100` | `imagenet100-linear-v1` | ImageNet-100 | Frozen global features, linear classifier; top-1 accuracy |
| `pascal` | `voc2012-linear-v1` | PASCAL VOC2012 | Frozen spatial features, linear pixel classifier; mIoU |
| `imagenet-1k` | `imagenet1k-linear-v1` | ImageNet-1K | Frozen global features, linear classifier; top-1 accuracy |
| `ade20k` | `ade20k-linear-v1` | ADE20K scene parsing challenge | Frozen spatial features, linear pixel classifier; mIoU |
| `spair71k` | `spair71k-match-v1` | SPair-71k | Cosine feature matching; PCK@0.1 |
| `clevr-count` | `clevr-count-v1` | CLEVR | Frozen global features, linear classifier; accuracy |
| `clevr-distance` | `clevr-distance-v1` | CLEVR | Frozen global features, linear classifier; accuracy |

Use the established CLEVR Count and Distance target definitions from VISSL/VTAB, including their bins. Distance classifies the closest object's distance, not distances between objects. This is image probing, not the full CLEVR question-answering task or a direct test of relational binding. Pin the source revision and record the target-generation rules in the task manifest before releasing these identifiers.

The `imagenet-100` and `pascal` recipes are intended for frequent monitoring: fixed, bounded sample banks, one probe seed, and a short, fixed probe budget without a hyperparameter sweep. Pin the ImageNet-100 class list and all sample IDs by checksum. Exact sample counts and update budgets require calibration before release; the names alone do not guarantee a runtime. These scores retain their own identities and are not substitutes for ImageNet-1K or ADE20K results. Tiny artificial fixtures belong in tests, not in the scientific benchmarks.

Callers can run `['imagenet-100', 'pascal']` each epoch and select a broader list every 25 epochs and at completion. Adding a benchmark to the list never changes the recipes of the other selected benchmarks. A different budget for the same dataset requires a distinct recipe identity, not an implicit mode switch.

For SPair, fit each complete image into a 448 square and pad. Retain per-category and viewpoint/scale/occlusion breakdowns as supporting metrics. Define PCK distance using 0.1 times the maximum side of the target object's bounding box in the original image. Bilinearly sample each source-keypoint feature, normalize vectors, and match to the valid target feature location with highest cosine similarity, without a learned matching head. Break ties by raster order. Headline PCK is correct eligible correspondences divided by all eligible correspondences. Object boxes are used only for scoring. Record interpolation and eligibility rules in the protocol; do not silently substitute a more powerful matching algorithm.

Report the grid-limited PCK ceiling: the fraction of eligible target keypoints whose nearest valid target feature location falls within the same PCK threshold. This diagnostic uses evaluation keypoints only after extraction and bounds the score attainable by the declared discrete matching rule. It is not a replacement for measured PCK. Probe3D averages category scores in its headline result; this task pools correspondences. Reference checks must distinguish matching-rule parity from aggregation and preprocessing differences.

## Public Python API

Expose one runner, `bench.run`, a PyTorch adapter base, and the small data types used by that interface. Internal dataset loaders, probes, caches, and framework-specific code are not public APIs.

```python
from pathlib import Path
import bench
from my_model.benchmark import EncoderAdapter

model = EncoderAdapter.from_checkpoint("outputs/run/checkpoints/best.pt")
model = model.to("cuda")

result = bench.run(
    model,
    names=["imagenet-100", "pascal"],
    evaluation="development",
    data_root=Path.home() / "datasets",
    output_dir=Path("outputs/bench/run-best"),
    execution=bench.Execution(device="cuda", batch_size=64),
)

print(result.metrics)
```

`EncoderAdapter` above belongs to the model's repository. `bench` must not require importing its training loop, optimizer, or experiment tracker to evaluate it.

Proposed runner signature:

```python
def run(
    model: VisionModel,
    *,
    names: list[str],
    evaluation: Literal["development", "final"] = "development",
    data_root: Path,
    output_dir: Path,
    execution: Execution | None = None,
) -> RunResult:
    ...
```

`names` is a nonempty list of published short names or exact versioned task identifiers. Short names resolve to the documented identifiers above; record both the requested names and resolved identities in results. Published aliases never silently retarget a different recipe. Unknown names, duplicate resolved tasks, unsupported outputs, or incompatible input shapes fail preflight with an actionable message. Callers select compatible tasks explicitly; the runner never silently drops tasks. Preflight checks required datasets, split manifests, and writable artifact/workspace paths before extraction. Missing or incomplete data produces the expected layout and a matching `bench prepare` command with the requested names and data root; `run` does not automatically download datasets.

`evaluation="development"` fits probes on training data and reports monitoring scores on development data. It never reads the final partition. `evaluation="final"` applies the same selected benchmark recipes to their held-out partitions after any probe selection on development data. Every result identifies its evaluation partition. Evaluation partition and benchmark selection are independent choices.

`Execution` has the fields `device`, `batch_size`, `workers`, `work_dir`, and `numerics`. Their defaults are CUDA, 64 extraction samples, 8 workers, no shared workspace, and `bf16-v1`. With `work_dir=None`, use `output_dir/.work` and remove it on successful completion. An explicit `work_dir` is a temporary training workspace shared across evaluations, with the bounded lifetime defined below. It is not a persistent feature-cache directory. Verified interrupted work resumes automatically; there is no resume toggle. The alternative `fp32-v1` numerical profile is explicit and recorded separately. `RunResult.metrics` is `dict[str, dict[str, float]]`; richer metric metadata belongs in the versioned result artifact.

The stable model seam is:

```python
from dataclasses import dataclass
from typing import Literal
from torch import Tensor, nn

@dataclass(frozen=True)
class ModelInfo:
    name: str
    revision: str
    readouts: dict[Literal["global", "spatial"], str]
    recipe: dict[str, object]

@dataclass(frozen=True)
class ImageBatch:
    images: Tensor
    valid_pixels: Tensor
    image_to_original: Tensor
    sample_ids: tuple[str, ...]

@dataclass(frozen=True)
class SpatialFeatures:
    values: Tensor
    grid_to_original: Tensor
    extent_in_original: Tensor

@dataclass(frozen=True)
class Extraction:
    features: Tensor | SpatialFeatures
    costs: dict[str, Tensor] | None = None

class VisionModel(nn.Module):
    info: ModelInfo

    def encode_global(self, batch: ImageBatch) -> Extraction:
        """Return features as one [B, Dg] tensor for classification."""
        raise NotImplementedError

    def encode_spatial(self, batch: ImageBatch) -> Extraction:
        """Return SpatialFeatures with [B, Ds, Hf, Wf] grid values."""
        raise NotImplementedError
```

Each adapter declares one fixed extraction procedure and readout per supported output. `readouts` is a nonempty mapping whose keys declare capabilities and whose values identify the corresponding readouts, for example `{"global": "mean_scene_prediction", "spatial": "predicted_scene_grid"}`. Only declared methods must be implemented. Classification tasks call `encode_global`; segmentation and correspondence call `encode_spatial`. The runner validates the returned feature type and shape for that method. It does not call a generic model `forward`, choose pooling for classification, or manufacture a spatial grid from a global vector. The two methods may use different native model APIs, layers, feature widths, and extraction procedures. Any pooling, query construction, or grid assembly belongs to the adapter.

`revision` identifies the adapter source. `recipe` is a JSON-serializable record of the resolved configuration, including normalization, each method's layer selection and readout construction, any internal sampling, resource limits, and seeds. The runner records and hashes the readout mapping and entire recipe without interpreting model-specific extraction fields, and independently hashes all model parameters and buffers. All learned submodules used in either method must be registered on the adapter; other external state must be fingerprinted in the recipe. The first release must exercise this seam with a conventional CNN, a ViT, and a real glimpse-based SSL adapter before freezing the interface.

### Input and output contract

- `images` is `[B, 3, H, W]`, RGB floating-point values in `[0, 1]`, transformed according to the task recipe and placed on the execution device. Each entry is one static image example.
- `valid_pixels` is Boolean `[B, 1, H, W]` and identifies image content versus padding. It contains no semantic labels.
- `image_to_original` is `[B, 3, 3]`. Each homogeneous affine matrix maps normalized input-canvas coordinates to normalized original-image coordinates. Both frames use x rightward, y downward, origin at the upper left, and extent `[0, 1]`. Image sizes may differ before resizing.
- `sample_ids` provides stable, opaque identifiers for deterministic per-example sampling and result alignment. IDs must not expose labels or class directory names. No labels, target embeddings, keypoints, object boxes, or annotation-bearing paths reach the adapter.
- `encode_global` returns an `Extraction` whose `features` is a finite `[B, Dg]` tensor: one feature vector per image, not class logits or a sequence of glimpse vectors. `encode_spatial` returns an `Extraction` whose `features` is `SpatialFeatures`, containing finite `[B, Ds, Hf, Wf]` values and finite, invertible `[B, 3, 3]` `grid_to_original` affine matrices. `Dg` and `Ds` may differ. A feature cell at row i and column j has normalized grid coordinates `((j + 0.5) / Wf, (i + 0.5) / Hf)` before that mapping. The matrix must encode the actual stride and offset; grids are not assumed to span the image uniformly.
- `extent_in_original` is finite `[B, 4]`, with each row giving `(xmin, ymin, xmax, ymax)` in normalized original-image coordinates, with `xmin < xmax` and `ymin < ymax`. It declares the rectangular image domain represented by the spatial grid, including any predictions for unobserved regions. It is distinct from observed pixel coverage, receptive-field size, and the boundary inferred from feature-center spacing. For example, a 2×2 grid with centers at 30% and 70% of each image axis can represent the full image with extent `(0, 0, 1, 1)` while preserving its actual centers in `grid_to_original`.
- Spatial outputs must represent all supplied image content. Preflight checks that the declared extent covers it; adapter acceptance tests verify that the extent describes the actual readout. A cropped or incomplete grid is unsupported in v1. Task scoring eligibility depends on annotations and the supplied canvas, never on adapter coverage. Sample source-keypoint features and segmentation logits bilinearly with `align_corners=False` and border replication beyond the outer feature centers within the declared extent. Target matching locations are feature centers inside valid image content; padding is ineligible. Pin these rules and label interpolation in each spatial task manifest. Coarse grids remain coarse for matching and its grid-limited ceiling; interpolation does not create additional native token locations.
- The runner owns the task's supplied image canvas, padding, and label transforms. The adapter owns normalization and its declared internal extraction procedure. It may transform or selectively access the supplied canvas, but must not load additional image pixels or annotations. Changing the supplied canvas requires a different task protocol.
- `costs`, when supplied, maps unit-qualified names to nonnegative finite `[B]` tensors, one measurement per example. Record actual work, including all internal passes. The runner preserves these measurements with temporary features and retains summaries in results; absent measurements mean unknown, not zero. Wall time and memory are measured by the runner separately.
- The runner calls only the method needed by the task, including during cold preflight. An adapter declaring only `global` can run classification tasks; one declaring only `spatial` can run dense tasks. Unsupported task lists fail before extraction. Feature width and grid size are discovered per method and recorded, not forced to match across methods or architectures. Verified bank or fitted-probe metadata supplies this information on reuse without either extraction method being called. Each method must work independently of whether the other was called earlier.

An adapter may perform multiple internal steps and stop independently for each example. This control flow belongs to the adapter; the benchmark imposes no observation count, query scheme, or policy interface. Any access restrictions claimed by an experiment must be enforced and tested by its model integration. Sampling must be reproducible from the declared recipe and sample ID, independent of extraction batch size, order, or resumption. The adapter returns one final representation per example for the selected recipe.

The caller owns the model and places it on the requested device. Adapters must implement side-effect-free extraction: no mutation of parameters, buffers, gradients, or external state, and no consumption of caller-owned random generators. The runner temporarily uses evaluation mode and inference mode for adapter execution, restores individual module training flags and any runner-managed global RNG state in a `finally` path, and uses private generators for probe initialization and sampling. It does not snapshot and restore arbitrary adapter state. Adapter acceptance tests verify unchanged model state and caller RNG streams on success and failure. Materialize ordinary detached tensors outside inference mode before probe training; detachment alone does not remove inference-tensor restrictions. Concurrent evaluation and training of the same live module is unsupported; training jobs should submit an immutable checkpoint to a separate evaluation process.

## Protocols and comparability

Keep three kinds of version independent:

1. Package version governs Python API compatibility.
2. Task versions govern datasets, preprocessing, probe training, and scoring.
3. Result schema version governs artifact parsing.

Each published task is immutable after release. A changed split, sample bank, label mapping, supplied image transform, probe search, probe seed set, metric, or probe checkpoint-selection rule requires a new task protocol identifier. Task fixes that change scores invalidate affected results and require a new task version.

Model weights, internal extraction choices, adapter normalization, selected layer, pooling, and readout belong to extraction identity. Changing them creates a new representation to evaluate under the same task. Benchmark feature standardization and any benchmark-controlled search over readouts belong to the task protocol. An adapter bug fix changes extraction identity and invalidates affected cached representations, without renaming an otherwise unchanged task.

Use semantic versioning for the package. Freeze the documented interface at 1.0 after the CNN, ViT, and glimpse-adapter acceptance gate. Internal modules carry no compatibility promise. Within 1.x, add optional capabilities or fields without breaking existing calls; remove or rename public fields only in a major release. Result readers reject unsupported major schema versions and tolerate unknown additive fields.

A result records the readable task identifier, resolved task digest, extraction identity, and evaluation partition separately. Task-recipe overrides are rejected in v1; changing a recipe requires publishing a distinct task identifier. Comparisons must show the readout, extraction recipe, input access, and resource budget; sharing a task identifier alone does not establish a matched experiment.

The initial recipes use fixed features, extracted once per example and extraction identity within active work and reused for all probe epochs. Regeneration repeats that extraction after pruning. They are not automatically equivalent to evaluations that repeatedly augment pixels during probe training. Use established implementations as references and compare baselines under the same recipe; do not attach published paper scores to a changed recipe.

### Glimpse-model comparisons

The model integration supplies a reproducible comparison script and condition manifest. For a declared supported budget `k > 1`, its classification comparison contains:

| Condition | Observations | Global readout |
| --- | --- | --- |
| `M1` | First glimpse | Mean raw encoder tokens |
| `C1` | First glimpse | Learned composed or predicted representation |
| `Mk` | First k distinct glimpses | Mean raw encoder tokens across those glimpses |
| `Ck` | Same k distinct glimpses | Learned composed or predicted representation |
| `Mk-repeat` | First glimpse repeated k times | Mean raw encoder tokens across repeats |
| `Ck-repeat` | Same k repeats | Learned composed or predicted representation |

Use the same checkpoint, source images, supplied canvases, observation seeds, crop resolution, and probe recipes. Distinct observations follow a fixed geometry-only sampling rule with a declared overlap limit; the model integration records their geometry. Budget curves use nested prefixes of those sequences. Repeats use identical pixels and geometry without new augmentation, and incur the declared repeated processing work. Fit a fresh probe for each condition. Match feature width where the model permits and always report feature width and probe parameter count. Unsupported conditions are explicit, with the corresponding comparisons unavailable; they cannot be silently replaced by a different readout. Label budgets outside the model's training range, including a one-glimpse composed readout when applicable, as extrapolation.

For a task score `S` where larger is better, report `S(Mk) - S(M1)` as the gain from additional observations under simple pooling, `S(Ck) - S(Mk)` as the gain from the learned readout at fixed observations, and `S(Ck) - S(Ck-repeat)` as the benefit of distinct evidence at fixed processing count. When `C1` is available, also report `[S(Ck) - S(Mk)] - [S(C1) - S(M1)]` to distinguish a larger multi-glimpse advantage from a learned readout that already helps with one glimpse. Report paired uncertainty for these differences. The learned readout may use extra compute; these comparisons alone do not establish equal-compute superiority or prove a particular internal mechanism.

Acceptance requires complete condition provenance, reproducible comparisons, and honest reporting of unsupported conditions, not positive gains. The release glimpse example must exercise the matched multi-glimpse pooling/composition and repeat/distinct controls. Its integration tests record actual crop accesses, verify that matched conditions receive identical pixels and geometry, check that `Mk-repeat` reproduces `M1` features within numerical tolerance for deterministic mean pooling, and check that changing unobserved pixels cannot affect fixed-observation outputs. A failing scientific hypothesis remains a valid benchmark result. Run the full matrix periodically and at completion; frequent monitoring may use a fixed declared subset whose total cost is calibrated separately.

Keep these comparisons in the model integration as separate adapter recipes passed to `bench.run`; they do not require a policy engine or new runner modes. Record whether a spatial readout contains observed features, predicted features, or a declared combination. A full-image encoder readout is a useful additional reference with its own input-access and cost declaration. Fixed benchmark canvases bound available detail: results on a 224-pixel canvas do not establish an advantage from acquiring high-resolution source pixels.

Glimpse comparisons claiming efficiency must supply `glimpses_count`, `encoder_pixels_count`, `encoder_passes_count`, `decoder_passes_count`, and `observed_area_fraction` in `Extraction.costs`. Count work per example across all internal passes, including repeated observations and any image-based acquisition preview. Encoder pixels count the pixel positions processed at encoder input resolution, including repeats and padding; an encoder pass is one image or crop encoding, independent of batching. Decoder passes count separate representation-prediction invocations per example, including separate query batches. Observed area is the union of inspected valid image regions divided by the supplied valid image area, in `[0, 1]`; it is not the predicted grid's extent. These quantities describe access and work, not architecture-independent FLOPs. Report them alongside measured latency and the extraction recipe.

### Initial frozen probe recipes

The following are proposed defaults for `imagenet-1k`, `clevr-count`, `clevr-distance`, and `ade20k`, to calibrate before v1 is frozen. Each named benchmark owns its resolved recipe.

| Setting | Global classification | Linear segmentation |
| --- | --- | --- |
| Supplied image | ImageNet: shorter side 256, center crop 224. CLEVR: fit entire image into 224 square and pad. | Fit entire image into 512 square and pad. |
| Feature bank | One fixed representation per source image | One fixed feature grid per source image |
| Readout | Train-only per-channel standardization, then linear classifier | Train-only per-channel standardization, then a shared linear classifier per location and geometric resampling of logits to the label canvas |
| Loss | Unweighted cross-entropy, mean over examples in the effective batch | Unweighted cross-entropy, sum over valid pixels divided by their count in the effective batch |
| Optimizer | AdamW | AdamW |
| Learning-rate candidates | 0.001, 0.003, 0.01 | 0.0001, 0.0003, 0.001 |
| Weight-decay candidates | 0, 0.1 | 0.01 |
| Training budget | 50 epochs, probe batch 1024 | 40 epochs, probe batch 16 |
| Schedule | 10% linear warmup, cosine decay to 0.001 times peak LR | Same |
| Probe seeds | 0, 1, 2 | 0, 1, 2 |
| Selection metric | Development top-1 | Development mIoU |

Use standardization statistics from the probe training partition only: over examples for global features and valid feature locations for spatial features. Pin weighting and the variance floor in the task manifest. Use FP32 for losses, statistics, and metric accumulation. Exclude padding and official ignore labels from segmentation loss and mIoU. Preserve full CLEVR scenes in the supplied image: cropping the task input would change counting targets. Recipes with a probe search select the best development checkpoint, breaking ties by earlier epoch then listed candidate order. Keep only the current candidate's best checkpoint and the search winner's checkpoint while searching; delete losing weights as soon as selection permits. Retain compact candidate scores and selection metadata. Do not save auxiliary last checkpoints or optimizer state for completed candidates.

The recipes in the table tune hyperparameters with seed 0 only and rerun the selected settings with seeds 1 and 2. Seeds measure probe variability, not independent pretraining variability. No seed or checkpoint is chosen on the final evaluation split. Optimizer batch sizes belong to the protocol; `Execution.batch_size` controls feature extraction only. Memory-saving accumulation must preserve sample order, effective probe batch, update count, and loss normalization. For segmentation, divide summed microbatch pixel losses by the valid-pixel count of the entire effective batch; averaging microbatch means is not equivalent. Pin shuffling, final partial-batch handling, and scoring cadence in the task manifest.

Every learned probe is initialized afresh for each model checkpoint using its prescribed seed, sample bank, and training budget. No classifier weights or optimizer state carry over from an earlier training epoch. A classifier updated alongside a changing encoder is an online probe with a different interpretation and is outside this package's frozen-representation protocol. Fitted-probe identity includes the extraction identity, training/development sample manifests, task digest, numerical profile, and probe seed; it excludes the reporting partition, requested benchmark list, and output directory. A verified completed selection includes the selected heads, training-only normalization statistics, and candidate/epoch selection metadata. Final evaluation reuses that selection from the active workspace without fitting or searching again. If it has been pruned, rebuild it using the same training/development protocol before reading final data, and record that regeneration occurred. Retention policy must never change the selection protocol.

`imagenet-100` and `pascal` use fixed optimizer settings and seed 0, scoring the final probe update. Freeze sample counts, resolution, optimizer settings, and update counts in their task manifests. Calibrate sample counts and probe budgets using early, middle, late, and nearby CNN, ViT, and glimpse-model checkpoints. Predeclare the checkpoint pairs and reserve checkpoints for acceptance after choosing the recipe. Measure setup, fresh feature extraction, probe fitting, and scoring separately, and report total latency with a new model-state digest and no reusable feature cache, plus contention when sharing a training GPU.

Start calibration with the following proposed acceptance targets. These are design targets, not measured performance claims. Pin reference GPU, CPU, memory, storage, numerical profile, adapter recipe, and evaluation cadence in the calibration manifest; a supported claim applies to that setup.

| Criterion | Measurement and proposed gate |
| --- | --- |
| Feedback latency | The `imagenet-100` plus `pascal` list for one declared adapter recipe finishes within 120 seconds in each of at least three fresh-checkpoint trials, including model loading, hashing, extraction, probes, scoring, and cleanup. Measure end to end in the calling harness or CLI so loading before `bench.run` is counted. Report time to the first task result too. |
| Training overhead | At the advertised cadence, evaluation adds at most 5% wall time to a fixed training-step interval, measured with and without evaluation under the same allocation. Report separate evaluation GPU time when using another GPU. Results must arrive before the next scheduled evaluation. |
| Meaningful change | Start with delta = 2 percentage points top-1 for ImageNet-100 and 1 percentage point mIoU for VOC. These specify the changes the monitoring recipe aims to resolve. |
| Sampling precision | On the predeclared checkpoint pairs, the paired 95% confidence-interval half-width is at most delta / 2. Resample source images, preserving class stratification for ImageNet and recomputing dataset mIoU for VOC; never treat pixels as independent samples. |
| Agreement with fuller probes | Detect the same direction with a paired interval excluding zero for at least 90% of predeclared pairs whose fuller-reference difference is at least delta and whose reference interval excludes zero. Predeclare a minimum number of qualifying pairs spanning all three model families. Report every pair and its intervals, including reversals; an insufficient set of qualifying pairs cannot establish sensitivity. |
| Probe budget adequacy | Doubling probe updates changes scores by at most delta / 4 for at least 90% of acceptance checkpoints and introduces no statistically supported reversal of a qualifying pair. |

The fuller reference uses the same dataset, label space, preprocessing, and readout, with a larger probe-training bank and a development-validated training budget. Give calibration recipes their own identities. Correlation with ImageNet-1K or ADE20K is useful supporting evidence, but a different task is not the ground truth for an ImageNet-100 or VOC checkpoint difference. Paired intervals quantify evaluation-sample uncertainty; they do not establish variability across independent pretraining runs.

Publish measured latency, overhead, precision, and pairwise agreement before recommending a cadence. If the targets conflict, record the failure and explicitly revise the proposed sample bank, probe budget, cadence, or detectable-change target, then repeat acceptance on reserved checkpoints. Once released, recipe changes require a new identifier; never silently shrink a published recipe to meet a deadline. Dataset acquisition and preparation are measured separately as one-time setup and must already be complete for monitoring timing.

For ImageNet, ADE20K, VOC, and CLEVR, create a fixed development partition from the official training set and reserve the official validation set for final reporting. Pin sample banks for each benchmark's training, development, and final partitions. Use stratification for ImageNet, deterministic source-image grouping for the other datasets, and persist the exact manifests. Do not refit on development data in v1. SPair uses its official train/validation/test partitions, with validation for monitoring and test for final reporting. These split choices differ from some published recipes and must be visible in results. Repeated development evaluation may guide training decisions; final evaluation must not select the training checkpoint or change the recipe.

The adapter recipe must record declared pretraining datasets/splits and data used for training-checkpoint or architecture selection, with explicit unknown entries when provenance is unavailable. Results distinguish pretraining image exposure from use of evaluation labels or scores for selection. `evaluation="final"` identifies a benchmark reporting partition; it cannot make previously used validation images untouched. If that partition influenced earlier model selection, report the exposure and do not describe its score as an untouched held-out result. New training integrations adopt the benchmark development split before selecting checkpoints; existing checkpoints retain their actual selection history.

Sample banks are ordered source-image IDs and targets, not copied image collections. Keep the preparation revision, seed, source checksums, and manifest digest so the same membership, labels, and transforms can be regenerated from shared source data. Rebuilding an evaluation set must reproduce its original manifest, not draw a fresh split. Do not duplicate source images, annotations, or prepared canvases per model or training checkpoint.

Before freezing recipes, calibrate on conventional CNN, ViT, and glimpse-based representations with a pretrained positive control and a weak control such as an untrained encoder. Check sensitivity to feature scaling, doubling the probe training budget, and optima at search-grid boundaries where a search is used. Resolve undertraining or inadequate search coverage on development data and record the evidence before assigning v1 identifiers.

For each task, the release manifest must pin: dataset edition and checksums, ordered sample IDs, split assignment, label mapping, transforms and interpolation, feature normalization, metric aggregation, probe settings, seed set, numerical profile, and reference implementation revision. The validation milestone must fill and review every field before a task is advertised as released.

## Dataset acquisition and preparation

Provide one explicit setup command using the same benchmark names and `data_root` as evaluation:

```nu
(uv run bench prepare 'names=[imagenet-100,pascal]' 'data_root=~/datasets')
(uv run bench prepare 'names=[imagenet-1k,ade20k,spair71k,clevr-count,clevr-distance]' data_root=/data/datasets)
```

`data_root` defaults to `~/datasets`. Expand `~` in both CLI paths and Python `run` paths, resolve relative paths against the caller's working directory, create the destination if necessary, and display the resolved root. Preparation requires no model, checkpoint, CUDA initialization, or experiment tracker. It uses fixed task recipes rather than user-selected split sizes or target overrides.

The command resolves and deduplicates source requirements, reuses compatible local archives or extracted data, downloads available missing sources, validates source checksums and required annotations, prepares the canonical dataset layout, and writes deterministic training/development/final sample manifests and targets. ImageNet-100 and ImageNet-1K share one ImageNet source tree; both CLEVR tasks share one set of images and scene annotations. Existing compatible layouts are indexed in place rather than moved or duplicated. Per-dataset path configuration applies consistently to preparation and evaluation.

Each acquisition recipe pins its dataset edition, official source page, download URLs or supported authenticated/manual acquisition route, archive checksums, expected layout and counts, and preparation revision. Use [ImageNet's download page](https://www.image-net.org/download.php), [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/), the [ADE20K scene parsing benchmark](https://sceneparsing.csail.mit.edu/), [SPair-71k](https://cvlab.postech.ac.kr/research/SPair-71k/), and [CLEVR](https://cs.stanford.edu/people/jcjohns/clevr/) as source references. Pin the scene parsing challenge edition for ADE20K and the main CLEVR images plus scene annotations required by the two target definitions. Verify actual download routes and digests during dataset acceptance; a project-page link alone is not a completed acquisition recipe.

If a source requires access credentials or a manually obtained archive, report the official acquisition route and exact expected local path, then continue preparing independent available datasets. Rerunning the same command resumes after the required input is supplied. Do not mark a task ready when any source or manifest is incomplete. The final report lists ready and incomplete tasks, source and manifest paths, counts and digests, download/preparation time, and disk use; return a nonzero exit status if any requested task is not ready.

Report expected download and peak preparation space before acquisition. Keep partial downloads in bench-owned staging and resume them where the source supports it; verify the complete archive before use. Extract into staging without allowing archive members to escape it, and commit verified data and manifests atomically under dataset preparation locks. A repeated successful command must reuse verified state without downloading or extracting again; changed or missing source files invalidate the affected verification. Remove bench-owned intermediate archives after successful extraction when the canonical source data suffices for regeneration, while preserving user-provided archives and unrelated files. Retain one shared source copy and compact preparation manifests, not per-model datasets or feature banks.

## Task visualizations

Every published benchmark must provide a task explanation and annotated real examples through the same CLI:

```nu
(uv run bench visualize 'names=[imagenet-100,pascal,spair71k,clevr-count,clevr-distance]' 'data_root=~/datasets')
(uv run bench visualize 'names=[imagenet-1k,ade20k]' 'data_root=~/datasets' output_dir=outputs/bench/task-examples)
```

`visualize` shares name resolution, `data_root`, and per-dataset paths with `prepare` and evaluation. It requires prepared source data and development manifests, and gives the matching preparation command when they are missing. It needs no model, checkpoint, probe fitting, GPU, or network access. In v1, examples come only from the development partition so inspecting the tasks does not expose the reserved final examples.

Write a standalone local HTML gallery at `output_dir/index.html`, annotated PNG figures, and a compact manifest recording task identities, preparation digests, sample IDs, and visualization revision. Default `output_dir` is `outputs/bench/visualizations`. Print the report path on completion. Use local relative image links and no remote assets or server. Keep an unannotated view of the exact supplied model canvas beside each annotated view; labels and scoring annotations are explanation-only overlays and never become benchmark model inputs. Reports are intentional user-facing artifacts and survive feature-workspace pruning.

Select up to six examples per task deterministically from its development manifest, spanning available labels or difficulty categories. Keep the example selection independent of companion tasks and model checkpoints. Each task explanation states what the model receives, the global or spatial readout required, how any probe is trained, the target definition, and how the primary metric is computed. Captions identify the sample and split, describe resizing/cropping/padding, and distinguish labels used to train a probe from annotations used only for scoring. Show actual annotations, not fabricated predictions or example benchmark scores.

| Benchmark | Annotated examples and explanation |
| --- | --- |
| `imagenet-100`, `imagenet-1k` | Show the original image with the supplied crop marked, the exact classification canvas, and the ground-truth class name. Explain frozen global features, the linear probe, top-1 accuracy, and the recipe's class set. |
| `pascal`, `ade20k` | Show the supplied canvas beside a semantic-mask overlay with a class legend. Mark padding and ignore regions distinctly and explain their exclusion from scoring. Describe the linear pixel classifier and dataset-level mIoU. |
| `spair71k` | Show source/target images side by side with eligible corresponding keypoints linked. Mark the target scoring box and a PCK acceptance radius around an example target keypoint, transformed from the original-image threshold. Explain cosine matching, PCK@0.1, and why the scoring annotations are unavailable to the model. |
| `clevr-count` | Mark the projected object centers from scene annotations and label the object count and classifier target. Explain that the target counts scene objects, including ones difficult to see, and that the supplied canvas preserves the full scene. |
| `clevr-distance` | Mark the object with the minimum annotated camera distance and display that value, the selected bin, and its boundaries. Explain that the target concerns the closest object's distance, not the separation of two objects. |

Use the same preparation, target-generation, coordinate transforms, ignore rules, and eligibility definitions as evaluation. Render overlays with Pillow and a small static HTML template. The visualization is a dataset/protocol inspection tool; model predictions and performance plots remain evaluation artifacts outside this command's v1 scope.

## Execution and performance

`Execution` separates runtime choices from the task recipe: device, extraction batch size, data-loader workers, temporary workspace, and selection of a named numerical profile. The task owns input resolution, probe seeds and budget, and allowed numerical profiles and tolerances; internal extraction randomness belongs to the adapter recipe. `bf16-v1` uses BF16 autocast on supported CUDA devices, FP32 reductions and probe optimization, and preserves output dtype in temporary feature banks. Unsupported hardware fails with a clear remedy; the caller can choose `fp32-v1`. Numerical profile is part of result and extraction identity. Comparisons across profiles require the registered parity check; they are not silently merged.

The GPU path uses eager execution, pinned host memory, bounded prefetch, and inference mode scoped to adapter execution. Measure setup, warmup, and total evaluation time as well as steady-state throughput. Compilation and its configuration/parity machinery are deferred beyond v1.

Within active work, store the final representation per sample, output, and complete extraction identity; retain usable cold-preflight outputs. Reuse features across hyperparameter candidates, seeds, and compatible requested tasks, including both CLEVR labels. For SPair, extract each distinct image once rather than once per pair. Internal observations or intermediate features are the adapter's responsibility. Changing model weights invalidates feature reuse. Delete each feature bank, its per-example costs, and intermediate predictions once its last consumer in the active run has committed results or a reusable probe selection. Do not retain banks for hypothetical later requests.

Store temporary banks as immutable, bounded tensor shards with an indexed manifest; use memory-mapped reads where supported and allow a bounded working set in memory. Preserve the produced feature dtype, spatial geometry and extent, and per-example extraction costs while needed. The implementation must operate without fitting all features in RAM or requiring a database service. Extraction identity includes model-state digest, adapter source revision/readouts/recipe, dataset/sample identity, input preprocessing, requested method/output, numerical profile, and relevant software versions. Global and spatial results never share a bank merely because their feature widths match. Display names alone are never cache keys. Labels and probe hyperparameters do not belong in the feature key, but do belong in task/result identities.

Acquire exclusive locks for the output directory and workspace before checking identities or mutating state. Concurrent requests targeting either location fail with an actionable busy message; a shared workspace has one active owner. Check the output identity under its lock. Write each shard atomically, then atomically commit its manifest entry. Publish probe-selection records, run manifests, and results atomically only after their referenced artifacts are complete. An interrupted extraction resumes only completed, verified shards. Incomplete probe candidates restart from the beginning in v1; completed candidate scores and surviving selected weights are reused. Repeated requests for a completed run with the same identity return its durable results without rebuilding deleted working state, after finishing any pending cleanup owned by that run. A conflicting identity in the same output directory fails rather than overwriting it.

### State lifetime and regeneration

Durable state consists of shared source data and preparation manifests, requested task-visualization reports, resolved recipes, scores, compact per-image statistics, and provenance. Prepared data may persist only when it is independent of model weights, adapter, readout, and model-specific sampling, and is reusable by every model evaluated under that input protocol. Key it by source and preparation digests, store one shared copy, and require a demonstrated preparation-cost benefit before materializing it. Model feature banks, probe weights, and feature-normalization statistics do not qualify. Prefer regenerating transformed images from source data.

An explicit training workspace retains only the selected probes and normalization statistics for the most recently evaluated model identity after a successful development run. Here model identity includes weights, adapter revision/recipe/readouts, and numerical profile, independent of task and output type. Before evaluating a different model identity, prune the workspace's previous model-specific state while holding its lock. Compact scores and provenance remain in their output directories. This bounds retained probe state to one model per training workspace; choosing an earlier checkpoint for final evaluation may require regeneration. Without an explicit shared workspace, remove all working state on successful return.

Final evaluation removes the workspace's remaining model-specific state after committing its results. If training ends without final evaluation, or interrupted work is abandoned, the caller must close the workspace with `(uv run bench prune work_dir=/path/to/training/.bench-work)`. This narrow CLI action acquires the same workspace lock and removes only bench-owned temporary state; it preserves source/prepared data, model checkpoints, and result directories. Pruning is idempotent and refuses an active workspace. Failures retain verified unfinished work only until retry or explicit pruning; cleanup failures report the remaining paths for retry. There is no automatic eviction service, permanent model cache, or keep-all-state option in v1.

Regeneration requires the caller's original model checkpoint, source datasets, recorded software environment, and resolved task/extraction recipes. `bench` stores their identities and locations, not duplicate checkpoints or datasets. Verify regenerated sample manifests exactly and regenerated scores within the declared numerical tolerance. Completed results remain readable after all model-specific working state has been deleted.

Preflight estimates peak temporary storage from sample count, feature shape, dtype, simultaneously live banks, probes, and atomic-write overhead. Dense banks can be tens of gigabytes even for small backbones. Prune obsolete owned state before allocating new banks, check available space, stream shards, and report the estimate before extraction. Never silently shrink a dataset, lower its resolution, change precision, or delete unrelated state to fit a budget.

Measure dense probe execution early, including feature reads, full-resolution logit resampling, loss/backward, fitting, development scoring, and cleanup. The proposed ADE20K search trains five probes for 40 epochs each. An illustrative 20,000-image bank of 768×32×32 BF16 features occupies 31.5 GB and entails about 6.3 TB of logical feature reads across those 200 training passes; physical disk traffic depends on memory caching. A 16×150×512×512 FP32 logit tensor alone occupies 2.34 GiB before gradients and temporaries. Independent heads may share feature reads when measurements justify it, while retaining separate initialization, optimizer state, batches, update counts, and checkpoint selection. Choose optimizations from these measurements rather than requiring a parallel-head implementation in v1.

Optimization acceptance is measured against the same unoptimized protocol: equal sample counts and predictions within tolerance, plus cold/reused-state wall time, images per second, peak GPU memory, peak host memory, peak temporary bytes, and bytes retained after completion. Reading a valid existing feature bank or completed result must invoke the encoder zero times; a new run after pruning must regenerate missing features. Label these cases separately. No arbitrary speedup is promised before measurement. Start with one process and one GPU; external pueue/SLURM jobs can evaluate checkpoints independently using separate workspaces.

## Results and training integration

`RunResult` is an immutable result with `schema_version`, `status`, `metrics`, and `output_dir`. `metrics` maps task identifiers to named numeric metrics. The JSON artifact is the complete source of provenance, not the in-memory convenience mapping.

Every run retains:

```text
output_dir/
  resolved.yaml
  manifest.json
  results.json
  statistics/         # compact per-image metric statistics
```

Feature banks, probe weights, normalization statistics, and intermediate predictions live only in the temporary workspace. They are not permanent result artifacts.

`results.json` records requested benchmark names, resolved task identifiers, per-task status and evaluation partition, metric name/value/unit/direction, seed results, selected hyperparameters and probe epochs/digests, sample and eligible-pixel counts, model metadata and state digest, requested extraction method and its readout, feature width/grid, data/task/extraction/probe digests, preparation and software/hardware/numerics provenance, declared pretraining and model-selection exposure, extraction-cost summaries, timing, memory, peak temporary and retained bytes, and durable output paths. Record whether probes were fitted, reused, or regenerated; preserve selection metadata after weights are pruned. Separate reused extraction costs from work performed in the current run. Do not advertise deleted temporary paths as available artifacts. Classification accuracy, PCK, and mIoU use percentages; record the scale explicitly.

Retain compact raw-feature health diagnostics from a fixed, bounded subset of each task's probe-training bank before benchmark standardization: RMS variation across images, mean pairwise cosine, and effective rank. For spatial outputs, measure across-image diagnostics separately at each grid position and report within-image token similarity separately. Pin the sample IDs, estimator definitions, sampling limits, and rank convention in the task manifest; report sample count and feature width. Compute diagnostics during fresh bank processing and retain their summaries for reuse without another extraction pass. These describe feature variation, not semantic quality. Record probe training and development loss alongside task scores to expose possible probe underfitting.

Keep sufficient compact statistics for paired analysis: per-image labels and predicted classes for classification, per-image per-class intersection and union counts for segmentation, and per-pair correct/eligible counts with source/target IDs and breakdown labels for SPair. Preserve seed identity and the corresponding grid-ceiling counts. Discard full logits, dense prediction masks, and matching tensors after accumulating these statistics. For segmentation, compute mIoU from the dataset confusion matrix, not a mean of per-image IoUs; summing the retained intersections and unions must reproduce it.

Use `running`, `complete`, and `failed` statuses, with task-specific failure details. Atomically publish each completed task's metrics and committed statistics in `results.json` before starting the next task, keeping the run status `running` until all requested tasks finish. Callers may read these partial results for timely feedback; the Python runner returns its `RunResult` after completion. Completed task artifacts survive a later task failure. A partial run is never summarized as completing all requested benchmarks. For recipes with multiple probe seeds, report their mean, individual scores, and standard deviation. Single-seed recipes report that seed without implying a multi-seed uncertainty estimate. Report tasks individually; no weighted cross-task average.

Paired analysis uses the same source images and retained per-image statistics to recompute the metric. The monitoring acceptance example reports score changes and paired uncertainty against the previous evaluated checkpoint. Final glimpse comparisons include paired baseline differences and results across a declared observation-seed set, with each seed represented by a separate adapter recipe and fresh probe fit. Keep observation-seed variation distinct from probe-seed variation. Repeated measurements of one image remain grouped for uncertainty estimation; averaging their probabilities would define a separate ensemble with the combined resource budget. SPair reports category scores and counts without naive pair-independent confidence intervals because source and target images recur across pairs. Statistical comparison belongs in result analysis and does not require model-specific experiment protocols in the runner.

The caller owns experiment tracking. Python callers receive numeric metrics in `RunResult.metrics`; callers launching the CLI read the same results from `results.json`. The training script can upload metrics and selected local artifacts to MLflow or another tracker using its existing run and training epoch. Include evaluation partition, resolved task identifier, and metric in logged names so monitoring and final results stay distinct. Local artifacts remain authoritative regardless of upload success. `bench` has no tracker dependency, configuration, run lifecycle, or upload logic.

Training integration evaluates immutable saved checkpoints in a separate process. The caller chooses benchmark lists and cadence, for example `['imagenet-100', 'pascal']` each epoch and `['imagenet-1k', 'ade20k', 'spair71k', 'clevr-count', 'clevr-distance']` every 25 epochs and at completion. Use development results during training, then evaluate the chosen benchmarks on their final partitions once the training checkpoint is selected. Coalesce duplicate scheduled evaluations of the same checkpoint. Evaluations within one training job use one explicit temporary workspace so obsolete checkpoint state is pruned as training progresses. Final evaluation closes that workspace; the caller invokes pruning when ending or abandoning training without final evaluation. Record training epoch/checkpoint and selection rule in run metadata. The benchmark does not choose training checkpoints, group tasks into speed tiers, or own a scheduler.

A training callback may launch the CLI and log its completed results. GPU allocation and whether training waits or continues belong to the caller; do not promise zero training overhead or hide an accumulating evaluation queue. Measure monitoring cost on representative hardware, including contention if training and evaluation share a GPU. Importing `bench` must not start workers, initialize CUDA, download data, or contact external services.

## CLI and package structure

Use `uv`, Hydra, and packaged hierarchical YAML configs. Preserve a single `bench` executable, moving its implementation into `src/bench/cli/` when implemented, with explicit `prepare`, `visualize`, and `prune` actions for dataset setup, task examples, and closing an idle workspace. Dispatch these actions before constructing any model or evaluation configuration that requires a checkpoint. Python callers do not have to use Hydra. Keep CLI and Python evaluation on the same runner.

Proposed Nushell invocation:

```nu
(uv run bench prepare 'names=[imagenet-100,pascal]' 'data_root=~/datasets')
(uv run bench visualize 'names=[imagenet-100,pascal]' 'data_root=~/datasets')
(uv run bench model=dinov3 'names=[imagenet-100,pascal]' 'data_root=~/datasets' evaluation=development model.checkpoint=/path/to/epoch.pt execution.work_dir=/path/to/training/.bench-work)
(uv run bench model=dinov3 'names=[imagenet-1k,ade20k,spair71k,clevr-count,clevr-distance]' evaluation=development model.checkpoint=/path/to/epoch.pt execution.work_dir=/path/to/training/.bench-work)
(uv run bench model=dinov3 'names=[imagenet-1k,ade20k,spair71k,clevr-count,clevr-distance]' evaluation=final model.checkpoint=/path/to/selected.pt execution.work_dir=/path/to/training/.bench-work)
(uv run bench prune work_dir=/path/to/training/.bench-work)
```

The example `model=dinov3` selects a shipped reference adapter. Model configs use Hydra `_target_` factories supplied by the model package or an explicit benchmark adapter, with no source-tree edits to the runner. `data_root` defaults to `~/datasets` for both preparation and CLI evaluation; per-dataset paths can be configured without modifying loaders. Keep Hydra's working-directory changes disabled so relative checkpoint/output paths are predictable. Use distinct output directories for checkpoint, requested benchmark list, and evaluation partition.

Initial implementation layout:

```text
src/bench/
  __init__.py          # documented public exports
  api.py              # runner interface and public data types
  runner.py           # preflight, extraction, probes, artifact workflow
  prepare.py          # source acquisition, validation, preparation workflow
  visualize.py        # task explanations and annotated example reports
  features.py         # extraction and disposable feature banks
  tasks/
    classification.py # ImageNet and CLEVR definitions/probes
    segmentation.py   # ADE20K and VOC definitions/probe
    correspondence.py # SPair matching
  adapters/           # only shipped reference adapters
  cli/
    main.py
    configs/
      config.yaml
      model/
      task/
      execution/
```

Keep task-specific source definitions, loading, target preparation, metrics, and preprocessing together. Preparation and evaluation consume the same definitions and manifest formats. Extract shared functions only when two tasks actually share behavior. No public dataset/probe plugin registry in v1. Model adapters are the supported external extension seam.

Use PyTorch/torchvision, NumPy, Pillow, and Hydra/OmegaConf as the initial implementation stack. Experiment trackers remain dependencies of the calling application. Add optional adapter dependencies only when used. Keep research training packages out of the base dependency graph. The scaffold currently requests Python 3.13; validate the chosen dependencies against it before changing that requirement. A reproducible `uv.lock` and a tested environment accompany the first release.

## Reuse and reference implementations

- [DINOv3 evaluation](https://github.com/facebookresearch/dinov3#evaluation): its [cached-feature logistic regression](https://github.com/facebookresearch/dinov3/blob/main/dinov3/eval/log_regression.py) is a direct reference for global probing; augmented linear classification and ADE20K segmentation are distinct recipes. Adapt needed algorithms behind the model seam without importing the training framework.
- [Probe3D](https://github.com/mbanani/probe3d): reference SPair/NAVI matching and geometry conventions. Pin the implementation revision and align preprocessing, interpolation, normalization order, and coordinate conventions for algorithm parity checks. Test the task's pooled metric separately from Probe3D's category average.
- [VISSL](https://github.com/facebookresearch/vissl): reference CLEVR targets and classic SSL protocols. Its archived status makes it a reference rather than a required runtime dependency.
- [MTEB model interface](https://docs.mteb.org/get_started/usage/defining_the_model/): useful for a later optional global-embedding export. MTEB's existing few-shot task definitions must keep their own identities; they are not aliases for full-data probes.
- [V-JEPA evaluation](https://github.com/facebookresearch/vjepa2#probe-based-evaluation): reference for a future Something-Something attentive probe. Define temporal inputs when video is implemented.

Check upstream licenses before copying code and retain required attribution. Record which details are reused and which differ. A source citation is not a claim of score parity.

## Acceptance and implementation order

1. **Prepare and inspect shared data.** Implement `bench prepare` with explicit/default data roots, local-data reuse, source acquisition, validation, and deterministic manifests, plus `bench visualize` task reports. Use small source fixtures to verify tilde/relative-path handling, shared-source deduplication, interrupted download/preparation recovery, checksum failures, archive extraction boundaries, concurrent preparation, idempotent reruns, actionable missing-access instructions, and incomplete-task reporting without a model or CUDA. Verify deterministic visualization selection, annotation alignment, class/bin labels, scoring overlays, and exclusion of final examples. Inspect a rendered report for each real benchmark as its task is implemented, alongside separate acceptance jobs for acquisition routes, layouts, labels, and digests.
2. **Establish the seam.** Implement `run`, result serialization, and CNN/ViT adapters, then exercise a real glimpse adapter. On fixture data, verify task dispatch to `encode_global` or `encode_spatial`, different feature widths, one-capability adapters, independent calls, shapes, coordinates/extents, normalization ownership, annotation isolation, and preservation of caller model/RNG state on success and exceptions. Verify ordinary tensors reach probe training, deterministic per-example extraction across batching/resumption, and invalidation when either method's extraction identity changes.
3. **Ship global probes and temporary banks.** Implement ImageNet-100 and CLEVR targets using prepared manifests. Verify fresh/reused parity, exact regenerated split membership, deterministic resumption, and zero extraction calls on valid bank/result reuse. Validate selected CLEVR labels against the pinned reference. Test concurrent output/workspace writers and interruption at artifact commits. Verify losing probe weights and finished banks are pruned, final evaluation reuses an available development selection across output directories/task lists, and pruned selections regenerate using training/development data only.
4. **Add spatial tasks.** Implement VOC first, then ADE20K and SPair. Verify affine alignment, declared full extent with inset token centers, border behavior, ignore labels, unequal-valid-pixel microbatch equivalence, mIoU reconstruction from compact statistics, known keypoint matches, the grid-limited PCK ceiling, and one extraction per distinct SPair image. Check matching against the pinned reference under aligned settings and verify the task's aggregation independently. Measure dense bank, full-resolution loss/backward, and probe costs before adding optimizations.
5. **Validate feedback, comparisons, and pruning.** Produce the monitoring calibration report against the declared latency, overhead, precision, agreement, and probe-budget gates using fresh checkpoint identities. Run the glimpse condition matrix and verify matched observations, access restrictions, per-method readouts, extraction costs, and paired analysis, regardless of whether composition improves scores. Exercise caller-selected lists and schedules, verify task recipes/results are independent of companion tasks, expose completed metrics before a later task finishes, and confirm monitoring never reads final partitions. Verify raw-feature diagnostics survive bank deletion, new checkpoints receive fresh probes and replace obsolete workspace state, and declared prior selection exposure appears in final reports. After standalone success, final evaluation, or explicit pruning, verify no model-specific temporary state remains and results are still readable. Test idempotent pruning, refusal of active workspaces, preservation of source/prepared data and model checkpoints, and regeneration after deleting all derived banks and probe weights.
6. **Calibrate and freeze v1.** Complete probe calibration across CNN/ViT/glimpse representations, including weak controls, scale sensitivity, training-budget checks, and search coverage. Record task and acquisition manifests and baseline artifacts, measure numerical-profile parity, total cost including cleanup, and peak/retained storage, then publish stable identifiers and the interface. No unresolved acquisition route, split, sample-bank, monitoring acceptance target, head, or metric defaults remain at release.

Fast CPU tests should finish in under 30 seconds and exercise observable behavior, not internal helper structure. GPU integration tests verify BF16/FP32 behavior, readout geometry, accumulation equivalence, and unchanged model-state hashes. Full dataset evaluations are explicit acceptance jobs rather than unit tests.

Success means a researcher can prepare selected datasets under a chosen root with one CLI command, inspect each task through annotated examples, implement one adapter exposing global vectors and/or spatial grids, and call `bench.run` with any supported benchmark list; obtain timely, calibrated feedback during training and final results with explicit selection provenance; isolate the measured contribution of additional glimpses and learned composition through matched comparisons; recover interrupted work without mixing identities; prune all model-specific working state at completion; and explain or regenerate every score from compact provenance, the caller's model checkpoint, shared source data, and the recorded preparation/extraction/task recipes.
