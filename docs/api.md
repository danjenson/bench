# Python API and result schema

This is the frozen v1 interface for package 1.0.0. [The acceptance audit](acceptance.md) describe verification and measurement scopes.
[SPEC.md](../SPEC.md) defines the complete input, geometry and evaluation contract.

## Public names

The supported Python surface is `bench.run`, `bench.VisionModel`,
`bench.ModelInfo`, `bench.ImageBatch`, `bench.SpatialFeatures`,
`bench.Extraction`, `bench.Execution` and `bench.RunResult`. Dataset loaders,
probe implementations, preparation internals and analysis helpers are internal.
Importing `bench` performs no data access, model loading or CUDA initialization.

```python
def run(
    model: VisionModel,
    *,
    names: list[str],
    evaluation="development",
    data_root: Path,
    output_dir: Path,
    execution: Execution | None = None,
) -> RunResult:
    ...
```

`names` is a nonempty list of published aliases or exact task identifiers.
Unknown names, duplicate resolved tasks and missing adapter capabilities fail
preflight. `evaluation` is `"development"` or `"final"`. The caller supplies
the adapter on the requested device. `run` uses the prepared datasets and never
downloads them. It fits probes on frozen features; it does not train the encoder.

## Adapter and data types

`VisionModel` is a `torch.nn.Module` with a `ModelInfo` in `info` and one or both
of these independently callable methods:

```python
def encode_global(self, batch: ImageBatch) -> Extraction: ...
def encode_spatial(self, batch: ImageBatch) -> Extraction: ...
```

The declared `readouts` determine which methods are supported. The runner calls
only the requested method and discovers its feature width. The adapter owns
normalization and readout construction. It receives pixels, validity, affine
geometry and opaque sample IDs; it receives no annotations. All learned
inference modules must be registered on the adapter. Extraction must not mutate
model state or consume caller-owned random streams. The runner restores module
training flags and its managed RNG state on success and failure.

These public data types are frozen dataclasses. Their field order and types are:

| Type | Fields |
| --- | --- |
| `ModelInfo` | `name: str`, `revision: str`, `readouts: dict[Literal["global", "spatial"], str]`, `recipe: dict[str, object]` |
| `ImageBatch` | `images: Tensor`, `valid_pixels: Tensor`, `image_to_original: Tensor`, `sample_ids: tuple[str, ...]` |
| `SpatialFeatures` | `values: Tensor`, `grid_to_original: Tensor`, `extent_in_original: Tensor` |
| `Extraction` | `features: Tensor \| SpatialFeatures`, `costs: dict[str, Tensor] \| None = None` |
| `Execution` | `device: str = "cuda"`, `batch_size: int = 64`, `workers: int = 8`, `work_dir: Path \| None = None`, `numerics: Literal["bf16-v1", "fp32-v1"] = "fp32-v1"`, `progress: bool \| None = None` |
| `RunResult` | `schema_version: str`, `status: Literal["running", "complete", "failed"]`, `metrics: dict[str, dict[str, float]]`, `output_dir: Path` |

Frozen dataclasses prevent field reassignment; they do not recursively freeze
the supplied tensors, dictionaries or module state. The adapter is responsible
for obeying the extraction contract.

Global features have shape `[B, Dg]`. Spatial values have shape
`[B, Ds, Hf, Wf]` and must include their actual grid-to-original affine transform
and full represented rectangular extent. The runner does not derive spatial
features from a global vector. See the SPEC for coordinate frames, interpolation,
extent checks and per-example costs.

`Execution()` selects CUDA, extraction batch size 64, eight workers, a temporary
workspace under the output directory and `fp32-v1`. Extraction batch size does
not change the probe's effective batch or training budget. CPU execution uses
`Execution(device="cpu")`. BF16 is explicit and requires qualified parity for
reference comparison; see [numerical-profiles.md](numerical-profiles.md).

`progress=None` automatically shows transient tqdm Rich bars on an interactive
stderr terminal and suppresses them in redirected logs. Set `progress=True` to
force display or `progress=False` when the caller owns the display or measures
latency. Bars report remaining extraction images, feature normalization, probe
updates (with a validation label), scoring images and correspondence pairs.
They close on success or failure and do not accumulate completed lines.
Progress is a display setting, excluded from model, probe and result identities;
end-to-end timings include any display overhead. Training cadence and metric
logging remain caller-owned.

## Results and compatibility

`RunResult.metrics` maps resolved task identifiers to named numeric metrics.
Accuracy, top-1, mIoU and PCK use percentages. There is no aggregate task score.
The durable `results.json` and referenced statistics contain the complete
provenance. A completed task commits before the next task starts; partial results
must not be interpreted as completion of the requested list. The Python runner
returns after successful completion or raises an error.

The result schema version is `1.0.0`. Result readers accept unknown additive
fields within schema major version 1 and reject unsupported major versions.
Compact-statistics files also record version `1.0.0`; completed result reads
verify each committed statistics file's checksum. Model/task/preparation/
numerical identities govern comparability and reuse independently of schema
compatibility.

Package, task and result-schema versions have separate meanings. During package
1.x, existing public calls and fields retain their meaning. Removing or renaming
a public field requires a package major version. Changing a released task's
samples, preprocessing, probe training, selection or scoring requires a new task
identifier. Schema changes that break existing readers require a schema major
version. Internal modules have no compatibility promise.

The separately packaged reference and monitoring JSON envelopes have their own
schema versions. Their original measurement provenance and numerical profiles
must be retained when comparing a caller's results. Reference scores do not
require loading a baseline model; reproducing a reference evaluation requires
its original public checkpoint and matched protocol.
