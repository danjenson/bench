# Continue benchmark acceptance on c3

The user chose to finish the current pretrained ResNet18 ADE evaluation on Ghost,
then move all remaining evaluations to c3. The package is implemented as 0.1.0;
the empirical gates in `SPEC.md` are still incomplete or failing. This transfer
does not establish a calibrated v1 release.

## Transfer and Ghost boundary

Ghost's `bench-acceptance-recovery` pueue group is paused. Task 3 remains running
until its first model, `resnet18-pretrained`, completes evaluation and its pinned
scientific verifier. Task 15, in `bench-c3-handoff`, then stops task 3. The watcher
checks the original command, creation time, group and model before stopping it.
Its five decision tests pass. The later Ghost tasks must remain held.

The watcher polls every 0.5 seconds; the following model could begin startup
before the stop. Treat any such output as interrupted and use a fresh c3 output.
The whole Ghost task will report killed, while the first model can be verified
complete. Read the model's status and boundary receipt separately.

Transfer evidence lives in `outputs/acceptance/c3-handoff-20261023/`:

- `guardian-launch.json`: persistent watcher identity.
- `ghost-boundary.json`: written only when the watcher reaches a terminal state.
  Require `action=stop-after-verified-model`, a completed first model and a
  terminal task status before declaring the requested Ghost boundary reached.
- `ghost-ade-boundary-state.json`: original queue state at the verified boundary.
- `transfer-launch.json`: persistent artifact-transfer task identity.
- `transfer-receipt.json`: written only after the final sync and checksum check.
  Require `status=complete` before treating the remote artifact set as complete.
- `manifest.json`: the 11 original plans, their SHA256 hashes, exact temporary
  source paths and file checksums. `queued-source-trees.tar.gz` preserves 823
  source/configuration files, independently checked against those hashes.
- `ghost-{bench,glimpse}-freeze.txt` and `ghost-glimpse-{pyproject.toml,uv.lock}`:
  dependency inventories. They are evidence, not portable requirements files.

Ghost pueue task 16 runs the transfer in group `bench-c3-transfer`. It first
copies static artifacts, excluding the active ADE family
directory, then waits for task 15 and synchronizes the completed result. It copies
`outputs/acceptance/`, including checkpoints, prepared banks, source snapshots,
plans, verifiers and prior results. Disposable `.work` and `__pycache__` directories
are excluded. No source files are deleted. The final checksum dry run must report
no source-file differences. This transfer is approximately 42 GiB.

Code is separate because the user will commit it. There was no Git remote
configured on Ghost when this handoff was prepared. After committing, run on
Ghost to copy the checkout, including Git history, into the staged destination:

```nu
(rsync -a --exclude=outputs/ --exclude=output/ --exclude=.venv/ --exclude=__pycache__/ --exclude=.pytest_cache/ --exclude=.playwright-cli/ --exclude=build/ --exclude=dist/ /data/repos/bench/ c3:/home/scratch/danson/repos/bench/)
```

This command leaves the separately transferred acceptance artifacts in place.
Do not make concurrent c3 code edits while synchronizing the checkout.

To inspect persistent Ghost work:

```nu
(pueue status)
(pueue log 15 --lines 20)
(pueue log 16 --lines 20)
(cat outputs/acceptance/c3-handoff-20261023/transfer-launch.json)
```

## c3 setup and relocation

Destination: `/home/scratch/danson/repos/bench`. c3 has an RTX 4090 with UUID
`GPU-23c48372-0992-f3a5-9759-f9469423137b`. At inspection it had no active pueue
tasks and about 230 GiB free on scratch. Recheck contention and capacity at launch.
Ghost used UUID `GPU-31141b5d-1faf-4631-9a5f-7dd7b0e3bc21`; timing on c3 is a
new hardware/host measurement and must be recorded as such.

The old commands cannot run unchanged: `/data` does not exist on c3, plans pin
absolute paths and Python binaries, and pueue task IDs are local to a host.
Keep transferred historical plans/results immutable. Create separate c3 execution
plans with an explicit link to each original plan hash and a relocation report.
Do not merely overwrite hashes until checks pass.

1. Verify the transfer receipt and source-archive digest. Extract the archive's
   safe relative members under `/tmp` to recover the 11 original source-tree
   names. Verify every extracted file against `manifest.json`. Preserve the
   archive: successful queue workers remove their temporary source trees.
2. Recreate dedicated environments under the benchmark checkout. The main
   environment uses Python 3.13.13 and the committed `uv.lock` with the `dinov3`
   extra. Glimpse evaluations need a separate environment; do not modify c3's
   existing `glimpse-ssl/.venv`. The inspected c3 training environment lacks
   Transformers and does not match all pinned dependencies.
3. Rebase prepared source roots and execution paths using the mapping below.
   Use the verified benchmark-owned source directories described below. Preserve
   the original prepared manifests. Their canonical identity excludes
   `source_root`, so changing that field preserves the bank/preparation digests;
   the manifest's file SHA256 changes and must be recorded in the new plan.
   Preserve split rows, sample order, labels and recipes exactly. Revalidate the
   relocated files through the package before evaluation.
4. Derive new plans, caller YAML/JSON and expected model identities on CPU.
   Preserve checkpoint bytes/state hashes, package snapshots, adapters/readouts,
   recipes, seeds, banks, numerical profiles, batches, workers and tolerances.
   Only declared location/environment/hardware changes may explain identity
   differences. Keep old evidence verifiable through an explicit path mapping;
   do not rewrite historical result identities to look like new evaluations.
5. Replace Ghost process/pueue dependencies with a serial c3 pueue chain and
   fresh output/state paths. Preserve scientific predecessor verification as
   well as serialization. The recovery runner is tied to frozen Ghost queue
   snapshots, so it is not a c3 launcher. The generic
   `run_pueue_calibration_queue.py` and its 12 tests are available as a starting
   point; its first dependency and pinned plan fields still need adaptation.
6. Verify relocated pins, dependency environments, CPU model identities and
   baseline results. Run the relevant queue/verifier tests and CUDA seam checks,
   then launch the remaining jobs via pueue. Record exact commands, IDs, source
   hashes and timing scope. Exclude the completed Ghost model from fresh work.

| Ghost location | c3 location |
| --- | --- |
| `/data/repos/bench` | `/home/scratch/danson/repos/bench` |
| `/data/cache/datasets/imagenet` | `/home/scratch/danson/repos/bench/outputs/c3-source-overlays/imagenet` |
| `/data/cache/datasets/ade20k/ADEChallengeData2016` | `/home/scratch/danson/repos/bench/outputs/c3-source-overlays/ade20k/ADEChallengeData2016` |
| `/data/cache/datasets/pascal-voc-2012/VOCdevkit/VOC2012` | `/home/scratch/danson/repos/bench/outputs/c3-source-overlays/pascal-voc-2012/VOCdevkit/VOC2012` |
| `outputs/acceptance/data/{CLEVR_v1.0,SPair-71k}` | Same relative paths in the transferred checkout |
| `/data/repos/glimpse-ssl/.venv/bin/python` | A new dedicated benchmark glimpse environment |

The c3 source audit verified 40,421 ADE and 2,929 VOC training/development files.
The existing datasets lacked ADE's `objectInfo150.txt` and VOC's split metadata.
The benchmark-owned directories supply the pinned Ghost metadata and hardlink
the existing images/masks. They use physical directories rather than directory
symlinks, because the package rejects source paths escaping the dataset root.
Shared datasets were not modified. Both historical glimpse model-source trees
are under `outputs/acceptance/reference-source/`; use those snapshots.

ImageNet on c3 contains only the 100-class subset. All 126,689 existing training
images matched the full prepared checksum list; 1,154,478 images were missing,
with zero checksum mismatches. Both the ordinary and fuller ImageNet-100
training/development banks are fully present. Benchmark-owned ImageNet staging
hardlinks the existing train/validation files. Full ImageNet development and its
final validation partition are not ready: the source audit did not inspect final
partitions, and c3 has only the subset's validation files.

The audit requests, launch identities, terminal reports, compressed failure log,
missing-file list and size calculations are retained in
`outputs/acceptance/c3-handoff-20261023/dataset-preflight/`. The last full audit
was c3 pueue task 8 and correctly failed for the missing ImageNet files; its
`with-metadata-overlays/result/readiness.json` reports partial readiness. c3 tasks
6 and 7 retain the earlier metadata-missing failures. These were CPU source
audits, not model evaluations. The checker was exercised on matching and altered
source bytes. The c3-only staging layout receipts are under
`outputs/c3-source-overlays/`.

### Full ImageNet transfer order

The missing training images total 132,411,896,262 bytes (132.4 GB). Transfer them
after all c3 full-ADE controls finish, pass verification and prune their workspaces,
before starting the full ImageNet family. The initial available space was
247,158,468,608 bytes. The approximately 44.7 GB acceptance-artifact transfer,
132.4 GB missing images and 63.6 GB largest ADE temporary estimate together leave
only about 6.4 GB, before environment installation and filesystem overhead.
Serializing the missing-image copy after ADE avoids this overlap. Recheck actual
free space and the next job's storage estimate before copying or launching.

After that ADE boundary, run on Ghost through pueue (using a Bash script for the
pipeline if needed):

```nu
(gzip -dc outputs/acceptance/c3-handoff-20261023/dataset-preflight/with-metadata-overlays/missing-imagenet-files.txt.gz | rsync -a --partial --files-from=- /data/cache/datasets/imagenet/ c3:/home/scratch/danson/repos/bench/outputs/c3-source-overlays/imagenet/)
```

Then rerun the full training/development checksum audit with its ImageNet root
changed to that staging directory, preserving the original request/results and
recording a new request hash. Require all 1,281,167 training images to match
before full ImageNet evaluation. This copy has not been started. The original
Ghost transfer task 16 copies `outputs/acceptance/` only; its success does not
mean these additional ImageNet images have been transferred.

Both environments pin torch 2.14.1 and torchvision 0.29.1 (runtime builds on Ghost
have `+cu130`), Hydra 1.3.7, OmegaConf 2.3.1 and safetensors 0.8.0. The main
environment has NumPy 2.5.4, Pillow 13.0.0 and Transformers 5.20.2, with no timm.
The glimpse environment has NumPy 2.5.3, Pillow 12.3.0, timm 1.0.30 and
Transformers 4.57.6. Consult each job's exact `environment_versions`; plans vary
in which packages they require. Recheck Python/binary pins and CUDA runtime too.

## Remaining queue

All plan names below are relative to `outputs/acceptance/`. Original task numbers
describe Ghost only. Preserve this scientific order when creating c3 jobs.

| Ghost task | Original plan | Remaining work |
| --- | --- | --- |
| 3 | `full-ade-family-plan.json` | Five controls after pretrained ResNet18: untrained ResNet18, two ViTs, two larger glimpse controls; 25 probe fits. |
| 4 | `full-imagenet-family-plan.json` | Four full ImageNet ViT/glimpse controls; 32 fits. |
| 5 | `full-spair-family-plan.json` | Six controls; 32,304 retained pair records. |
| 6 | `public-cnn-batch16-plan.json` | Three CNN checkpoint timing trials with extraction batch 16. |
| 7 | `monitor-fuller-selection-plan.json` | One CNN diagnostic, 16 variants and 22,048 records. |
| 8 | `weak-vit-current-parity-plan.json` | FP32/BF16 weak-ViT monitoring checks. |
| 9 | `grid-glimpse-gpu-seam-corrected-plan.json` | Corrected native Grid model contract check. |
| 10 | `grid-glimpse-early-monitoring-corrected-plan.json` | Epochs 81/100/105, FP32 and BF16; six evaluations. |
| 12 | `full-clevr-search-grid-plan.json` | Four ViT/glimpse joint LR/WD searches; 16 variants, 240 fits. |
| 13 | `grid-glimpse-middle-monitoring-plan.json` | Epochs 525/530, FP32 and BF16; four evaluations. |
| 14 | `grid-glimpse-budget-scale-plan.json` | Epochs 81/530, 100/200 updates and feature scales 0.1/10; 16 variants. |

At handoff preparation the Ghost ResNet18 ADE search and seed 1 were complete;
seed 2 was running. Intermediate reports `full-ade-completed-search-check.json`
and `full-ade-seed-1-check.json` match baseline selection metadata and scores to
within 0.00001 mIoU points. The full prediction reconstruction and model verifier
must still finish. The winner is at the LR grid's upper boundary (0.001), so
search coverage remains unresolved. Resumed wall time excludes earlier work and
downtime and cannot establish fresh-run latency.

## Outstanding evidence beyond the queue

- The original Grid checkpoint collector ended at 11:29:47 UTC on 2026-10-23
  with `status=incomplete`. The c3 process was absent, and its last observed
  checkpoint was epoch 629, step 1,244,791. Pueue task 2 exited successfully
  because it reported that terminal state; collection itself did not complete.
  Epoch 500 was missed; 750/755/995/1000 were never reached. The prospective
  middle supplement did capture 525/530. Preserve all receipts. Do not replace
  late/reserved checkpoints with epoch 629 or restart backbone training.
- Reserved acceptance has not run. Its original plan is
  `monitor-reserved-acceptance-plan.json`: CNN 15/16/44/45, ViT 5/6/14/15, Grid
  250/255/750/755. Grid 250/255 are captured; 750/755 are unavailable. Do not use
  reserved benchmarks to tune recipes.
- Existing monitoring controls have unresolved sampling-precision, convergence,
  doubled-budget and CNN latency failures. Weak ViT FP32/BF16 classification
  differed by 0.12 points against a 0.1-point tolerance. Queued diagnostics do
  not guarantee these gates will pass.
- Full-reference agreement, enough qualifying pairs across all three families,
  three fresh timing trials at ≤120 seconds, and measured training overhead
  ≤5% remain required. No matched training interval has been supplied, and no
  training intervention is authorized. Do not infer overhead from log gaps or
  recommend an evaluation cadence yet.
- Keep all original gates/tolerances. A finished queue or passing unit suite is
  insufficient to freeze v1. `docs/acceptance.md` contains the full ledger.

Suggested opening message for the new c3 chat:

> Continue `/home/scratch/danson/repos/bench` using `docs/c3-handoff.md` and
> `SPEC.md`. Finish the remaining acceptance evaluations on c3. Ghost was to
> finish only pretrained ResNet18 ADE; inspect the boundary and transfer receipts
> first. Verify source/data/environment identities, derive relocated execution
> plans, and launch/monitor the remaining serial pueue jobs. Preserve completed
> evidence and original release gates. Do not restart backbone training, silently
> substitute missing checkpoints, or treat v1 as released.
