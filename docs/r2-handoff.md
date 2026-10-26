# r2 benchmark acceptance handoff

The remaining workload is staged on **r2 / clpc35.cs.ox.ac.uk**. Evaluations
remain paused. Full SPEC acceptance is unfinished; the scope remains the
official public pretrained ResNet18 and ViT-B/16 references. Private model
studies are excluded. Read [acceptance.md](acceptance.md) for the remaining
scientific and release gates.

## Host and environment

Connect with `ssh r2`. The checkout is
`/home/scratch/danson/repos/bench`, and the new execution root is
`outputs/r2-acceptance-20261026/`. Fresh results, states, logs, source snapshots
and plans live there. The source c3 checkout and completed results are preserved.

GPU: NVIDIA RTX 5000 Ada Generation, 32 GB, UUID
`GPU-e82ac849-9237-5670-9b20-5354929b6d05`. The copied verified Python is
`outputs/environments/bench/bin/python`: Python 3.13.13, Torch 2.14.1+cu130,
Torchvision 0.29.1+cu130. Its binary SHA256 remains
`566ab51fd7c7226015d06a6179ca840e6513b65bd5404e1472d2379a0435d80a`.
Package versions and the binary match the original environment receipt. All four
queued model/readout CPU contracts pass. FP32/BF16 CUDA matrix smoke checks pass;
these supply no numerical-profile or performance qualification on the new GPU.

## Data and completed evidence

All 1,454,193 required unique training/development source files were checksummed
on r2, including all **1,281,167 ImageNet images**. Existing ImageNet at
`/home/scratch/danson/datasets/imagenet` matched every pinned checksum and is
reused read-only through `outputs/c3-source-overlays/imagenet`. The previous
132 GB Ghost transfer is unnecessary on r2. Existing r2 data was not modified.
CLEVR, SPair, ADE and VOC inputs were copied using their exact sample manifests.
Final partition contents were not copied or evaluated by the migration.

The fourteen completed ADE fits remain historical c3 results. Fresh r2
reconstruction verifies all 28,294 retained records, source counts, recipes and
hashes. Both models' selected learning rates are below the upper search edge;
all completed ADE temporary workspaces are absent. No fit was restarted.
Original task 37's exact-float verifier failure and its raw state remain intact;
the separate successful recovery and ResNet18 extension proofs establish the
completed scientific evidence.

Receipts on r2:

- `outputs/r2-migration-20261026/{request,copy-receipt,source-audit,origin-queue}.json`.
- `outputs/r2-acceptance-20261026/origin-dense-verification.json`.
- `outputs/r2-acceptance-20261026/imagenet-source-audit/receipt.json`, explicitly
  recording existing-source reuse and zero ImageNet images transferred.
- `outputs/r2-acceptance-20261026/gpu-runtime-probe.json`.
- `outputs/r2-acceptance-20261026/relocation-validation.json` and each plan's
  `.validation.json`: all relocated source/checkpoint/runtime pins, model/readout
  contracts and full train/development data checks pass.

Shared preparation is reused once through the new execution root's `data`
symlink. Original c3-named input directories on r2 are immutable provenance;
new r2 execution outputs have distinct ownership. The frozen worker changes
only its execution root and ImageNet availability-receipt path. Task recipes,
models, datasets, numerical profiles, budgets and tolerances are unchanged.

## Paused queue and next action

Use the **private Pueue configuration**, not r2's default scheduler:

```bash
cd /home/scratch/danson/repos/bench
pueue -c outputs/r2-acceptance-20261026/pueue/config.yml status
```

| Task | Work | Dependency |
| --- | --- | --- |
| 0 | Full ImageNet-1K pretrained ViT-B/16 | Verified source availability |
| 1 | Full SPair pretrained ResNet18 and ViT-B/16 | Successful 0 |
| 2 | Joint CLEVR LR/WD/budget search for pretrained ViT-B/16 | Successful 1 |

All three tasks are queued; group `bench-r2-acceptance` is **Paused**, with
parallelism one. No benchmark evaluation has started on r2. Exact task birth
times, commands, dependencies, plan hashes and the resume command are in
`outputs/r2-acceptance-20261026/current-execution.json`. Inspect live identities
against that registry before modifying the queue. The private daemon preserves
r2's unrelated default queue and uses its own state/socket/log directory.

Only after authorization to resume acceptance, run:

```bash
pueue -c outputs/r2-acceptance-20261026/pueue/config.yml start --group bench-r2-acceptance
```

Each worker repeats its full preflight and requires the registered GPU, a fresh
owned output, sufficient capacity, independent result verification and actual
cleanup. Preserve frozen plans and inputs; derive/revalidate a new plan if any
execution or scientific contract changes. Do not rerun completed ADE fits.

The old c3 waiting tasks 56–59 are archived and removed after the validated r2
queue is checked. Completed c3 tasks and all their results remain preserved.
The broad acceptance goal stays paused. Remaining monitoring statistics,
training-overhead measurements, numerical qualification, calibration and release
audit are still open; queued-job success alone will not establish acceptance.
