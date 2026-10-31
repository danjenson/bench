# Numerical profiles

`fp32-v1` is the Python API and CLI default. The released monitoring recipe
uses FP32 for classification and segmentation, with the same profile for every
checkpoint. CPU execution uses FP32. `bf16-v1` remains an explicit CUDA option;
its availability does not establish parity with a reference score.

A reference comparison must match the numerical profile recorded with that
reference. Qualifying BF16 requires matched real evaluations under the same
model state, adapter, task recipe, sample bank, software and hardware, followed
by independent reconstruction using `bench.analysis.numerical_parity`. Preserve
the individual seed results and the registered 0.1-percentage-point tolerance.
Profiles retain distinct identities even when parity passes. Confidence intervals
do not replace the numerical tolerance, and checkpoint scores must not determine
which profile that checkpoint uses.

## Retained failures

The October 27 classification screen used the historical 2,500-image ImageNet-100
development bank, seed zero, and 400/800 probe updates on an RTX 5000 Ada. Thirteen
of twenty FP32/BF16 comparisons pass; all seven failures below remain failures.
Differences are BF16 minus FP32 in top-1 percentage points.

| Public checkpoint | Probe updates | Difference | Tolerance |
| --- | ---: | ---: | ---: |
| CNN 1 | 400 | −0.12 | 0.10 |
| CNN 1 | 800 | −0.20 | 0.10 |
| CNN 49 | 400 | −0.16 | 0.10 |
| CNN 50 | 400 | +0.12 | 0.10 |
| ViT 10 | 800 | +0.16 | 0.10 |
| ViT 20 | 800 | −0.12 | 0.10 |
| ViT 9 | 400 | −0.12 | 0.10 |

The [original restriction receipt](../outputs/r2-acceptance-20261026/numerical-profile-restrictions-20261027.json)
retains exact model/checkpoint hashes, request and statistics hashes, hardware,
and the paired intervals for each failure. Its task-to-profile mapping describes
the historical caller; the released monitoring recipe uses FP32 for both tasks.
The [October 30 reconstruction](../outputs/r2-acceptance-20261026/numerical-profile-policy-current-20261030/summary.json)
independently reproduces all twenty pairs from 100,000 saved prediction records
and checksum-bound prepared labels, retaining the same thirteen passes and seven
failures. It performs no encoder inference or new BF16 qualification.
The [default-policy receipt](../outputs/r2-acceptance-20261026/numerical-default-policy-20261028/manifest.json)
records the subsequent API/CLI change to FP32.

The earlier seed-zero untrained torchvision ViT-B/16 control also fails on
**ImageNet-100 classification**: FP32 9.76 versus BF16 9.88 top-1, a 0.12-point
difference. That historical measurement used an RTX 4090, 2,500 development
images and a different implementation. It remains restricted to FP32 and is
an internal correctness control. Its segmentation comparison passed. The
[retained evidence reconciliation](../outputs/c3-acceptance-20261023/public-reference-scope-20261024/retained-numerical-evidence.json)
preserves all eight original public-reference/internal-control comparisons,
including their source, model and hardware identities.
The original run files for that older control are absent from this workspace;
its restriction relies on the retained reconciliation receipt. The October 30
reconstruction inventories every original evidence path and makes this limit
explicit.

## Qualification scope

Earlier passing public ResNet18/ViT-B/16 comparisons and segmentation candidate
screens apply to their exact recorded settings. They establish no blanket BF16
claim for the current 7,500-image classification bank, changed recipes, other
models, full-task probes or different hardware. Unmeasured combinations remain
unqualified for reference comparison. The current FP32 monitoring acceptance
route does not require rerunning failed BF16 cases to relabel them as passes.

These evidence paths are for release review. Benchmarking a caller-owned model
does not require downloading the historical acceptance checkpoints or feature
banks. Calibration, fresh monitoring acceptance, measured overhead and final
v1 source/reference qualification are summarized in [acceptance](acceptance.md).
