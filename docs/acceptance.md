# Acceptance status

The October 24, 2026 scope revision follows the user requirement that supported
benchmark models have established, publicly available and well-tested checkpoint
or model files. [The model policy](reference-models.md) and
[reference manifest](reference-models.json) define the retained scope. Official
ResNet18 and ViT-B/16 weights are the current pretrained references. The DINOv3
adapter remains available; an exact official checkpoint requires local
qualification before becoming a released reference.

Private model prototypes, custom composition experiments and their training
trajectory studies are excluded. Their missing artifacts are no longer release
requirements. Completed historical results remain preserved; exclusion is a
scope change and does not establish a passing measurement. The previous full
ledger is retained as [historical evidence](../outputs/c3-acceptance-20261023/public-reference-scope-20261024/previous-worktree/docs/acceptance.md).

The current c3 queue has completed initial ADE diagnostics/reverification (37/54),
running ResNet18 extension 55, held ImageNet transfer 56, and queued public
ImageNet/SPair/CLEVR evaluations 57–59. Plans
were derived and CPU contracts revalidated before enqueueing. Their exact launch
and scientific-input audit are in the current scope directory. Its
`current-execution.json` points to the current family-plan metadata revision 2,
which removes stale study inventories while preserving the scientific jobs.

## Core correctness and operation

The implementation includes deterministic dataset preparation and visualization,
model-independent adapters, global and spatial extraction, frozen probes,
correspondence scoring, atomic results, exclusive output/workspace ownership,
verified resumption, identity invalidation and explicit pruning. Existing CPU
fixtures and CUDA checks cover geometry, annotation isolation, caller model/RNG
preservation, accumulation, reuse, regeneration and cleanup. Current verification
for this scope revision is recorded under
`outputs/c3-acceptance-20261023/public-reference-scope-20261024/`.

Source preparation and annotated galleries for the seven dataset tasks remain
in scope. Dataset task definitions, source checksums, sample manifests, scoring,
probe settings and existing numerical tolerances are unchanged. Internal
synthetic and seeded-untrained controls test implementation behavior and are not
advertised pretrained reference checkpoints.

The revised 0.1.0 wheel and source archive also build with the declared
`uv_build` backend (0.13.5, in an isolated temporary tool environment). Both
artifacts preserve all 37 package files exactly and contain no historical output
or excluded-model files. The extracted wheel resolves its packaged preparation
configuration and supported root CLI help without importing PyTorch. These checks
cover distribution contents and CLI/configuration loading; real-data acceptance
remains separate. Receipts: `package-artifact-check.json` and
`package-smoke-check.json` under the current scope directory.

The frozen acceptance packages also have an explicit source-provenance audit
against the current worktree. All seven task digests match. Across five worker
snapshots, the Python differences are the documented explicit-FP32 probe
initialization fix and feature-value read optimization; the ADE snapshots also
have formatting-only differences in ImageNet class metadata. Archive hashes
link these changes to the existing nine initialization-parity cases and 756 CPU
probe prediction comparisons. Extraction and dense numerical paths match by AST
comparison under the recorded FP32 default. Source identities remain distinct.
This reconciles provenance and component evidence; it adds no current-source GPU
score, end-to-end timing or universal profile qualification. Receipt:
`package-source-provenance.json` in the current scope directory.

## Retained real-data evidence and remaining evaluation

| Task | Official pretrained references | Remaining work |
| --- | --- | --- |
| ImageNet-100 and VOC monitoring | ResNet18 and ViT-B/16 real-data scores, repeatability and numerical comparisons exist. | Resolve probe-budget and precision adequacy before advertising calibrated monitoring or a cadence. |
| Full ImageNet-1K | ResNet18 control and wider development searches are verified. | Transfer missing training/development source images after dense diagnostics clean up; run and verify ViT-B/16. |
| ADE20K | Both official pretrained controls and all twelve initial LR/budget diagnostics are independently verified. ViT's upper-rate boundary is resolved. | Finish and verify the ResNet18 extension, resolve any remaining search edge, and prune before the ImageNet copy. |
| CLEVR count and distance | Both official pretrained reference architectures have verified full development results. | Complete retained ViT joint LR/WD and budget calibration; CNN expanded searches already provide evidence. |
| SPair-71k | Source preparation, geometry/reference kernel checks and independent verifier fixtures exist. | Run and independently verify both official pretrained references on the full development partition. |

All six pretrained ResNet18 ADE LR/budget diagnostics completed at 08:36 UTC on
October 25. Their independent verification reconstructs 12,126 records against
4,043 source files and 373,568,791 eligible development pixels. The native-rate
40-epoch seed-zero score differs from the original baseline by about -0.00000217
points against the unchanged 0.1-point tolerance. Its 161 changed canvas
prediction hashes establish score parity rather than bitwise identity. Actual
workspace deletion and zero retained temporary bytes are confirmed.

The best 40-epoch rate is 0.003 (17.506731404189928 mIoU), but the best 80-epoch
rate is the largest tested, 0.01 (17.493675269088396). The latter requires a wider
development search; completion of these fits does not pass search coverage.
ViT-B/16's six diagnostics are now complete and independently verified too.
ImageNet transfer stays held until the ResNet18 search edge is resolved and the
extension's actual cleanup is verified.
Evidence: `ade-search/results/resnet18-pretrained/verification.json` under the
current scope directory; the earlier native-only receipt remains preserved.

The ResNet18 extension is running as task 55 after successful independent
verification task 54. It adds two
fits at LR 0.03 and the existing 40/80-epoch budgets, reusing the six completed
results as independently reconstructed evidence. All package files and model,
data, recipe, execution and tolerance contracts remain unchanged. Its full CPU
model/environment/data checks pass. The transfer/ADE/recovery guards pass 34
focused tests. Measured prior fitting
times estimate about five additional GPU hours, plus setup and verification.
ViT's native-rate 40/80-epoch pair now passes independent reconstruction of all
4,042 records and 71 request pins. The scores are 30.020188985261655 and
30.04456958203907 mIoU, a doubled-budget difference of 0.024380596777415775
points. Selected epochs are 34 and 54. The first score matches baseline seed
zero exactly; the different budgets change all 2,021 prediction hashes. Receipt:
`ade-search/budget-checks/vit_b_16-pretrained/native-rate/verification.json`.

All six ViT variants are independently reconstructed: LR 0.003 scores
29.87683680759634/29.917411593712593 at 40/80 epochs; LR 0.01 scores
29.73027292606625/29.852875771449316. Both budgets select the native LR 0.001,
below the tested upper rate. Its 12,126 records, source-label counts and all
statistics hashes reconcile. The verification SHA256 is
`c20572c680999ed1e253a760f421aeb983b4c9c23f483eaf7c86f1dab616ed6a` at
`ade-search/results/vit_b_16-pretrained/verification.json`. Its actual workspace
is absent and retained temporary bytes are zero. Different-LR comparisons are
optimizer experiments, not numerical-profile parity tests. No full-probe
convergence threshold or full release acceptance follows from these diagnostics.

Original task 37 ended Failed(1) at 17:38:11 UTC after all twelve fits completed
and pruned: its frozen checker compared aggregate dictionaries by exact float
equality and rejected a 7.1e-15 discrepancy. Task 54 succeeded at 17:38:57 UTC,
independently reconstructing all 24,252 records with the existing 1e-10
metric-coherence bound and unchanged 0.1-point score tolerance. Original raw
state and failure remain preserved; no fit was restarted. The separate
`ade-search/recovery-verification/effective-state.json` has SHA256
`1a5061fc9bf372dde4001de572e27ea397fbd50351d8909f15bbe2d810fb71c3` and
truthfully retains `needs-search-extension` for ResNet18 alone. Its stale raw
per-model running row does not identify a live original worker.

Task 55 started at 17:38:57 UTC, passed CPU preflight and has active GPU
calibration. The requested intermediate milestone is reached; the handoff audit
at `handoff-20261025/manifest.json` checks all twelve statistics files, preserved
state, exact queue identities, live extension and held copy, and 71/134/137/148
static pins. Both original workspaces are absent; preserve the extension's
active workspace. Full acceptance remains open for the next agent. The current
transfer remains stashed, with no image copy started.

The current ImageNet copy is held task 56, with an independently checked combined
boundary for the initial public diagnostics and incremental search. Its 148
static input pins and verified runtime pass configuration validation; the live
dense results and cleanup remain required. Tasks 57–59 use the unchanged
metadata-v2 family plans. Their source/checkpoint pins, prior CPU contracts and
fresh outputs were rechecked before queue rebinding. Superseded tasks were
removed unstarted, and their exact identities and frozen requests are preserved.

The retained monitoring numerical evidence has been reconstructed under the
current analysis code and reconciled against the public checkpoint manifest.
All four historical pretrained comparisons pass the unchanged 0.1-point
tolerance: ResNet18 differs by 0.04 top-1 and 0.02359 mIoU points; ViT-B/16 by
0 and 0.01396 points. These checks use 2,500 ImageNet-100 and 256 VOC development
records per profile. All 28 pinned input files remain unchanged, and CUDA is
uninitialized. The receipt preserves each run's historical implementation and
hardware; it supplies no new GPU execution, full-task/profile qualification or
monitoring-budget acceptance. Evidence: `retained-numerical-evidence.json` in the
current scope directory.

The same reconciliation retains three passing internal untrained comparisons
and the recorded weak-ViT FP32/BF16 discrepancy:
0.12 percentage points against the unchanged 0.1-point tolerance. Use FP32 for
that internal diagnostic until resolved. Profile qualification remains specific
to the retained model, task, recipe and hardware; existing passes do not imply
universal numerical equivalence.

## Remaining release requirements

The requested intermediate handoff milestone is reached: the twelve initial
ADE fits are independently verified and the two-fit ResNet18 extension is
running. The handoff and cleanup checks are recorded above. The remaining
requirements below stay open for the next agent; ImageNet transfer stays held
until the complete ADE science/cleanup boundary passes. See
[c3-continuation.md](c3-continuation.md) for the exact stopping conditions.

1. Finish the retained full-task controls and development calibration, including
   search coverage, feature-scale sensitivity and adequate probe budgets. Resolve
   recipe choices before freezing task identifiers or API v1.
2. Keep independent score/statistics reconstruction, source/sample checks,
   identities and zero retained temporary state for every completed control.
3. Qualify numerical profiles on the supported reference cases. Retain all known
   failures; model exclusion cannot be reported as a successful parity test.
4. Establish monitoring claims on eligible public model releases. The proposed
   targets remain 120-second fresh-checkpoint latency, measured training overhead
   at most 5%, paired interval half-width at most delta/2, directional agreement
   at least 90%, and doubled-budget score stability for at least 90% of acceptance
   checkpoints with no supported qualifying-pair reversal. Delta remains two
   top-1 points and one mIoU point. Predeclare at least three qualifying pairs
   spanning CNN and ViT, including each family. Existing precision and budget
   failures remain open; matched training overhead is unmeasured, and no cadence
   recommendation is supported.
5. Record public checkpoint/model provenance and exact local qualification for
   every advertised reference. DINOv3 eligibility alone supplies no completed
   local checkpoint acceptance.

The source-only monitoring-cohort review is recorded in
`public-monitoring-cohort-review.json` under the current scope directory. LAION
documents a public release with upstream classification, retrieval and ADE20K
evaluation, making it a candidate for exact local qualification. Its specific
pinned artifact access was not established by the browser in this review.
[Publisher release and evaluations](https://github.com/LAION-AI/scaling-laws-for-comparison).
The previously collected CNN zoo still has conflicting dataset/provenance
metadata; its native-loading checks do not qualify it for the current reference
set. The alternative LAWA trajectory's access remains unverified after a
historical deletion observation. No reviewed cohort was admitted, and no new
GPU work was queued. Eligible predeclared monitoring pairs, reserved acceptance
and matched training-overhead evidence remain outstanding.

The package remains a 0.1 development implementation. No stable-v1 release or
completion claim follows from this scope revision. Current execution details
are in [c3-continuation.md](c3-continuation.md).
