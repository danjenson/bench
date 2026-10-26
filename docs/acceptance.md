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

As of October 26, the remaining evaluations are staged on r2 with fresh plans,
outputs and a private paused queue: full ImageNet ViT (0), full SPair controls
(1), and ViT CLEVR joint calibration (2). All fourteen planned ADE fits are
complete and verified. r2's existing ImageNet matched all 1,281,167 pinned
training/development image checksums; no missing-image transfer is needed there.
All 1,454,193 required source files passed checksums. Environment, CPU model/
readout contracts and full data preflights pass on r2, with fresh reconstruction
of all 28,294 ADE records. The old c3 waiting tasks are archived and removed.
The broader acceptance goal stays paused. Read [r2-handoff.md](r2-handoff.md).

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
| Full ImageNet-1K | ResNet18 control and wider development searches are verified; all source images are checksum-verified on r2. | Run and verify the staged ViT-B/16 evaluation. |
| ADE20K | Both official pretrained controls and all fourteen planned LR/budget diagnostics are independently verified and pruned; both upper-rate boundaries are resolved. | Broader recipe/scale/budget qualification remains; no additional ADE fit is currently queued. |
| CLEVR count and distance | Both official pretrained reference architectures have verified full development results. | Complete retained ViT joint LR/WD and budget calibration; CNN expanded searches already provide evidence. |
| SPair-71k | Source preparation, geometry/reference kernel checks and independent verifier fixtures exist. | Run and independently verify both official pretrained references on the full development partition. |

All fourteen planned pretrained ADE LR/budget diagnostics are complete,
independently verified and pruned. Initial ResNet18 and ViT-B/16 runs contribute
24,252 records, and the ResNet18 LR 0.03 extension contributes 4,042. Both models'
source counts reconcile against 4,043 files and 373,568,791 eligible development
pixels. All temporary workspaces are absent and retained temporary bytes are zero.

ResNet18 selects LR 0.003 at 40 epochs (17.506731404189928 mIoU) and LR 0.01 at
80 epochs (17.493675269088396). Testing LR 0.03 scored 17.424993430556192 and
17.47618981145191 respectively, resolving its upper-rate boundary. The extension
finished at 22:43:18 UTC on October 25; its verification SHA256 is
`dd0392fcd01b4145a45a494ecbf05f876c0ed9f27cee06a51e6964a948d40158` at
`ade-extensions/resnet18-lr0p03/results/resnet18-pretrained/verification.json`
under the original public-reference scope directory.

ViT selects LR 0.001 at both budgets (30.020188985261655 and
30.04456958203907 mIoU), below the tested upper rate. Native-rate selected epochs
are 34/54. Its verification SHA256 is
`c20572c680999ed1e253a760f421aeb983b4c9c23f483eaf7c86f1dab616ed6a` at
`ade-search/results/vit_b_16-pretrained/verification.json`. ResNet18's native
40-epoch score differs from its original baseline by -0.00000217 points; ViT's
matches exactly. The unchanged registered tolerance is 0.1 points.

Original task 37's Failed(1) outcome remains preserved: after fitting and cleanup,
its frozen checker rejected a 7.1e-15 exact-float dictionary discrepancy. Recovery
54 independently checked all twelve original fits using the existing 1e-10
metric-coherence bound, without refitting. Its separate effective state and
original raw-state archive remain unchanged. Extension 55 subsequently succeeded.
The earlier handoff milestone remains historical evidence; no completed fit was
restarted during the migration.

On r2, `outputs/r2-acceptance-20261026/origin-dense-verification.json` records fresh
reconstruction of all 28,294 records and the combined resolved search/cleanup
boundary. `outputs/r2-migration-20261026/source-audit.json` verifies all required
train/development source bytes. Three relocated family plans pass CPU model,
environment and complete data preflight. The private r2 queue remains paused;
the source c3 waiting tasks were archived and removed. GPU runtime smoke passes,
but new hardware supplies no borrowed timing or universal numerical-parity claim.
The remaining probe-scale/budget, monitoring and release requirements below are
still open; these development diagnostics alone do not establish full acceptance.

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

The requested October 25 handoff milestone was reached, and the October 26
workload migration is staged and validated on r2. Evaluations and the broader
acceptance goal remain paused. The following requirements stay open. Read
[r2-handoff.md](r2-handoff.md) for the current queue and resume instructions.

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
are in [r2-handoff.md](r2-handoff.md).
