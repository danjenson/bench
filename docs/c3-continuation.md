# c3 acceptance continuation (historical)

**October 26: continuation moved to r2.** The ResNet18 extension completed at
22:43 UTC on October 25, passed independent verification and pruned. Both models'
upper-rate boundaries are resolved. Source waiting jobs 56–59 have been archived
and removed; the c3 group is paused. The current source execution registry points
to the r2 successor. Read [r2-handoff.md](r2-handoff.md) before taking action.
The remaining text is the preserved October 25 milestone snapshot.

The active scope is `public-reference-models-20261024`, defined by
[reference-models.md](reference-models.md). Current benchmark references are the
declared official pretrained ResNet18 and ViT-B/16 checkpoints. Private model
studies and missing unpublished checkpoints are excluded. Preserve historical
results, source archives, transfers and cancelled-run evidence.

## Requested stopping point reached

The user requested stopping after independent verification of all twelve initial
ADE fits and the start of the two-fit ResNet18 extension, followed by a handoff
and cleanup. That milestone is reached. Task 54 succeeded at 17:38:57 UTC on
October 25, and task 55 started immediately afterward. Its CPU model preflight
passed, its calibration report is running, and NVIDIA confirms active GPU work
in the verified Python environment. The handoff audit is
`handoff-20261025/manifest.json` under the scope directory below.

Both initial dense workspaces are absent, and their reports retain zero temporary
bytes. All observation cells and audit subprocesses have ended. Preserve task
55's active workspace. The broader acceptance goal is to be paused after this
handoff, as requested; the next agent owns subsequent acceptance work. Keep
ImageNet transfer 56 stashed until its complete scientific and cleanup gate
passes. This milestone is not full SPEC acceptance.

## Queue snapshot at the October 25 handoff

The current serial queue, checked at 17:42 UTC on October 25, is:

| Task | Work | State |
| --- | --- | --- |
| 37 | Initial public ADE LR/budget diagnostics | Terminal Failed(1) at 17:38:11; all twelve fits complete/pruned; frozen verifier's exact-float assertion failed |
| 54 | Verification-only recovery of completed initial diagnostics | Success at 17:38:57; all 24,252 records independently verified |
| 55 | ResNet18 LR 0.03 at existing 40/80-epoch budgets | Running since 17:38:57; validated CPU preflight passed; GPU calibration active |
| 56 | Missing full ImageNet transfer and checksum audit | Stashed; dependency 55 plus independent combined science/cleanup gate |
| 57 | Full ImageNet-1K pretrained ViT-B/16 | Queued after 56 |
| 58 | Full SPair pretrained ResNet18 and ViT-B/16 | Queued after 57 |
| 59 | Joint CLEVR LR/WD/budget search for pretrained ViT-B/16 | Queued after 58 |

All work uses `bench-c3-acceptance`, parallelism one. Do not restart task 37 or
change its frozen inputs. The scientific plan still has twelve initial ADE fits
and two additional ResNet18 fits. Verification recovery introduces no fit.

The scope directory is
`outputs/c3-acceptance-20261023/public-reference-scope-20261024/`.
Its `current-execution.json` is authoritative for current queue identities.
Bind creation time, command, group and dependencies; IDs have been reused.
Current receipts and inputs are:

- `launch.json`: unchanged initial ADE task 37.
- `ade-search/recovery-verification/launch.json`: completed verification task 54,
  frozen request/runner, 134 static pins and runtime validation.
- `ade-search/recovery-verification/effective-state.json`: verified initial
  results; SHA256 `1a5061fc9bf372dde4001de572e27ea397fbd50351d8909f15bbe2d810fb71c3`.
- `handoff-20261025/manifest.json`: milestone/cleanup audit, current curated
  queue, all twelve statistics hashes and 71/134/137/148 static-pin checks.
- `ade-extensions/resnet18-lr0p03/launch-v3.json`: task 55,
  `request-v3.json` and `runner-v3/`.
- `imagenet-transfer-v2/queue-launch-v4.json`: current downstream queue.
- `imagenet-transfer-v2/hold-v4.json`: exact held task 56 identity.
- `imagenet-transfer-v2/request-v4.json` and `runner-v4/`: copy worker.
- `imagenet-transfer-v2/family-requeue-audit.json`: unchanged metadata-v2 family
  plans, all source/checkpoint pins, prior CPU contracts and fresh outputs.

All six ResNet18 initial diagnostics finished at 08:36 UTC on October 25.
Independent verification checks 12,126 records, 4,043 source files and
373,568,791 eligible pixels. Calibration, verification and all six statistics
hashes match. Its actual workspace is absent and retained temporary bytes are
zero. The verification SHA256 is
`c0ec913ab916030d74e9189145f1c21780d40e9e82b620b0caad51afa75c3fe3`.

ResNet18 selects LR 0.003 at 40 epochs (17.506731404189928 mIoU), but the upper
tested LR 0.01 at 80 epochs (17.493675269088396). Task 55 adds LR 0.03 at both
budgets with unchanged seed zero, WD 0.01, source/model/task/execution and
tolerance contracts. Its verifier reconstructs the six prior fits and two new
fits. All 37 package files are unchanged from the original diagnostic snapshot.
CPU model/environment and full 18,189/2,021 training/development validation pass.
Estimated new fitting/scoring time is 18,200 seconds, about five hours, based on
completed corresponding budgets; setup and verification add time.

All six ViT-B/16 diagnostics also finished and are independently verified.
The reconstructed mIoU scores are:

| LR | 40 epochs | 80 epochs |
| --- | --- | --- |
| 0.001 | 30.020188985261655 | 30.04456958203907 |
| 0.003 | 29.87683680759634 | 29.917411593712593 |
| 0.01 | 29.73027292606625 | 29.852875771449316 |

Both budgets select LR 0.001, below the tested upper rate. Native-rate selected
epochs are 34/54; their difference is 0.024380596777415775 points. The native
40-epoch score and predictions match original baseline seed zero exactly. All
12,126 ViT records, 4,043 source files and 373,568,791 eligible pixels reconcile.
The verified artifact at `ade-search/results/vit_b_16-pretrained/verification.json`
has SHA256 `c20572c680999ed1e253a760f421aeb983b4c9c23f483eaf7c86f1dab616ed6a`.
Its actual workspace is absent and retained temporary bytes are zero. Earlier
partial/native proofs remain preserved. Different-LR score differences are
optimizer experiments, not numerical-profile parity tests.

Original task 37 ran from October 24 17:23:40 UTC to October 25 17:38:11 UTC.
After all fits completed and pruned, its frozen verifier failed at line 138's
exact dictionary float comparison, as anticipated. Producer
30.044569582039077 differs from reconstruction 30.04456958203907 by 7.1e-15.
The reviewed verifier applies the existing 1e-10 metric-coherence bound; the
registered score tolerance remains 0.1 points. No fit was restarted.

Task 54 ran from 17:38:50 to 17:38:57 UTC. It independently verified all 24,252
records, source/class counts, recipes, model/execution metadata, baseline parity
and hashes, with CUDA uninitialized. It preserved the original failed state
byte-for-byte at `ade-search/recovery-verification/original-state-preserved.json`
(SHA256 `62b911e347faaf6f68c454093d9072d274587af81823759f134e95946986e06d`).
The old raw state still labels the ViT row running; that is preserved stale
metadata, not a live original worker. Use Pueue identities and the separate
`effective-state.json`. Never rerun task 37 or overwrite its outcome.
The existing ResNet18 verification is unchanged; the missing ViT proof was
published after successful independent verification. Recovery proofs are under
`ade-search/recovery-verification/proofs/`. Thirty-four focused guard tests pass.

Task 55 is running, bound to successful recovery 54 and that separate effective
state. Its source bundle, model, two fits and output locations are unchanged.
State: `ade-extensions/resnet18-lr0p03/state.json`; output:
`ade-extensions/resnet18-lr0p03/results/resnet18-pretrained/`; calibration log:
the exact job `log` path in `request-v3.json`. Preserve its active `.work`.
The handoff GPU observation records host PID 1069220 in the verified Python;
host PIDs are not visible in the command sandbox's `/proc`. Authoritative Pueue
task 55, its exact command/birth identity and active calibration are confirmed.
Task 56 independently requires successful exact 54/55 identities, the effective
original state and incremental proof, original 24,252 plus new 4,042 records,
matching source evidence and actual workspace deletion. It reconstructs the
scientific evidence again before copying. Both budgets on each public control
must select an interior LR. ViT's initial search meets that boundary. If the
ResNet18 extension still selects an upper edge, derive and revalidate a further
extension while the copy stays
held. Keep the original raw failure and truthful search-edge states intact.

The copy worker's 148 static pins and verified runtime pass configuration
validation. This supplies no passing combined scientific boundary while the
fits are unfinished. Before enqueueing task 56, compare its live identity with
`hold-v4.json` and verify all completed science/cleanup. The worker repeats
runtime, source pins, queue identity and scientific verification before any
image bytes are copied.

Copy 1,154,478 missing training/development images (132,411,896,262 bytes) from
Ghost after all ADE work prunes. Capacity uses the unchanged public ViT ImageNet
plan, its peak estimate and a 10 GB margin. Checksum all 1,281,167 source images;
final partitions remain unread. Tasks 57–59 keep the metadata-v2 scientific
plans and commands. Full source/model/environment preflight repeats in each
worker after transfer, with fresh scoped outputs and strict ownership.

All superseded waiting tasks were removed unstarted, and their identities and
frozen inputs remain preserved. Older queue revisions are historical, including
38–41 and 44–53. The initial reverse hold-task binding was caught before any
affected work started. The latest scientific-verification rebinding is archived
under `ade-search/recovery-verification/superseded-downstream-queue.json`.
Do not recreate old tasks. Private-model task 13 was explicitly cancelled during
an excluded evaluation and its workspace pruned; it is not scientifically
complete. Previous worktree/queue evidence remains under `previous-worktree/`
and `previous-queue.json`.

Verified Python: `outputs/environments/bench/bin/python` (Python 3.13.13,
Torch 2.14.1+cu130). Environment receipt:
`outputs/environments/setup-receipt.json`. GPU UUID:
`GPU-23c48372-0992-f3a5-9759-f9469423137b`.

The read-only Ghost preflight at 18:16 UTC on October 24 found no live Pueue
tasks and confirmed one missing ImageNet file's metadata. It copied no images
and changed no jobs. Full source integrity still requires the held checksum audit.

`package-source-provenance.json` audits five frozen packages against all 37
current package files. All seven task digests and parsed class definitions match.
Python differences are the documented FP32 probe initialization and feature-value
read optimization, linked by exact archive hashes to existing CPU parity checks.
Extraction and dense numerical paths match by AST under the recorded FP32
default. Keep distinct implementation identities; this adds no current-source
GPU or end-to-end performance qualification and changes no frozen input.

`public-monitoring-cohort-review.json` reviews earlier public trajectory
candidates. LAION documents release/evaluation evidence; exact local qualification
remains pending. The older CNN source has unresolved provenance, and another
ResNet trajectory's access is unverified. No cohort was admitted or GPU work
added. Keep the active reference manifest unchanged while its hash is pinned.
Monitoring precision/budget, eligible pairs, reserved acceptance, numerical
qualification and matched training overhead remain open in
[acceptance.md](acceptance.md). Queue success alone is not SPEC acceptance.
