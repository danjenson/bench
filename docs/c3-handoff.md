# c3 benchmark acceptance handoff

The current supported reference-model scope was revised on October 24, 2026 by
the user: established publicly released and well-tested checkpoint/model files
are required. Read [reference-models.md](reference-models.md),
[acceptance.md](acceptance.md) and [c3-continuation.md](c3-continuation.md).

The user-requested intermediate handoff is reached as of 17:42 UTC on October
25. All twelve initial public ADE fits are independently verified (24,252
records), both original workspaces are pruned, and the two-fit ResNet18 LR 0.03
extension is running. The broader acceptance work stops here for this agent.
Full SPEC acceptance remains unfinished. Start the next session with
`outputs/c3-acceptance-20261023/public-reference-scope-20261024/handoff-20261025/manifest.json`
and `current-execution.json` in that scope directory.

The old handoff and queue describe a superseded experimental acceptance study.
Their original documentation is preserved under
`outputs/c3-acceptance-20261023/public-reference-scope-20261024/previous-worktree/`.
The old immutable handoff manifest remains at
`outputs/acceptance/c3-handoff-20261023/manifest.json` for historical provenance;
it is not the current supported model set.

Keep completed scores, original source archives and transfer receipts intact.
Use the verified main environment and RTX 4090 recorded in the continuation.
Inspect existing tasks before starting replacements, derive fresh scoped plans,
relocate execution paths and revalidate model/environment/source identities.
The previous chain was paused and its unstarted tasks removed after archiving
their identities. Original task 37 ended Failed(1) after all twelve fits
completed, at the anticipated exact-float metadata comparison. Verification-only
task 54 succeeded and preserved that actual outcome and raw state. Its separate
`ade-search/recovery-verification/effective-state.json` verifies both controls;
only ResNet18's upper LR edge remains unresolved. Never restart 37 or infer a
live worker from its preserved stale per-model status.

Current task 55 is running the ResNet18 extension. Task 56 remains stashed;
57–59 are queued behind it for ImageNet/SPair/CLEVR. Inspect the exact live
identities, including birth time, command, group and dependencies, before acting.
The continuation records frozen request-v3/runner-v3 for the extension and
request-v4/runner-v4 for the held copy, plus all release conditions.
Superseded waiting tasks remain historical evidence.

Next: observe 55 through completion, inspect its independent eight-variant
ResNet18 proof and actual cleanup, and resolve any remaining upper search edge
using a separately derived/revalidated extension. Release only the exact held
56 identity after the complete combined scientific boundary passes. Preserve
55's active workspace until its owner finishes. All agent observation cells
and audit subprocesses have ended; historical inputs, checkpoints and source
data remain intact. Remaining monitoring, numerical, calibration and release
requirements are listed in the acceptance ledger.

Missing full ImageNet training/development images must still be transferred from
Ghost after retained ADE diagnostics finish and temporary work is pruned.
Recheck capacity, exact copy lists and all source checksums before full ImageNet
evaluation. The source-overlay and shared prepared manifests remain reusable.
No backbone training restart or undocumented recipe/tolerance change is needed
for this scope revision.
