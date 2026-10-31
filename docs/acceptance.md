# v1.0.0 verification

The released scope contains seven fixed task recipes and fourteen development
references for official pretrained torchvision ResNet18 and ViT-B/16. All sixteen
positive/weak calibration controllers and twenty-four reporting seeds complete.
The optional DINOv3 adapter requires qualification of each exact checkpoint.

| Gate | Verified result |
| --- | --- |
| Calibration | 16/16 controllers; closed learning-rate/decay boundaries and adequate doubled budgets. |
| Reporting | All 24 selected positive seeds independently reconstruct. |
| References | 14/14 entries, 891,114 record bindings, with checkpoint, numerical, task, sample and source provenance. |
| Release source | 26 qualified Python modules; seven release-status-only identity changes; scoring fields and sample bytes unchanged. |
| CUDA correctness | 39 tests pass, without failures or skips. |
| Installed CPU behavior | 411 tests pass in 25.23 seconds, without failures or skips, with original-repository and external-network access blocked. |
| CLI/API/schema | Package 1.0.0 metadata, all seven recipes, FP32 defaults and schema-major compatibility verified. |
| Offline resources | All fourteen scores and 873 interval occurrences across six reports read using only the three packaged resources. |
| Packaging | Actual 1.0.0 source distribution and wheel built from that distribution; exact source/runtime payloads and offline lockfile checks pass. |

The runtime inputs are datasets and the caller's model. Recipes, reference scores
and monitoring tables are included; baseline model weights, fitted heads, feature
banks and historical acceptance artifacts are not needed for comparisons.
Reproducing a reference evaluation requires its original public checkpoint and
matched protocol. References describe development partitions and recorded
selection exposure, rather than independent held-out upstream evaluations.

## Monitoring measurements

Protected development and six prospectively reserved public checkpoint states
pass the declared direction and budget gates. Each task detects all five
qualifying fresh directions confidently, with coverage across CNN and ViT.
Meaningful fuller-reference changes require two top-1 points or one mIoU point
with a paired interval excluding zero. The agreed October 27 policy requires
at least 90% confident direction agreement and at least three qualifying pairs
across both families. Every paired 95% interval and half-width is published;
there is no absolute interval-width cap. Fuller-reference and doubled-probe
budgets pass their registered stability checks.

Six declared ModelZoo/LAION standalone two-task calls take 112.49–116.27 seconds.
Three revised official torchvision CNN calls take 79.77–91.39 seconds; the official
ViT call takes 116.58 seconds. Actual matched training overhead measures 3.6658%
CNN and 3.4506% ViT, including evaluator startup, waits and complete sampled
allocation qualification. These measurements retain their actual source,
adapter, data, hardware and optimizer-step cadence. They do not establish a
universal runtime or recommended cadence for another training setup.

## Numerical and historical qualifications

FP32 is the API/CLI default and the profile of all fourteen included references.
Thirteen of twenty historical monitoring FP32/BF16 comparisons pass; all seven
failures and the separate earlier weak-ViT classification failure remain FP32
restricted. Unmeasured combinations remain unqualified. See
[numerical profiles](numerical-profiles.md) for exact failures and scope.

Sixteen historical 0.1x/10x feature-scale controls at variance floor 1e-6 reconstruct,
with maximum absolute difference 0.00013364 percentage points. They retain their
100-update/global-224/dense-512 settings. Whole measured-source bytes were not
recovered; current normalization components and full-task calibration are
separately qualified. These historical measurements do not establish empirical
coverage of every changed task, budget or numerical profile.

Original failures remain preserved in local verification evidence: the earlier
148.38-second CNN caller, inadequate reference-budget prefixes and the
weak-ViT/ADE accounting failure. Its scientific results and actual zero retained
state have an independent CPU recovery; the original peak counter remains
unqualified. Three historical source alternatives remain explicit. The compiled
loss candidate failed its prospective precision criterion and is excluded.

## Verification resources

The installed `bench/reference-scores.json` preserves each reference's original
measurement and final source/preparation binding. `bench/monitoring-evidence.json`
and `bench/monitoring-intervals.csv` preserve all 873 interval occurrences,
including failed prefixes and uncertain directions. The JSON binds the CSV by
SHA-256. Reading these resources requires no historical output tree.

The repository keeps user documentation, source, examples, verification helpers
and their complete tests. Machine-specific handoffs, migration programs, raw
acceptance logs, model downloads and generated distributions are excluded from
the release source. Registry upload and hosted publication are separate from a
local Git release tag.
