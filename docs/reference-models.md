# Supported reference models

The supported benchmark set uses established public model releases. A qualifying
release supplies a public checkpoint or model file, documented construction and
preprocessing, and upstream evaluation or tests. A public URL alone is insufficient.
Reference acceptance pins the publisher, artifact checksum, model implementation,
library version, readouts and data-selection exposure.

| Model | Public upstream release | Package qualification |
| --- | --- | --- |
| torchvision ResNet18 | [Official weights and validation](https://docs.pytorch.org/vision/2.0/models/generated/torchvision.models.resnet18.html), `ResNet18_Weights.IMAGENET1K_V1` | Existing real-data reference checks; remaining task calibration is tracked in the acceptance ledger. |
| torchvision ViT-B/16 | [Official weights and validation](https://docs.pytorch.org/vision/2.0/models.html), `ViT_B_16_Weights.IMAGENET1K_V1` | Existing real-data reference checks; remaining task calibration is tracked in the acceptance ledger. |
| DINOv3 | [Official model and weight releases](https://github.com/facebookresearch/dinov3#pretrained-models) | Adapter retained; the exact checkpoint, library compatibility and local real-data qualification must be recorded before advertising a released reference. Publisher access terms apply. |

[reference-models.json](reference-models.json) pins the two existing public weight
artifacts. DINOv3 source eligibility does not certify an arbitrary local checkpoint.
Upstream published scores describe upstream evaluation, rather than score parity
with this package's different probe and split protocols.

Private training runs and experimental model studies are excluded from the
supported benchmark and release requirements. The generic adapter API remains
model independent. Small synthetic fixtures and seeded-untrained instances of
the same released model code remain internal correctness controls; they do not
count as public pretrained reference checkpoints.

The October 24 scope revision preserves previous results and plans as historical
evidence. Excluded model studies supply no current release requirement or passing
gate. Retained numerical, precision, budget, search and resource thresholds are
unchanged. See [acceptance.md](acceptance.md) for outstanding work.
