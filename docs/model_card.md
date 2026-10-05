# Model Card: BeltWatch segmentation models

> **No model has been trained on the full dataset yet.** This card describes
> the models, data, and evaluation as built. Sections marked *pending* are
> filled in from tracked runs, never from expectations.

## Intended use

Segment **cardboard, soft plastic, rigid plastic, and metal** in RGB
inspection images of a **paper-stream recycling conveyor**. The output is a
*suggested* estimate of visible coverage inside an operator-defined
inspection region, plus a review priority, which a person confirms or
corrects before any audit record is exported.

Primary users: facility quality technicians, line supervisors, process
engineers, and the computer-vision engineers who support them.

## Not for

- Estimating contamination **by weight**, bale purity, or regulatory compliance.
- Certifying that an image is clean: background means "outside the four
  target labels", not "verified paper".
- Materials outside the four-class taxonomy (glass, textiles, food waste, ...).
- Unreviewed, automatic decisions about loads, suppliers, or people.
- Other facilities, cameras, conveyor types, or lighting without local
  validation data.

## Models

| Model | Architecture | Initialization | Parameters |
|---|---|---|---|
| All-background | Constant prediction | none | 0 |
| Random forest | Per-pixel multiscale intensity, edge, and texture features plus Lab colour, 384-px working resolution | none | (trees) |
| U-Net | ResNet-18 encoder (frozen BatchNorm statistics), GroupNorm decoder | torchvision ImageNet-1k weights | 14.3 M |
| SegFormer-B0 | MiT-B0 hierarchical transformer encoder, all-MLP decoder | `nvidia/mit-b0` ImageNet weights (**noncommercial license**) | 3.7 M |

The neural models share one preprocessing pipeline (`pp-1`: RGB, long side
768, ImageNet normalization, padding to a multiple of 32) used identically in
training and serving. They train on 512 × 512 crops and are evaluated on full
frames, with predictions restored to the original resolution. Checkpoints
refuse to load under a different preprocessing version or label map.

**Deployed model:** *pending*. It will be selected on validation foreground
macro IoU, minority-class IoU, memory, and measured CPU latency. A well-tuned
U-Net is a legitimate winner.

## Training data

ZeroWaste-f (Zenodo record 6412647, version 1.2.1; see the
[dataset card](dataset_card.md)): 4,503 frames from one paper-recycling
facility's conveyor, with polygon labels for the four materials.

Split policy **official-repaired**:

- the release's splits are kept
- val sequences 02 and 10 form a grouped calibration holdout
- training frames within 500 frames of evaluation frames in the same sequence
  are excluded, along with training duplicates of evaluation images
- the test split is locked
- sequence 08 is the only unseen-recording slice

Every run records the split manifest ID, the dataset MD5, the git commit, and
the encoder weights in MLflow.

## Evaluation

Protocol: [evaluation.md](evaluation.md). The primary metric is
**foreground macro IoU**, from confusion counts aggregated over the
evaluation set. Supporting metrics:

- per-class IoU, Dice, precision, and recall
- visible-coverage MAE in percentage points
- recall versus review workload
- ECE overall and on foreground pixels
- p50/p95 latency, throughput, and memory on the target CPU

Bootstrap intervals resample recording groups, never pixels.

| Result | Value |
|---|---|
| Validation metrics, all models | *pending* |
| Calibration (temperature; ECE before and after) | *pending* |
| Locked test, declared release only | *pending* |
| Unseen-recording slice (sequence 08) | *pending* |
| CPU benchmark on 4 vCPU / 8 GB | *pending* |

Proposed targets (planning values, not results): at least 5 points of
foreground macro IoU above the tuned random forest, foreground macro IoU of
at least 0.40, coverage MAE of at most 5 points, and warm p95 of at most 3 s
per image on CPU.

## Known and expected failure modes

Expected from the data and the task, and to be confirmed or refuted by the
error-analysis report:

- translucent or crumpled soft plastic blending into paper
- printed paper or glossy flyers predicted as cardboard
- small metal pieces missed, a rare class at low resolution
- coverage overestimated where regions merge, or on shadows
- overlapping material and ambiguous boundaries
- blur, poor exposure, and cameras or lighting unlike the training data

## Uncertainty and review

Each result carries per-pixel normalized entropy, the uncertain fraction of
the region, input-quality checks (sharpness, exposure), and a route:
`retake_image`, `manual_review`, `review`, or `none`. A deterministic random
share of low-risk images is always reviewed. Entropy cannot detect every
unfamiliar input, and a confident background prediction does not certify
cleanliness.

## Release process

A release is an immutable bundle containing:

- the model with its SHA-256 checksum
- the preprocessing version and label map
- the temperature and its calibration report
- the review policy
- provenance, limitations, and licenses

It is promoted with a recorded reason and the version it replaces.
Thresholds and temperatures are chosen on val or calibration, never on test.
A temperature that worsens foreground ECE is refused unless explicitly
allowed. Rollback restores the complete bundle. See
[operations.md](operations.md).

## Ethical and privacy considerations

- Images should show only the conveyor. Facility imagery needs the operator's
  permission, no recording of workers, and agreed retention rules.
- Uploads expire after 24 hours by default, and images never appear in logs.
- Reviewer feedback is used for training only with the audit's opt-in
  consent, and only after annotation review.
- The dataset is attributed and noncommercial (license discrepancy recorded),
  and the SegFormer weights are noncommercial.
