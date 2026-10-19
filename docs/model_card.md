# Model Card: BeltWatch segmentation models

> **Release `beltwatch-0.1.0`** (SegFormer-B0, declared 2026-10-15). Every
> number on this card comes from a tracked run. The release's own outputs,
> including its single locked-test evaluation, are in
> [`docs/results/beltwatch-0.1.0/`](results/beltwatch-0.1.0/README.md).

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

**Deployed model: SegFormer-B0**, release `beltwatch-0.1.0`: MLflow run
`abacd486`, weights SHA-256 `7be82fde…`, preprocessing `pp-1`, temperature
1.04, review policy `review-policy-1`. It was chosen on val, before the test
split was read, on these grounds:

- Foreground macro IoU 0.319 against the U-Net's 0.293. The paired
  difference, +0.026 (block interval −0.032 to +0.087), does not separate
  the two.
- Metal IoU 0.173 against 0 (block interval of the difference +0.008 to
  +0.324). The U-Net never predicts metal. The U-Net is ahead on rigid
  plastic, by 0.047 (−0.072 to +0.020).
- On the Kaggle CPU benchmark the SegFormer met the 0.5 images/s target
  (0.509) and the U-Net missed it (0.468). The SegFormer used more memory:
  976 MB peak against 748 MB.

Training configuration: the baseline (cross-entropy, ordinary crop sampling,
no class weights). Of the declared ablations, CE + Dice, material-aware crops
and class weights, none met the adoption rule ([training.md
§6](training.md#6-declared-experiments-segformer-b0)). Over three seeds the
configuration scores 0.3186 / 0.3198 / 0.3215 on val (mean 0.3200, range
0.29 points). The release is the seed-0 run, fixed before the last two seeds
were trained.

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

Claims are limited to **held-out time windows of the same recordings**: the
official test split shares recordings with training, except sequence 08
([dataset card](dataset_card.md#splits-and-leakage-findings)). Val selected
the best epochs, tuned the random forest and chose the model, so val numbers
are optimistic. The test split was read once, for the declared release.

### Locked test split (929 images, declared release, scored once)

Scored on 2026-10-15 at commit `1c2e080`. Intervals: 95% paired percentile
bootstrap over recording sequences (5 groups) and over 1000-frame blocks
within sequences (69 groups). The block intervals assume more independence
than the data guarantees.

| Model | Foreground macro IoU | Sequence interval | Block interval |
|---|---|---|---|
| All-background | 0.000 | | |
| Random forest (tuned) | 0.152 | 0.125–0.161 | 0.140–0.164 |
| **SegFormer-B0 (release)** | **0.336** | 0.285–0.368 | 0.309–0.378 |
| Release − random forest | **+0.184** | +0.158 to +0.207 | +0.162 to +0.219 |

Per class (IoU / precision / recall):

| Class | Release | Random forest | Release − forest, IoU (sequence interval) |
|---|---|---|---|
| cardboard | 0.536 / 0.666 / 0.733 | 0.425 / 0.476 / 0.800 | +0.111 (+0.097 to +0.128) |
| soft plastic | 0.509 / 0.766 / 0.603 | 0.165 / 0.195 / 0.515 | +0.344 (+0.320 to +0.368) |
| rigid plastic | 0.109 / 0.425 / 0.128 | 0.005 / 0.049 / 0.005 | +0.104 (+0.036 to +0.194) |
| metal | 0.191 / 0.382 / 0.277 | 0.013 / 0.087 / 0.015 | +0.178 (+0.104 to +0.192) |

The all-background model has IoU 0 and recall 0 for every material; its
pixel accuracy is 0.820, against 0.895 for the release.

**Estimated visible coverage** (full frame as the region), MAE in
percentage points, with the mean signed error:

| Model | Total MAE (bias) | Cardboard | Soft plastic | Rigid plastic | Metal |
|---|---|---|---|---|---|
| All-background | 17.97 (−17.97) | 11.38 | 5.74 | 0.79 | 0.07 |
| Random forest | 16.81 (+16.43) | 8.57 | 10.15 | 0.84 | 0.07 |
| Release | **4.98** (−0.65) | 4.17 | 2.32 | 0.73 | 0.07 |

**Unseen recording (sequence 08, 215 of the 929 test images).** Release
foreground macro IoU 0.321: cardboard 0.575, soft plastic 0.495, rigid
plastic **0.038**, metal 0.177. Random forest 0.151. One recording is a
single sample of a new time and scene; it does not establish generalization
to other recordings.

**Calibration** (temperature 1.04, fitted on all pixels of the calibration
split):

| Set | NLL before → after | ECE overall | ECE foreground |
|---|---|---|---|
| Calibration split (fit) | 0.3278 → 0.3275 | 0.0088 → 0.0093 | 0.195 → 0.187 |
| Val | 0.2844 → 0.2823 | 0.019 → 0.015 | 0.265 → 0.258 |
| Test | 0.2922 → 0.2899 | 0.019 → 0.015 | 0.242 → 0.234 |

The effect is small. Foreground ECE stays high because its subset includes
material predicted as background, which no temperature that keeps the
predicted class can fix ([evaluation.md](evaluation.md#calibration)).

**Recall versus review workload** (images ordered by predicted total
coverage; audit-positive means more than 5% true coverage):

| Ordering | Area | Recall at 10% / 20% / 30% / 50% reviewed |
|---|---|---|
| Random (all-background, random tie order) | 0.503 | 0.098 / 0.203 / 0.306 / 0.507 |
| Random forest | 0.538 | 0.112 / 0.223 / 0.335 / 0.546 |
| Release | 0.542 | 0.112 / 0.223 / 0.335 / 0.555 |
| Perfect ordering (for reference) | 0.552 | 0.112 / 0.223 / 0.335 / 0.558 |

Prevalence is **89.7%** (833 of 929 images). At that prevalence even a
perfect ordering gains at most 1–6 points of recall over random at these
workloads, so this curve cannot show the review ordering to be useful. The
release's first 30% of reviews are all audit-positive, as are the forest's.
The ordering with the uncertainty share and random audits was not measured.

### Validation (370 images; used for selection)

Kaggle CPU re-scores of the trained checkpoints. Block intervals over 31
groups; the sequence intervals have too few groups (4) and are not reported.

| Model | Foreground macro IoU (block interval) | Cardboard | Soft plastic | Rigid plastic | Metal | Coverage MAE (bias), pp |
|---|---|---|---|---|---|---|
| All-background | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 16.97 (−16.97) |
| Random forest (tuned) | 0.135 (0.122–0.154) | 0.396 | 0.135 | 0.007 | 0.002 | 17.76 (+17.69) |
| U-Net (ResNet-18) | 0.293 (0.272–0.317) | 0.526 | 0.470 | 0.176 | 0.000 | 3.89 (+1.61) |
| SegFormer-B0 (release) | 0.319 (0.246–0.394) | 0.517 | 0.456 | 0.129 | 0.173 | 4.43 (−2.34) |

Paired differences on val (block intervals): SegFormer − forest +0.184
(+0.119 to +0.252), U-Net − forest +0.158 (+0.142 to +0.174), SegFormer −
U-Net +0.026 (−0.032 to +0.087).

### CPU benchmark

**Planning host profile (4 vCPU / 8 GB).** The bundle was rebuilt from the
declared weights and checked against the declaration, then benchmarked with
4 threads on 20 val frames at 1920 × 1080. Each run used a GitHub-hosted
runner with 4 vCPU, with the serving image limited to 4 CPUs and 8 GB
([procedure](operations.md#benchmark-on-the-planning-host),
[records](results/beltwatch-0.1.0/host-benchmark/)). GitHub assigned a
different CPU model to each run. These are shared cloud VMs, not dedicated
hardware.

| Measure | Run 1 (EPYC 7763) | Run 2 (EPYC 9V45) | Target |
|---|---|---|---|
| Full pipeline p50 / p95 | 0.76 s / 0.79 s | 0.46 s / 0.48 s | p95 ≤ 3 s |
| Prediction alone p50 | 0.32 s | 0.18 s | |
| Throughput (sequential) | 1.32 images/s | 2.17 images/s | ≥ 0.5 |
| Peak resident memory | 1071 MB | 1057 MB | ≤ 2 GB |
| End to end, one audit at a time, p50 / p95 | 1.61 s / 2.12 s | 1.08 s / 1.59 s | reported only |

The end-to-end times run from submission to a finished audit through the
HTTPS proxy, the API, the queue and the worker, with status polled every
0.5 s. In run 2, four audits in flight completed 1.63 audits/s (p95 1.93 s).
Run 1's four-in-flight figure is invalid: it reused the regions of the
sequential run, so most of its audits were served from the result cache.

**Kaggle CPU session** (4 x86 cores, more than 8 GB of RAM), from the
release session:

| Measure | Value | Target |
|---|---|---|
| Full pipeline p50 / p95 | 1.33 s / 1.59 s | p95 ≤ 3 s |
| Prediction alone p50 / p95 | 0.58 s / 0.75 s | |
| Throughput (sequential) | 0.73 images/s | ≥ 0.5 |
| Peak resident memory | 958 MB | ≤ 2 GB |

The overlay PNG takes 0.31 s of each image (0.75 s in an earlier Kaggle
session, before PNG encoding moved to zlib level 3). Prediction alone also
ran faster in this session, so part of the change is the machine.

### Targets

| Proposed target | Result on test | |
|---|---|---|
| ≥ 5 points over the tuned forest, positive interval | +18.4 points (sequence interval +15.8 to +20.7) | met |
| Foreground macro IoU ≥ 0.40 | 0.336 (sequence interval 0.285–0.368) | **missed** |
| Total coverage MAE ≤ 5 points | 4.98 | met, by 0.02 points; no interval was computed |
| Recall vs workload clearly better than random | area 0.542 vs 0.503 at 89.7% prevalence | **missed** (not demonstrable at this prevalence) |
| Seed-to-seed spread ≤ 2 points (val) | 0.29 points over 3 seeds | met |
| Warm p95 ≤ 3 s, ≥ 0.5 images/s, ≤ 2 GB | at most 0.79 s, at least 1.32 images/s, at most 1071 MB on the 4 vCPU / 8 GB profile (two runs); 1.59 s, 0.73 images/s, 958 MB on Kaggle | met |

Rigid plastic (0.109, and 0.038 on sequence 08) and metal (0.191) are weak.
A coverage MAE this close to the threshold could fall on either side on
another sample: on val, the configuration's seed-2 run scored 5.38.

## Known and expected failure modes

Observed in the val error analysis ([error_analysis.md](error_analysis.md),
2026-10-14 and 2026-10-15). The test split was not inspected.

- **Rigid plastic predicted as soft plastic**: 48% of rigid-plastic pixels,
  mostly crumpled, translucent containers. A large white clamshell tray was
  missed entirely.
- **Metal missed as background**: 71% of metal pixels. Some of the largest
  metal false positives are reflective printed packaging.
- **Background predicted as cardboard**: 26% of predicted cardboard,
  including kraft-paper envelopes and small brown paper pieces.
- **Material missed as background in general**: about a third of cardboard
  and soft-plastic pixels, so coverage is underestimated (val bias −2.3
  points; on test −0.7).
- **Dense, cluttered frames** have the largest coverage errors.
- **Images with little material** score poorly: foreground macro IoU 0.06
  below 1% true coverage and 0.15 at 1–5% (val, few images).

Expected from the task but not examined separately: overlapping material and
ambiguous boundaries, shadows, blur and poor exposure, inspection regions
that include non-belt areas, and cameras or lighting unlike the training
data.

## Uncertainty and review

The release's review policy, `review-policy-1`, holds design defaults
(coverage threshold 5%, entropy threshold 0.5, uncertain-fraction threshold
10%, random audit rate 10%). They were **not tuned** on val or calibration,
and their effect on review workload has not been measured.

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

Before the locked test split is read, the release is **declared** in
`configs/releases/<version>.yaml`. The declaration fixes:

- the weights' SHA-256 and MLflow run
- the temperature and its fit scope
- the review policy and preprocessing versions
- the frozen baselines scored next to the release
- the val scores the release session must reproduce
- the test evaluations and comparisons

The notebook's `release` mode builds the bundle and checks each step against
the declaration (`python -m beltwatch.release.declaration`). It also checks
runtime parity: the bundle as served and the checkpoint as evaluated must
give the same probabilities (`python -m beltwatch.release.parity`). Only
then does it score the test split, once. A completed test run is recorded in
`complete.json`, and a second one for the same release is refused.

The first declared release is `beltwatch-0.1.0`: SegFormer-B0, the seed-0
run of the finalist configuration, at T = 1.04. Its review policy,
`review-policy-1`, holds design defaults that were not tuned on val. Its
session reproduced the declared temperature and val scores (release 0.31856,
refitted forest 0.13477), and its test result is above. Any later tuning
needs a new untouched evaluation set before a new unbiased claim.
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
