# Evaluation Protocol

How every BeltWatch model, including the baselines, is scored. The code is in
[`src/beltwatch/evaluation/`](../src/beltwatch/evaluation) and
[`src/beltwatch/inference/postprocessing.py`](../src/beltwatch/inference/postprocessing.py).

> Results appear only from tracked runs. The first declared release's
> results are in the [model card](model_card.md#evaluation), and its raw
> outputs in [`results/beltwatch-0.1.0/`](results/beltwatch-0.1.0/README.md).

## Splits

Results use the **official-repaired** split policy described in the
[dataset card](dataset_card.md#split-policy-official-repaired). Every reported
result states the split manifest ID from `data/splits/zerowaste-f-splits.json`.

| Split | Used for |
|---|---|
| train | fitting models |
| val | model and hyperparameter selection, iterative error analysis |
| calibration | temperature scaling only |
| test (locked) | final evaluation of declared releases only |
| unseen-recording slice | test images from sequence 08, reported separately |

All valid test images are evaluated, including those the application would
route to review. No difficult cases are dropped.

## Segmentation metrics

- **Primary: foreground macro IoU.** Pixel confusion counts are summed over the
  whole evaluation set first. IoU = TP / (TP + FP + FN) is then computed per
  class, and the four target materials (cardboard, soft plastic, rigid plastic,
  metal) are averaged. Background is excluded. This is not a mean of per-image
  IoUs.
- **Supporting:** per-class IoU, Dice, precision, and recall; five-class mIoU
  as a conventional reference; pixel accuracy, shown only to illustrate how
  misleading it is (an all-background model scores highly on it).
- A class that never appears in the predictions or the ground truth has an
  undefined IoU. It is listed as undefined rather than counted as 0 or 1. A
  class that is present but completely missed scores 0.
- Pixels labelled `255` (padding and undefined regions) are ignored.

Evaluation runs on full frames at the working resolution (long side 768).
Predictions are mapped back to the original 1920×1080 frame with
`restore_probabilities` before metrics and coverage are computed.

## Visible coverage

The quantity shown to users, implemented once in `compute_coverage` and
applied to both predictions and ground truth:

```
coverage_c = pixels labelled c inside the inspection region
             / valid pixels inside the inspection region
```

- Reported as **estimated visible coverage**. It is not contamination by
  weight.
- **Coverage MAE** is in **percentage points of region area**: predicting 8%
  against a true 5% is an error of 3 points. Reported for the total and per
  class, together with the mean signed error (bias).
- Dataset evaluation uses the full frame as the inspection region.

## Review workload

- **Audit-positive image:** ground-truth target coverage above **5%** of the
  inspection region. This is a project convention, not an industry standard.
- Images are reviewed in descending score order. The curve plots the fraction
  of audit-positive images found against the fraction of images reviewed, and
  every reviewed image counts toward workload. Ties are broken randomly.
- Reported: prevalence, area under the curve (random ordering is about 0.5),
  and recall at 10%, 20%, 30%, and 50% workload.
- The achievable gain shrinks as prevalence grows: a perfect ordering's
  area is 1 − prevalence / 2. On ZeroWaste-f about 90% of val and test
  images are audit-positive at the 5% convention, so a perfect ordering
  scores about 0.55 against 0.50 for random, and the curve separates
  orderings poorly. The convention was declared before results and is kept;
  the model card reports the perfect ordering next to the measured ones.
- Orderings compared: random, the classical baseline, the neural model, and
  the neural model with an uncertainty share and random audits.

## Review-policy evaluation

Declared 2026-10-19, before any review-policy result on val
(`configs/review_policy_tuning.yaml`, version `rp-tuning-1`). The workload
curve above ranks images. This section measures the routing decision: which
images the policy sends to a person.

- **Data:** the val split, scored by the `beltwatch-0.1.0` bundle (T 1.04)
  through the serving pipeline (`process_image`), with the full frame as the
  inspection region. The calibration split stays reserved for temperature
  scaling, and the test split is spent.
- **Measured per policy:**
  - review rate: the share of images with `review_required`
  - recall: the share of audit-positive images that reach a person
  - false-alarm share: the share of flagged images that are not audit-positive
  - missed audit-positive images
  - routes (`retake_image`, `manual_review`, `review`, `none`) and reasons,
    including random audits
- **Grid:** coverage threshold 0.02 to 0.15, uncertain-fraction threshold
  0.05 to 0.30. The entropy threshold, priority weight, random audit rate
  (10%) and quality thresholds stay at review-policy-1's values. The random
  audit sample is keyed to the candidate version, so it is the same for
  every candidate.
- **Selection:** among candidates with recall of at least 0.95, the lowest
  review rate wins.
- **Adoption:** the selected candidate becomes `review-policy-2` only if
  review-policy-1 misses the recall floor, or if the candidate's review rate
  is at least 5 points lower.
- **Limits:**
  - Tuned and measured on the same val images, so the result is optimistic.
  - No untouched evaluation set remains, so no unbiased claim is possible.
  - An adopted policy needs a new declared release, and a new evaluation set
    before any claim beyond val.
  - The 5% audit-positive convention is unchanged.
  - At about 90% prevalence, no policy with a 0.95 recall floor can review
    much less than about 85% of images.

## Calibration

- **Temperature scaling** is fitted on the **calibration split only**
  (`python -m beltwatch.evaluation.calibration`), by minimizing the negative
  log-likelihood over uniformly sampled valid pixels. It works from
  probabilities, so it applies to the random forest as well as the neural
  models.
- **Expected calibration error** (15 equal-width bins, top-label) is reported
  **overall and on foreground pixels** (where the ground truth or the
  prediction is a target material). Background dominates the pixel count and
  can hide poor confidence on the materials. Every evaluation report includes
  both, together with reliability-diagram data, and the calibration command
  writes SVG reliability diagrams.
- **Release guard:** a release built with `--calibration` refuses a
  temperature that makes foreground ECE worse, unless the regression is
  explicitly allowed and recorded in the manifest.
- **One temperature trades foreground against overall calibration.** On
  the first full-dataset models (docs/error_analysis.md, 2026-10-14), fitting
  on all pixels left foreground ECE near 0.26 on val. Fitting on foreground
  pixels (`--fit-scope foreground`) roughly halved it but raised overall ECE
  from 0.019 / 0.038 to 0.061 / 0.084 (SegFormer / U-Net).
- **Per-class temperatures** (`--fit-scope per-class`) fit one temperature
  for each predicted class. A pixel's logits are divided by the temperature
  of its predicted class, so no prediction changes and IoU and coverage are
  unaffected; only confidence moves. Each class is fitted on the pixels
  predicted as that class, which minimizes the total NLL, since a pixel's NLL
  depends only on its own class's temperature. Evaluation and releases take
  either form (`--temperature 1.2` or `--temperature 1,1.4,1.6,1.8,2`, in
  label order). Measured on the first models, it did not fix foreground
  ECE (docs/error_analysis.md, second 2026-10-14 entry).
- **Foreground ECE is partly a measure of missed material.** Its subset
  includes material pixels predicted as background, which are wrong by
  construction and share their predicted class with correct background
  pixels. No recalibration that keeps the predicted class can separate them,
  so this ECE falls mainly with better recall. It is still reported as
  defined, next to overall ECE and NLL.
- **Choosing the temperature:** among the fit scopes that improve NLL,
  overall ECE and foreground ECE over the uncalibrated model on val, the one
  with the lowest val NLL. For SegFormer-B0 only the all-pixel fit
  qualifies; for the U-Net it is the per-class fit.

## Confidence intervals

- Percentile bootstrap that resamples **groups, never pixels**. Groups are
  recording sequences, or time blocks within sequences (`time_block_groups`).
  The group definition is stated with every interval.
- Model comparisons use a **paired** bootstrap: both models are evaluated on
  the same images and the same resampled groups, and the difference gets its
  own interval. Every report saves its per-image confusion matrices
  (`confusions.npz`), and `scripts/compare.py <report A> <report B>` computes
  the intervals for `A - B` on foreground macro IoU and each target class's
  IoU, under both group definitions. It refuses reports from different
  splits, split manifests, or image sets. With `--a` and `--b`, each side is
  several reports (for example one per training seed), and the difference
  is between their means; every resample applies the same groups to all of
  them.
- The test split covers only five sequences, so sequence-level intervals are
  wide. Time-block groups give narrower intervals but assume more
  independence than the data guarantees. Both are reported.

## Provenance

Every output records what produced it. `report.json`, `calibration.json` and
`error_analysis.json` carry a `provenance` block: the git commit (`-dirty` if
tracked files were modified), the model file, and its SHA-256. The benchmark
records the commit and the release's weights digest. Release manifests record
the commit they were built from. A neural model's digest matches the
`best_checkpoint_sha256` tag that MLflow logged at training time.

## Runtime benchmark

`scripts/benchmark.py --release model_releases/<v> --threads 4` measures, in a
fresh process:

- warm p50 and p95 latency of prediction alone, and of the full worker
  pipeline (quality checks, prediction, coverage, uncertainty, routing, and
  PNG encoding of the artifacts)
- a breakdown of the full pipeline into `process_image` and the PNG encoding
  of each artifact (mask, overlay, uncertainty), using the worker's own
  encoder (`jobs.pipeline.encode_png`, zlib level 3)
- sequential throughput
- peak resident memory

It reports these against the planning targets: p95 of at most 3 s per image,
at least 0.5 images per second, and a peak of at most 2 GB. The hardware is
recorded with the result, and the numbers count only on the documented target
(4 vCPU, 8 GB).

## Acceptance targets (proposed, not results)

- Foreground macro IoU at least 5 percentage points above the tuned
  random-forest baseline, with a positive paired-bootstrap interval where
  the groups allow it.
- Foreground macro IoU of at least 0.40, and total coverage MAE of at most
  5 percentage points.
- Recall versus review workload clearly better than random ordering.
- Every class's result published, including weak rare classes.

If a target is missed, it is reported and investigated. The metric is never
swapped for a more flattering one.
