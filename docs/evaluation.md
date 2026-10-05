# Evaluation Protocol

How every BeltWatch model, including the baselines, is scored. The code is in
[`src/beltwatch/evaluation/`](../src/beltwatch/evaluation) and
[`src/beltwatch/inference/postprocessing.py`](../src/beltwatch/inference/postprocessing.py).

> No results are reported yet. Numbers appear here only after they come from
> tracked runs.

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
- Orderings compared: random, the classical baseline, the neural model, and
  the neural model with an uncertainty share and random audits.

## Confidence intervals

- Percentile bootstrap that resamples **groups, never pixels**. Groups are
  recording sequences, or time blocks within sequences (`time_block_groups`).
  The group definition is stated with every interval.
- Model comparisons use a **paired** bootstrap: both models are evaluated on
  the same images and the same resampled groups, and the difference gets its
  own interval.
- The test split covers only five sequences, so sequence-level intervals are
  wide. Time-block groups give narrower intervals but assume more
  independence than the data guarantees. Both are reported.

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
