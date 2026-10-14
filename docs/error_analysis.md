# Error Analysis

`scripts/error_report.py` generates a static HTML report (with JSON) for a
model on **val**. The locked test split is refused.

```bash
uv run python scripts/error_report.py --model unet --model-dir models/unet-resnet18-ce \
    --split val --out reports/errors-unet
open reports/errors-unet/report.html
```

## What the report contains

| Section | Purpose |
|---|---|
| Per-class IoU | Find the weak materials |
| Confusion matrix (row-normalized SVG; raw counts in JSON) | See what each material is mistaken for, or missed as |
| False positives / false negatives per material | The images with the most wrongly predicted and the most missed pixels |
| Largest coverage errors | The images where the number shown to users is most wrong |
| High-confidence mistakes | Wrong pixels predicted with probability above 0.9, the most harmful errors |
| Uncertain but correct | Where uncertainty routes work to reviewers even though the model is right |
| Foreground macro IoU by slice | Sequence, ground-truth coverage, sharpness, and brightness |

Galleries show the original, the ground truth, and the prediction side by
side. Colours: cardboard orange, soft plastic teal, rigid plastic magenta,
metal yellow, grey ignored.

## From pattern to next experiment

| Failure | Likely next investigation |
|---|---|
| Transparent plastic disappears | Resolution, relevant examples, boundary ambiguity |
| Printed paper becomes cardboard | Label consistency and learned texture shortcuts |
| Small metal pieces are missed | Class frequency, crop scale, and resolution |
| Large coverage overestimates | Merged regions, shadows, or preprocessing errors |
| New-camera images fail | Domain shift and camera normalization |
| All models fail at the same boundary | Annotation ambiguity |

Verify every pattern against the actual images before acting on it, and
choose the next experiment from the most consequential recurring failure, not
from architecture curiosity. Findings and the experiments they led to are
recorded below as they happen.

## Findings log

### 2026-10-14: first full-dataset models, val (370 images)

Models: U-Net run `f3b13047` (best epoch 18, commit `b37f003`) and
SegFormer-B0 run `abacd486` (best epoch 6, commit `d72dda9`, same code), both
trained on Kaggle. Reports and error reports come from Kaggle CPU sessions. The
calibration, error-report and benchmark outputs do not record the code
commit; `main` was at `1d3acf0` when they ran. The best epochs were selected
on val, so these numbers are optimistic. Patterns below were checked against
the val galleries.

Where each material's pixels go (share of ground-truth pixels; SegFormer /
U-Net):

| Truth | Correct | → background | → soft plastic | → rigid plastic | → cardboard |
|---|---|---|---|---|---|
| cardboard | 0.665 / 0.755 | 0.325 / 0.231 | 0.009 / 0.013 | 0.000 / 0.001 | |
| soft plastic | 0.570 / 0.657 | 0.332 / 0.224 | | 0.013 / 0.046 | 0.071 / 0.073 |
| rigid plastic | 0.140 / 0.238 | 0.322 / 0.249 | **0.483 / 0.432** | | 0.053 / 0.081 |
| metal | 0.224 / 0.000 | **0.706 / 0.597** | 0.021 / 0.124 | 0.032 / 0.254 | 0.018 / 0.025 |

1. **Rigid plastic is mostly predicted as soft plastic** (48% / 43% of its
   pixels). The gallery examples are crumpled, translucent containers. A
   large white clamshell tray in sequence 01 is missed by both models:
   SegFormer predicts it as background, the U-Net mostly as cardboard.
2. **Metal is mostly missed as background** (71% / 60%). The U-Net never
   predicts metal; it sends a quarter of metal pixels to rigid plastic,
   for example a crushed metallic item in clutter. SegFormer finds some
   metal, but its largest metal false positives are a reflective, printed
   package labelled soft plastic (sequence 04, frames 561–641).
3. **Background → cardboard is the largest false-positive source** (3.3% /
   5.2% of background pixels, which is 26% / 33% of all predicted
   cardboard). Examples include a kraft-paper envelope and small brown paper
   pieces. This drives the U-Net's coverage overestimate (bias +1.6 pp).
   SegFormer instead misses more of every material as background, hence its
   underestimate (−2.3 pp).
4. **Dense, cluttered frames lose large regions.** The largest coverage
   errors are crowded frames from sequences 01, 03 and 12. In the worst
   SegFormer frame, printed cardboard and transparent bags are missed.
5. **Low-coverage images are weak.** Foreground macro IoU is 0.06 / 0.01
   below 1% true coverage (4 images) and 0.15 / 0.16 at 1–5% (35 images),
   against 0.35 / 0.32 at 15% or more.
6. **The sequences rank differently.** SegFormer: 01 0.374, 03 0.366,
   12 0.265, 04 0.156. U-Net: 12 0.373, 03 0.328, 01 0.296, 04 0.222.
   SegFormer's overall lead comes from sequences 01 and 03, where its metal
   detections count; it is behind on 04 and 12.

**Calibration (temperature fitted on the calibration split, checked on
val).** A single temperature cannot fix overall and foreground calibration
at once. Fitted on all pixels (SegFormer T 1.04, U-Net T 1.25), it brings
val overall ECE to 0.015 / 0.014, but foreground ECE stays at 0.258 / 0.258.
Fitted on foreground pixels (T 1.74 / 1.96), it brings foreground ECE to
0.142 / 0.140 but raises overall ECE to 0.061 / 0.084 and worsens NLL.

**Next investigations** (not yet run):
- Rigid vs soft plastic: label consistency for crumpled translucent
  containers, then material-aware crop sampling (finding 1).
- Metal: crop scale and inference resolution, since the pieces are small,
  and class weighting (finding 2).
- Calibration: one temperature per predicted class. This keeps every
  prediction unchanged and can treat background and the materials
  separately.
