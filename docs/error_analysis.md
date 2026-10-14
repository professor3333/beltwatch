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
  separately. (Run on 2026-10-14; see the next entry.)

### 2026-10-14: per-class temperatures and the pipeline breakdown

Same checkpoints as above (weights SHA-256 `7be82fde…` SegFormer, `394dfde9…`
U-Net). Kaggle CPU session at commit `25451a5`, recorded in every output.
Temperatures were fitted on the calibration split (200 images, 4M sampled
pixels) and checked on val (370 images, 740k sampled pixels). IoU and
coverage are unchanged by every fit (SegFormer 0.3186, U-Net 0.2929 val
foreground macro IoU).

Val calibration, SegFormer-B0:

| Fit scope | Temperature | NLL | ECE overall | ECE foreground |
|---|---|---|---|---|
| none | 1 | 0.2844 | 0.0191 | 0.2654 |
| all pixels | 1.04 | **0.2823** | **0.0151** | 0.2577 |
| foreground pixels | 1.74 | 0.3101 | 0.0609 | **0.1417** |
| per predicted class | 1.06, 0.90, 1.03, 0.89, 5.68 | 0.2849 | 0.0166 | 0.2709 |

Val calibration, U-Net:

| Fit scope | Temperature | NLL | ECE overall | ECE foreground |
|---|---|---|---|---|
| none | 1 | 0.3315 | 0.0381 | 0.3009 |
| all pixels | 1.24 | 0.3140 | 0.0136 | 0.2581 |
| foreground pixels | 1.96 | 0.3630 | 0.0836 | **0.1404** |
| per predicted class | 1.18, 1.44, 1.32, 2.03, 1.00 | **0.3081** | **0.0134** | 0.2365 |

Per-class temperatures are in label order (background, cardboard, soft
plastic, rigid plastic, metal). The U-Net never predicts metal, so its metal
temperature stays 1.

1. **Per-class temperatures did not fix foreground calibration.** For the
   U-Net they improve on the single temperature in all three measures, but
   foreground ECE only drops to 0.237. For SegFormer they are worse than
   the single temperature on val in all three, and foreground ECE (0.271)
   is worse than uncalibrated.
2. **Why no argmax-preserving fit can fix foreground ECE.** The foreground
   subset is the pixels where the ground truth *or* the prediction is a
   material. On val, 28% (SegFormer) and 18% (U-Net) of that subset are
   material pixels predicted as background (from the error-report
   confusion counts). Those pixels are wrong by construction. They are
   predicted background, like the true background pixels that make up 93%
   (SegFormer) and 95% (U-Net) of background predictions, and a temperature
   cannot tell the two apart without the label. Foreground ECE is therefore partly a
   measure of missed material. It improves with better recall, not with
   recalibration. Fitting on foreground pixels lowers it only by making
   every prediction less confident, which raises overall ECE three- to
   four-fold and worsens NLL.
3. **SegFormer's metal temperature (5.68) rests on very few pixels.** Metal
   is 0.02% of calibration pixels (12 of 200 images contain any), against
   0.28% on val. The fit flattens metal predictions that are mostly false
   positives on the calibration sequences. That this is what worsens its
   val result is likely but not verified.

**Decision: SegFormer-B0 uses the all-pixel temperature (T = 1.04).** It is
the only scope that improves NLL, overall ECE and foreground ECE over the
uncalibrated model on val. (By the same rule the U-Net would use its
per-class temperatures, which have the lower val NLL of its two qualifying
fits.) Foreground ECE stays reported as measured
(0.258); it is not replaced by a metric the model does better on. Choosing
the scope on val is model selection, which is what val is for.

**Pipeline breakdown** (4-core Kaggle CPU, 20 val frames, p50 seconds):

| Model | Full pipeline | `process_image` | Prediction alone | Overlay PNG | Uncertainty PNG | Mask PNG | Images/s |
|---|---|---|---|---|---|---|---|
| SegFormer-B0 | 1.95 | 0.96 | 0.65 | 0.75 | 0.19 | 0.03 | 0.509 |
| U-Net | 2.15 | 1.19 | 0.83 | 0.75 | 0.16 | 0.03 | 0.468 |

PNG encoding is about half of the full pipeline. The U-Net misses the 0.5
images per second target; SegFormer meets it narrowly. The artifact PNGs are
now encoded at zlib level 3, which on full-size val frames encoded the
overlay about 2.3× faster for a file about 7% larger. The effect on Kaggle is
not yet measured.
