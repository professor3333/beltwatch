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

_None yet: no model has been trained on the full dataset._
