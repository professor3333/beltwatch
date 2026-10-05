"""Pixel confusion matrices and segmentation metrics.

Confusion counts are **aggregated over the whole evaluation set** before IoU is
computed. The headline metric, foreground macro IoU, averages the IoU of the
four target materials and excludes background, so abundant background pixels
cannot dominate it. Per-image confusion matrices are kept so the bootstrap can
re-aggregate over resampled recording groups.
"""

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from beltwatch.labels import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, TARGET_CLASS_IDS

Confusion = npt.NDArray[np.int64]
"""``C x C`` counts; rows are ground truth, columns are predictions."""


def confusion_matrix(
    pred: npt.NDArray[np.integer],
    target: npt.NDArray[np.integer],
    num_classes: int = NUM_CLASSES,
    ignore_index: int = IGNORE_INDEX,
) -> Confusion:
    """Confusion counts for one image. Pixels whose target is ``ignore_index`` are skipped."""
    if pred.shape != target.shape:
        raise ValueError(f"prediction {pred.shape} and target {target.shape} differ in shape")
    keep = target != ignore_index
    t = target[keep].astype(np.int64)
    p = pred[keep].astype(np.int64)
    if t.size and (t.max() >= num_classes or p.max() >= num_classes or p.min() < 0):
        raise ValueError("class id out of range")
    counts = np.bincount(t * num_classes + p, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


@dataclass(frozen=True)
class SegmentationMetrics:
    iou: dict[str, float]
    dice: dict[str, float]
    precision: dict[str, float]
    recall: dict[str, float]
    foreground_macro_iou: float
    mean_iou: float
    pixel_accuracy: float
    undefined_classes: tuple[str, ...]
    """Classes absent from both predictions and ground truth (IoU undefined)."""

    def as_dict(self) -> dict[str, object]:
        return {
            "foreground_macro_iou": self.foreground_macro_iou,
            "mean_iou": self.mean_iou,
            "pixel_accuracy": self.pixel_accuracy,
            "iou": self.iou,
            "dice": self.dice,
            "precision": self.precision,
            "recall": self.recall,
            "undefined_classes": list(self.undefined_classes),
        }


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else math.nan


def metrics_from_confusion(confusion: Confusion) -> SegmentationMetrics:
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    names = [CLASS_NAMES[c] for c in range(confusion.shape[0])]

    iou = {n: _ratio(tp[c], tp[c] + fp[c] + fn[c]) for c, n in enumerate(names)}
    dice = {n: _ratio(2 * tp[c], 2 * tp[c] + fp[c] + fn[c]) for c, n in enumerate(names)}
    precision = {n: _ratio(tp[c], tp[c] + fp[c]) for c, n in enumerate(names)}
    recall = {n: _ratio(tp[c], tp[c] + fn[c]) for c, n in enumerate(names)}

    foreground = [iou[CLASS_NAMES[c]] for c in TARGET_CLASS_IDS]
    defined_fg = [v for v in foreground if not math.isnan(v)]
    defined_all = [v for v in iou.values() if not math.isnan(v)]
    return SegmentationMetrics(
        iou=iou,
        dice=dice,
        precision=precision,
        recall=recall,
        foreground_macro_iou=float(np.mean(defined_fg)) if defined_fg else math.nan,
        mean_iou=float(np.mean(defined_all)) if defined_all else math.nan,
        pixel_accuracy=_ratio(float(tp.sum()), float(confusion.sum())),
        undefined_classes=tuple(n for n, v in iou.items() if math.isnan(v)),
    )


def foreground_macro_iou(confusion: Confusion) -> float:
    return metrics_from_confusion(confusion).foreground_macro_iou
