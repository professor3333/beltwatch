"""Segmentation losses that respect the ignore index.

* Cross-entropy over all five classes (the baseline loss).
* Foreground Dice: soft Dice averaged over the four target materials, computed
  over the whole batch. It directly rewards overlap on the minority classes that
  cross-entropy tends to under-weight.
"""

import torch
import torch.nn.functional as F

from beltwatch.labels import IGNORE_INDEX, NUM_CLASSES, TARGET_CLASS_IDS


def cross_entropy(
    logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor | None = None
) -> torch.Tensor:
    return F.cross_entropy(logits, target, weight=class_weights, ignore_index=IGNORE_INDEX)


def foreground_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0
) -> torch.Tensor:
    """1 - mean soft Dice over target classes; ignore pixels are excluded."""
    valid = (target != IGNORE_INDEX).unsqueeze(1)
    probs = torch.softmax(logits.float(), dim=1) * valid
    onehot = F.one_hot(target.clamp(0, NUM_CLASSES - 1), NUM_CLASSES).permute(0, 3, 1, 2)
    onehot = onehot.float() * valid
    classes = list(TARGET_CLASS_IDS)
    p, t = probs[:, classes], onehot[:, classes]
    intersection = (p * t).sum(dim=(0, 2, 3))
    denominator = p.sum(dim=(0, 2, 3)) + t.sum(dim=(0, 2, 3))
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1 - dice.mean()


def segmentation_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    kind: str,
    dice_weight: float = 1.0,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    ce = cross_entropy(logits, target, class_weights)
    if kind == "ce":
        return ce
    if kind == "ce_dice":
        return ce + dice_weight * foreground_dice_loss(logits, target)
    raise ValueError(f"unknown loss {kind!r}")
