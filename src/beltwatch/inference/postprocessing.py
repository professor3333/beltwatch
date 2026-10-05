"""Turn per-pixel predictions into the quantities shown to users.

``compute_coverage`` is the single implementation of *estimated visible
coverage*, used both by the inference worker and by evaluation (which applies
it to ground-truth masks too), so the number a user sees is exactly the number
that was evaluated.

Coverage of class *c* is the number of pixels labelled *c* inside the
inspection region divided by the number of valid pixels inside the region. It
is a fraction of visible image area, **not** contamination by weight.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw

from beltwatch.labels import CLASS_NAMES, IGNORE_INDEX, TARGET_CLASS_IDS

FULL_FRAME: tuple[tuple[float, float], ...] = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def rasterize_region(
    polygon: Sequence[Sequence[float]], width: int, height: int
) -> npt.NDArray[np.bool_]:
    """Rasterize a polygon with coordinates normalized to [0, 1] into a boolean mask."""
    if len(polygon) < 3:
        raise ValueError("an inspection region needs at least 3 vertices")
    points = []
    for vertex in polygon:
        if len(vertex) != 2:
            raise ValueError(f"vertex {vertex!r} is not an (x, y) pair")
        x, y = float(vertex[0]), float(vertex[1])
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(f"vertex {vertex!r} is outside the normalized [0, 1] range")
        points.append((x * width, y * height))
    canvas = Image.new("1", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(points, fill=1, outline=1)
    return np.asarray(canvas, dtype=bool)


@dataclass(frozen=True)
class Coverage:
    per_class: dict[str, float]
    """Visible coverage per target material, as a fraction in [0, 1]."""
    total_target: float
    valid_pixels: int


def compute_coverage(
    labels: npt.NDArray[np.integer],
    region: npt.NDArray[np.bool_],
    valid: npt.NDArray[np.bool_] | None = None,
) -> Coverage:
    """Estimated visible coverage of each target material inside ``region``.

    ``valid`` excludes undefined pixels (and ``IGNORE_INDEX`` labels are always
    excluded). An empty region raises, because coverage would be undefined.
    """
    if labels.shape != region.shape:
        raise ValueError(f"labels {labels.shape} and region {region.shape} differ in shape")
    keep = region & (labels != IGNORE_INDEX)
    if valid is not None:
        keep &= valid
    n_valid = int(keep.sum())
    if n_valid == 0:
        raise ValueError("inspection region contains no valid pixels")
    counts = np.bincount(labels[keep].astype(np.int64), minlength=len(CLASS_NAMES))
    per_class = {CLASS_NAMES[c]: float(counts[c]) / n_valid for c in TARGET_CLASS_IDS}
    return Coverage(per_class, float(sum(per_class.values())), n_valid)


# Overlay colours (RGB) per class ID; background is not drawn.
CLASS_COLORS: dict[int, tuple[int, int, int]] = {
    1: (255, 159, 28),  # cardboard: orange
    2: (46, 196, 182),  # soft plastic: teal
    3: (231, 29, 120),  # rigid plastic: magenta
    4: (255, 230, 0),  # metal: yellow
}
UNCERTAIN_COLOR = (90, 120, 255)


def normalized_entropy(proba: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Per-pixel entropy of ``C x H x W`` probabilities, scaled to [0, 1]."""
    p = np.clip(proba, 1e-8, 1.0)
    entropy: npt.NDArray[np.float32] = (
        -(p * np.log(p)).sum(axis=0) / np.log(proba.shape[0])
    ).astype(np.float32)
    return entropy


def render_overlay(
    rgb: npt.NDArray[np.uint8],
    labels: npt.NDArray[np.integer],
    uncertain: npt.NDArray[np.bool_],
    region: npt.NDArray[np.bool_],
    alpha: float = 0.5,
) -> npt.NDArray[np.uint8]:
    """Blend class colours over the image, hatch uncertain pixels, dim outside the region."""
    out = rgb.astype(np.float32)
    for class_id, color in CLASS_COLORS.items():
        sel = labels == class_id
        out[sel] = (1 - alpha) * out[sel] + alpha * np.asarray(color, dtype=np.float32)
    h, w = labels.shape
    yy, xx = np.mgrid[0:h, 0:w]
    hatch = uncertain & (((xx + yy) // 6) % 2 == 0)
    out[hatch] = 0.4 * out[hatch] + 0.6 * np.asarray(UNCERTAIN_COLOR, dtype=np.float32)
    out[~region] *= 0.35
    return np.clip(out, 0, 255).astype(np.uint8)


def render_uncertainty(entropy: npt.NDArray[np.float32]) -> npt.NDArray[np.uint8]:
    """Grayscale heat map: brighter means more uncertain."""
    return (np.clip(entropy, 0, 1) * 255).astype(np.uint8)
