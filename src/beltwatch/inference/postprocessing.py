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
