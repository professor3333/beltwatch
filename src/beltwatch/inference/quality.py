"""Input-quality checks: is this image usable at all?

Separate from model uncertainty: a sharp, well-exposed image can still be
uncertain, and a blurred image can yield a confident but untrustworthy answer.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image

from beltwatch.review.policy import QualityPolicy

_ANALYSIS_LONG_SIDE = 768


@dataclass(frozen=True)
class QualityReport:
    sharpness: float
    """Variance of the Laplacian of the grayscale image (higher = sharper)."""
    mean_brightness: float
    clipped_fraction: float
    issues: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "sharpness": round(self.sharpness, 3),
            "mean_brightness": round(self.mean_brightness, 3),
            "clipped_fraction": round(self.clipped_fraction, 6),
            "issues": list(self.issues),
        }


def assess_quality(rgb: npt.NDArray[np.uint8], policy: QualityPolicy) -> QualityReport:
    img = Image.fromarray(rgb).convert("L")
    scale = _ANALYSIS_LONG_SIDE / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)))
    gray = np.asarray(img, dtype=np.float64)
    lap = (
        -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    sharpness = float(lap.var()) if lap.size else 0.0
    mean = float(gray.mean())
    clipped = float(((gray <= 2) | (gray >= 253)).mean())

    issues = []
    if sharpness < policy.min_sharpness:
        issues.append("blurred")
    if mean < policy.min_mean_brightness:
        issues.append("underexposed")
    if mean > policy.max_mean_brightness:
        issues.append("overexposed")
    if clipped > policy.max_clipped_fraction:
        issues.append("clipped_exposure")
    return QualityReport(sharpness, mean, clipped, tuple(issues))
