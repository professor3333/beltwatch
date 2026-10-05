"""Coverage error: how far the visible coverage shown to users is from the truth.

Errors are reported in **percentage points of inspection-region area**. For
example, predicting 8% coverage when the truth is 5% is an error of 3 points.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from beltwatch.inference.postprocessing import Coverage

AUDIT_POSITIVE_THRESHOLD = 0.05
"""Project convention: an image is audit-positive if ground-truth target coverage
exceeds 5% of the inspection region. This is not an industry acceptance standard."""


def is_audit_positive(truth: Coverage, threshold: float = AUDIT_POSITIVE_THRESHOLD) -> bool:
    return truth.total_target > threshold


@dataclass(frozen=True)
class CoverageErrors:
    total_mae_pp: float
    per_class_mae_pp: dict[str, float]
    total_bias_pp: float
    """Mean signed error; positive means coverage is overestimated."""
    n_images: int

    def as_dict(self) -> dict[str, object]:
        return {
            "total_mae_pp": self.total_mae_pp,
            "per_class_mae_pp": self.per_class_mae_pp,
            "total_bias_pp": self.total_bias_pp,
            "n_images": self.n_images,
        }


def coverage_errors(predicted: Sequence[Coverage], truth: Sequence[Coverage]) -> CoverageErrors:
    if len(predicted) != len(truth) or not truth:
        raise ValueError("need equally many predicted and true coverages, at least one")
    diff_total = np.array(
        [p.total_target - t.total_target for p, t in zip(predicted, truth, strict=True)]
    )
    classes = list(truth[0].per_class)
    per_class = {
        c: float(
            np.mean(
                [
                    abs(p.per_class[c] - t.per_class[c])
                    for p, t in zip(predicted, truth, strict=True)
                ]
            )
        )
        * 100
        for c in classes
    }
    return CoverageErrors(
        total_mae_pp=float(np.abs(diff_total).mean() * 100),
        per_class_mae_pp=per_class,
        total_bias_pp=float(diff_total.mean() * 100),
        n_images=len(truth),
    )
