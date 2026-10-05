"""Recall of audit-positive images versus the fraction of images reviewed.

Images are reviewed in descending score order. Every reviewed image counts
toward workload, whatever its label. Random ordering retrieves positives in
proportion to the fraction reviewed, so its expected curve is the diagonal; the
achievable gain depends on prevalence, which is always reported.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

DEFAULT_WORKLOADS = (0.1, 0.2, 0.3, 0.5)


@dataclass(frozen=True)
class WorkloadCurve:
    fraction_reviewed: npt.NDArray[np.float64]
    recall: npt.NDArray[np.float64]
    prevalence: float
    n_images: int
    n_positive: int

    def recall_at(self, workload: float) -> float:
        """Recall after reviewing ``workload`` (0 to 1) of the images."""
        k = int(np.ceil(workload * self.n_images))
        return float(self.recall[k]) if self.n_positive else float("nan")

    def area(self) -> float:
        """Area under the recall-vs-workload curve (random ordering is about 0.5)."""
        return float(np.trapezoid(self.recall, self.fraction_reviewed))

    def summary(self, workloads: Sequence[float] = DEFAULT_WORKLOADS) -> dict[str, object]:
        return {
            "n_images": self.n_images,
            "n_positive": self.n_positive,
            "prevalence": self.prevalence,
            "area": self.area(),
            "recall_at": {f"{w:.2f}": self.recall_at(w) for w in workloads},
        }


def workload_curve(
    scores: Sequence[float], positives: Sequence[bool], *, seed: int = 0
) -> WorkloadCurve:
    """Curve for reviewing images in descending ``scores`` order.

    Ties are broken randomly (with ``seed``) so a constant score behaves like
    random ordering rather than favouring the input order.
    """
    if len(scores) != len(positives) or not scores:
        raise ValueError("need equally many scores and labels, at least one")
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(positives, dtype=bool)
    tiebreak = np.random.default_rng(seed).random(len(s))
    order = np.lexsort((tiebreak, -s))
    hits = np.concatenate([[0], np.cumsum(y[order])])
    n_pos = int(y.sum())
    recall = hits / n_pos if n_pos else np.zeros_like(hits, dtype=np.float64)
    fraction = np.arange(len(s) + 1, dtype=np.float64) / len(s)
    return WorkloadCurve(fraction, recall.astype(np.float64), n_pos / len(s), len(s), n_pos)
