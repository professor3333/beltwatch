"""Group bootstrap confidence intervals over recording groups.

Pixels and frames from the same recording are not independent, so resampling
happens over **groups** (e.g. recording sequences, or time blocks within them),
never over pixels. With few groups, intervals are wide; that is reported, not
hidden. A paired bootstrap resamples the same groups for two models so their
difference gets its own interval.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from beltwatch.evaluation.segmentation import Confusion

MetricFn = Callable[[Confusion], float]


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    n_groups: int
    n_resamples: int
    confidence: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "estimate": self.estimate,
            "low": self.low,
            "high": self.high,
            "n_groups": self.n_groups,
            "n_resamples": self.n_resamples,
            "confidence": self.confidence,
        }


def time_block_groups(
    sequence_ids: Sequence[str | None], frame_indices: Sequence[int | None], block_frames: int
) -> list[str]:
    """Group label ``"<sequence>:<block>"`` for frames binned into ``block_frames``-long blocks."""
    groups = []
    for seq, frame in zip(sequence_ids, frame_indices, strict=True):
        block = "na" if frame is None else str(frame // block_frames)
        groups.append(f"{seq}:{block}")
    return groups


def _group_sums(
    per_image: npt.NDArray[np.int64], groups: Sequence[str]
) -> tuple[npt.NDArray[np.int64], int]:
    names, inverse = np.unique(np.asarray(groups), return_inverse=True)
    sums = np.zeros((len(names), *per_image.shape[1:]), dtype=np.int64)
    np.add.at(sums, inverse, per_image)
    return sums, len(names)


def bootstrap_metric(
    per_image: npt.NDArray[np.int64],
    groups: Sequence[str],
    metric: MetricFn,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile CI for ``metric`` of the summed confusion, resampling whole groups."""
    if len(per_image) != len(groups):
        raise ValueError("one group label per image is required")
    sums, k = _group_sums(per_image, groups)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(n_resamples, k))
    values = np.array([metric(sums[d].sum(axis=0)) for d in draws])
    alpha = (1 - confidence) / 2
    return Interval(
        metric(sums.sum(axis=0)),
        float(np.nanquantile(values, alpha)),
        float(np.nanquantile(values, 1 - alpha)),
        k,
        n_resamples,
        confidence,
    )


def paired_bootstrap_difference(
    per_image_a: npt.NDArray[np.int64],
    per_image_b: npt.NDArray[np.int64],
    groups: Sequence[str],
    metric: MetricFn,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """CI for ``metric(A) - metric(B)`` on the same images, resampling groups jointly."""
    if per_image_a.shape != per_image_b.shape or len(per_image_a) != len(groups):
        raise ValueError("both models must be evaluated on the same images and groups")
    sums_a, k = _group_sums(per_image_a, groups)
    sums_b, _ = _group_sums(per_image_b, groups)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(n_resamples, k))
    values = np.array(
        [metric(sums_a[d].sum(axis=0)) - metric(sums_b[d].sum(axis=0)) for d in draws]
    )
    alpha = (1 - confidence) / 2
    return Interval(
        metric(sums_a.sum(axis=0)) - metric(sums_b.sum(axis=0)),
        float(np.nanquantile(values, alpha)),
        float(np.nanquantile(values, 1 - alpha)),
        k,
        n_resamples,
        confidence,
    )
