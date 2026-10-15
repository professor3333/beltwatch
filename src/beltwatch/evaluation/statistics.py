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
    return paired_bootstrap_mean_difference(
        [per_image_a],
        [per_image_b],
        groups,
        metric,
        n_resamples=n_resamples,
        confidence=confidence,
        seed=seed,
    )


def paired_bootstrap_mean_difference(
    runs_a: Sequence[npt.NDArray[np.int64]],
    runs_b: Sequence[npt.NDArray[np.int64]],
    groups: Sequence[str],
    metric: MetricFn,
    *,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """CI for the mean of ``metric`` over runs A minus the mean over runs B.

    Each run (for example one training seed) is a per-image confusion array on
    the same images. Every resample draws one set of groups and applies it to
    all runs on both sides, so the interval covers the variation between images
    for the seed-averaged difference. With one run per side this is
    :func:`paired_bootstrap_difference`.
    """
    runs = [*runs_a, *runs_b]
    if not runs_a or not runs_b:
        raise ValueError("each side needs at least one run")
    if any(r.shape != runs[0].shape for r in runs) or len(runs[0]) != len(groups):
        raise ValueError("every run must be evaluated on the same images and groups")
    sums_a = [_group_sums(r, groups)[0] for r in runs_a]
    sums_b = [_group_sums(r, groups)[0] for r in runs_b]
    k = len(sums_a[0])

    def mean_difference(pick: npt.NDArray[np.intp] | slice) -> float:
        mean_a = np.mean([metric(s[pick].sum(axis=0)) for s in sums_a])
        mean_b = np.mean([metric(s[pick].sum(axis=0)) for s in sums_b])
        return float(mean_a - mean_b)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(n_resamples, k))
    values = np.array([mean_difference(d) for d in draws])
    alpha = (1 - confidence) / 2
    return Interval(
        mean_difference(slice(None)),
        float(np.nanquantile(values, alpha)),
        float(np.nanquantile(values, 1 - alpha)),
        k,
        n_resamples,
        confidence,
    )
