import math

import numpy as np
import pytest

from beltwatch.evaluation import coverage as cov
from beltwatch.evaluation import review_workload as rw
from beltwatch.evaluation import segmentation as seg
from beltwatch.evaluation import statistics as st
from beltwatch.inference.postprocessing import Coverage
from beltwatch.labels import BACKGROUND, CARDBOARD, IGNORE_INDEX, METAL, SOFT_PLASTIC


def test_confusion_ignores_ignore_pixels() -> None:
    target = np.array([[0, 1, IGNORE_INDEX]])
    pred = np.array([[0, 2, 4]])

    c = seg.confusion_matrix(pred, target)

    assert c.shape == (5, 5)
    assert c.sum() == 2
    assert c[0, 0] == 1
    assert c[1, 2] == 1


def test_confusion_validates_inputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        seg.confusion_matrix(np.zeros((2, 2)), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="out of range"):
        seg.confusion_matrix(np.array([7]), np.array([0]))


def test_metrics_hand_computed() -> None:
    # 10 cardboard pixels: 6 right, 4 predicted background; 2 background predicted cardboard.
    target = np.array([CARDBOARD] * 10 + [BACKGROUND] * 10)
    pred = np.array([CARDBOARD] * 6 + [BACKGROUND] * 4 + [CARDBOARD] * 2 + [BACKGROUND] * 8)

    m = seg.metrics_from_confusion(seg.confusion_matrix(pred, target))

    assert m.iou["cardboard"] == pytest.approx(6 / 12)
    assert m.dice["cardboard"] == pytest.approx(12 / 18)
    assert m.precision["cardboard"] == pytest.approx(6 / 8)
    assert m.recall["cardboard"] == pytest.approx(6 / 10)
    assert m.iou["background"] == pytest.approx(8 / 14)
    # soft/rigid/metal never appear: undefined, excluded from the macro averages.
    assert set(m.undefined_classes) == {"soft_plastic", "rigid_plastic", "metal"}
    assert m.foreground_macro_iou == pytest.approx(0.5)
    assert m.mean_iou == pytest.approx((0.5 + 8 / 14) / 2)


def test_all_background_prediction_scores_zero_foreground_iou() -> None:
    target = np.array([BACKGROUND] * 95 + [CARDBOARD, SOFT_PLASTIC, METAL, METAL, CARDBOARD])
    pred = np.zeros_like(target)

    m = seg.metrics_from_confusion(seg.confusion_matrix(pred, target))

    assert m.pixel_accuracy == pytest.approx(0.95)  # looks great...
    assert m.foreground_macro_iou == 0.0  # ...but finds nothing


def test_missed_class_counts_as_zero_not_undefined() -> None:
    m = seg.metrics_from_confusion(seg.confusion_matrix(np.array([0]), np.array([METAL])))
    assert m.iou["metal"] == 0.0
    assert "metal" not in m.undefined_classes


def test_aggregation_is_over_pixels_not_mean_of_images() -> None:
    small = seg.confusion_matrix(np.array([1]), np.array([1]))  # IoU 1 on 1 pixel
    big = seg.confusion_matrix(np.zeros(99, dtype=int), np.ones(99, dtype=int))  # IoU 0
    assert seg.metrics_from_confusion(small + big).iou["cardboard"] == pytest.approx(0.01)


def coverage(total: float, cardboard: float | None = None) -> Coverage:
    c = total if cardboard is None else cardboard
    per = {"cardboard": c, "soft_plastic": total - c, "rigid_plastic": 0.0, "metal": 0.0}
    return Coverage(per, total, 100)


def test_coverage_errors_in_percentage_points() -> None:
    errors = cov.coverage_errors([coverage(0.08), coverage(0.00)], [coverage(0.05), coverage(0.02)])

    assert errors.total_mae_pp == pytest.approx(2.5)  # (3 + 2) / 2 points
    assert errors.total_bias_pp == pytest.approx(0.5)  # (+3 - 2) / 2
    assert errors.per_class_mae_pp["cardboard"] == pytest.approx(2.5)
    assert errors.n_images == 2


def test_audit_positive_threshold_is_strict() -> None:
    assert cov.is_audit_positive(coverage(0.051))
    assert not cov.is_audit_positive(coverage(0.05))


def test_perfect_ranking_reaches_full_recall_at_prevalence() -> None:
    curve = rw.workload_curve([0.9, 0.8, 0.1, 0.0], [True, True, False, False])

    assert curve.prevalence == 0.5
    assert curve.recall_at(0.5) == 1.0
    assert curve.recall_at(0.25) == 0.5
    assert curve.area() > 0.7


def test_constant_scores_behave_like_random() -> None:
    labels = [i % 4 == 0 for i in range(400)]
    curve = rw.workload_curve([0.0] * 400, labels)
    assert abs(curve.area() - 0.5) < 0.06
    assert abs(curve.recall_at(0.5) - 0.5) < 0.1


def test_workload_curve_without_positives() -> None:
    curve = rw.workload_curve([0.1, 0.2], [False, False])
    assert curve.n_positive == 0
    assert math.isnan(curve.recall_at(0.5))


def test_workload_summary_keys() -> None:
    s = rw.workload_curve([1, 0], [True, False]).summary()
    assert set(s) == {"n_images", "n_positive", "prevalence", "area", "recall_at"}
    assert s["recall_at"] == {"0.10": 1.0, "0.20": 1.0, "0.30": 1.0, "0.50": 1.0}


def per_image_conf(pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.stack([seg.confusion_matrix(np.array([p]), np.array([t])) for p, t in pairs])


def test_time_block_groups() -> None:
    assert st.time_block_groups(["01", "01", "02", None], [10, 1500, 10, None], 1000) == [
        "01:0",
        "01:1",
        "02:0",
        "None:na",
    ]


def test_bootstrap_interval_contains_estimate_and_is_degenerate_for_identical_groups() -> None:
    conf = per_image_conf([(1, 1), (0, 1)] * 10)
    groups = [f"g{i // 2}" for i in range(20)]  # every group: one hit, one miss

    interval = st.bootstrap_metric(conf, groups, seg.foreground_macro_iou, n_resamples=200)

    assert interval.estimate == pytest.approx(0.5)
    assert interval.low == pytest.approx(0.5)
    assert interval.high == pytest.approx(0.5)
    assert interval.n_groups == 10


def test_bootstrap_interval_widens_with_heterogeneous_groups() -> None:
    conf = per_image_conf([(1, 1)] * 10 + [(0, 1)] * 10)
    groups = [f"g{i}" for i in range(20)]

    interval = st.bootstrap_metric(conf, groups, seg.foreground_macro_iou, n_resamples=500)

    assert interval.low < 0.5 < interval.high


def test_paired_difference_detects_consistent_improvement() -> None:
    better = per_image_conf([(1, 1)] * 16 + [(0, 1)] * 4)
    worse = per_image_conf([(1, 1)] * 8 + [(0, 1)] * 12)
    groups = [f"g{i % 10}" for i in range(20)]

    diff = st.paired_bootstrap_difference(
        better, worse, groups, seg.foreground_macro_iou, n_resamples=500
    )

    assert diff.estimate == pytest.approx(0.8 - 0.4)
    assert diff.low > 0


def test_bootstrap_requires_matching_groups() -> None:
    with pytest.raises(ValueError, match="group"):
        st.bootstrap_metric(per_image_conf([(1, 1)]), ["a", "b"], seg.foreground_macro_iou)
