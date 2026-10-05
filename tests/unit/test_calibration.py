import xml.etree.ElementTree as ET

import numpy as np
import pytest

from beltwatch.evaluation import calibration as cal
from beltwatch.inference.predictor import TemperatureScaled


def synthetic(n: int, true_temperature: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Logits whose labels are drawn from softmax(logits / true_temperature)."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(0, 3, (n, 5)).astype(np.float32)
    probs = cal.softmax_with_temperature(logits, true_temperature)
    labels = np.array([rng.choice(5, p=p) for p in probs])
    return logits, labels


@pytest.mark.parametrize("true_t", [0.5, 1.0, 2.5])
def test_fit_recovers_the_generating_temperature(true_t: float) -> None:
    logits, labels = synthetic(20000, true_t)
    assert cal.fit_temperature(logits, labels) == pytest.approx(true_t, rel=0.1)


def test_temperature_reduces_nll_and_ece_for_overconfident_scores() -> None:
    logits, labels = synthetic(20000, 3.0)  # the raw scores are overconfident
    t = cal.fit_temperature(logits, labels)
    before = cal.calibration_summary(logits, labels, 1.0)
    after = cal.calibration_summary(logits, labels, t)
    assert t > 1.5
    assert after["nll"] < before["nll"]
    assert after["overall"]["ece"] < before["overall"]["ece"]


def test_ece_hand_computed() -> None:
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.6, 0.4]])
    labels = np.array([0, 1, 0, 0])  # 90% bin: 50% right; 60% bin: 100% right
    bins = cal.expected_calibration_error(probs, labels, n_bins=10)
    assert bins.ece == pytest.approx((2 * 0.4 + 2 * 0.4) / 4)
    assert sum(bins.count) == 4


def test_perfectly_calibrated_scores_have_low_ece() -> None:
    logits, labels = synthetic(30000, 1.0)
    probs = cal.softmax_with_temperature(logits, 1.0)
    assert cal.expected_calibration_error(probs, labels).ece < 0.02


def test_foreground_mask_uses_truth_or_prediction() -> None:
    probs = np.array([[0.9, 0.1, 0, 0, 0], [0.2, 0.8, 0, 0, 0], [0.9, 0.1, 0, 0, 0]])
    labels = np.array([0, 0, 3])
    assert cal.foreground_mask(probs, labels).tolist() == [False, True, True]


def test_sample_pixels_skips_ignore() -> None:
    proba = np.full((5, 4, 4), 0.2, dtype=np.float32)
    truth = np.full((4, 4), 255, dtype=np.uint8)
    truth[0, :2] = 1
    log_probs, labels = cal.sample_pixels(proba, truth, 10, np.random.default_rng(0))
    assert labels.tolist() == [1, 1]
    assert log_probs.shape == (2, 5)


def test_temperature_scaled_predictor_softens_and_preserves_argmax() -> None:
    class Sharp:
        name = "sharp"

        def predict_proba(self, rgb: np.ndarray) -> np.ndarray:
            p = np.full((5, 2, 2), 0.01, dtype=np.float32)
            p[1] = 0.96
            return p

    raw = Sharp().predict_proba(np.zeros((2, 2, 3), np.uint8))
    soft = TemperatureScaled(Sharp(), 3.0).predict_proba(np.zeros((2, 2, 3), np.uint8))
    assert soft[1, 0, 0] < raw[1, 0, 0]
    assert np.array_equal(soft.argmax(0), raw.argmax(0))
    assert np.allclose(soft.sum(0), 1.0, atol=1e-5)
    with pytest.raises(ValueError):
        TemperatureScaled(Sharp(), 0.0)


def test_reliability_svg_is_valid_xml() -> None:
    logits, labels = synthetic(2000, 2.0)
    before = cal.calibration_summary(logits, labels, 1.0)["overall"]
    after = cal.calibration_summary(logits, labels, 2.0)["overall"]
    root = ET.fromstring(cal.reliability_svg(before, after, "test"))
    assert root.tag.endswith("svg")


def test_cli_refuses_fitting_outside_the_calibration_split() -> None:
    with pytest.raises(SystemExit):
        cal.main(["--model", "unet", "--model-dir", "x", "--split", "val", "--out", "y"])
