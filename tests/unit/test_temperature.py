import numpy as np
import pytest
import torch

from beltwatch.evaluation import calibration as cal
from beltwatch.inference import preprocessing as pp
from beltwatch.inference.predictor import TemperatureScaled
from beltwatch.inference.temperature import (
    check_temperature,
    is_identity,
    parse_temperature,
    scale_log_probs,
)

PER_CLASS = (1.0, 1.5, 2.0, 2.5, 3.0)


def test_parse_and_check() -> None:
    assert parse_temperature("1.5") == 1.5
    assert parse_temperature("1,1.5,2,2.5,3") == PER_CLASS
    assert check_temperature([1, 1, 1, 1, 1]) == (1.0,) * 5 and is_identity((1.0,) * 5)
    for bad in ("0", "-1", "1,2", "1,2,3,4,0"):
        with pytest.raises(ValueError):
            parse_temperature(bad)


def test_per_class_scaling_keeps_every_prediction_and_uses_the_predicted_class() -> None:
    z = np.random.default_rng(0).normal(0, 3, (1000, 5))

    scaled = scale_log_probs(z, PER_CLASS, axis=1)

    predicted = z.argmax(axis=1)
    assert np.array_equal(scaled.argmax(axis=1), predicted)
    np.testing.assert_allclose(scaled, z / np.asarray(PER_CLASS)[predicted][:, None])
    np.testing.assert_allclose(scale_log_probs(z, (2.0,) * 5, axis=1), z / 2.0)


def test_restore_and_wrapper_agree_on_per_class_temperatures() -> None:
    geometry = pp.compute_geometry(64, 32, pp.PreprocessConfig(long_side=64))
    logits = torch.from_numpy(np.random.default_rng(1).normal(0, 3, (5, 32, 64)).astype("f4"))

    restored = pp.restore_probabilities(logits, geometry, PER_CLASS)
    plain = pp.restore_probabilities(logits, geometry)

    class Fixed:
        name = "fixed"

        def predict_proba(self, rgb: np.ndarray) -> np.ndarray:
            return plain.numpy()

    wrapped = TemperatureScaled(Fixed(), PER_CLASS).predict_proba(np.zeros((32, 64, 3), np.uint8))
    assert torch.equal(restored.argmax(0), plain.argmax(0))
    np.testing.assert_allclose(wrapped, restored.numpy(), atol=1e-4)
    uniform = pp.restore_probabilities(logits, geometry, (2.0,) * 5)
    torch.testing.assert_close(uniform, pp.restore_probabilities(logits, geometry, 2.0))


def overconfident_by_class(
    n: int, temperatures: tuple[float, ...], seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Labels drawn from softmax(z); the scores are z sharpened by the predicted class's T."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1.5, (n, 5))
    probs = cal.softmax_with_temperature(z.astype(np.float32), 1.0)
    labels = (probs.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
    scores = z * np.asarray(temperatures)[z.argmax(axis=1)][:, None]
    return scores.astype(np.float32), labels


def test_fit_class_temperatures_recovers_each_class() -> None:
    scores, labels = overconfident_by_class(60000, PER_CLASS)

    fitted = cal.fit_class_temperatures(scores, labels)

    assert fitted == pytest.approx(PER_CLASS, rel=0.15)
    one = cal.fit_temperature(scores, labels)
    assert cal.nll(scores, labels, fitted) < cal.nll(scores, labels, one)


def test_a_class_never_predicted_keeps_temperature_one() -> None:
    scores = np.tile(np.array([[3.0, 0.0, 0.0, 0.0, 0.0]], np.float32), (50, 1))
    labels = np.zeros(50, dtype=np.int64)
    labels[:10] = 1

    fitted = cal.fit_class_temperatures(scores, labels)

    assert fitted[1:] == (1.0, 1.0, 1.0, 1.0) and fitted[0] != 1.0
