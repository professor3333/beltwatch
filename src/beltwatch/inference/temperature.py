"""Calibration temperatures: one shared by every pixel, or one per predicted class.

A per-class temperature divides each pixel's logits by the temperature of the
class that pixel is predicted as. The divisor is positive and the same for all
of the pixel's logits, so the predicted class never changes: IoU and coverage
are exactly those of the uncalibrated model, and only confidence moves. This
lets the overconfident material predictions be softened without also
softening the far more numerous background pixels, which a single temperature
cannot do.
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from beltwatch.labels import NUM_CLASSES

Temperature = float | tuple[float, ...]


def check_temperature(value: float | Sequence[float]) -> Temperature:
    """A positive scalar, or one positive temperature per class in label order."""
    if isinstance(value, int | float):
        if value <= 0:
            raise ValueError("temperature must be positive")
        return float(value)
    temperatures = tuple(float(v) for v in value)
    if len(temperatures) != NUM_CLASSES:
        raise ValueError(f"expected {NUM_CLASSES} per-class temperatures, got {len(temperatures)}")
    if any(t <= 0 for t in temperatures):
        raise ValueError("temperatures must be positive")
    return temperatures


def parse_temperature(text: str) -> Temperature:
    """Command-line form: ``1.3`` or one comma-separated value per class."""
    parts = [p for p in text.split(",") if p.strip()]
    if len(parts) == 1 and "," not in text:
        return check_temperature(float(parts[0]))
    return check_temperature([float(p) for p in parts])


def is_identity(temperature: Temperature) -> bool:
    if isinstance(temperature, tuple):
        return all(t == 1.0 for t in temperature)
    return temperature == 1.0


def describe(temperature: Temperature) -> str:
    if isinstance(temperature, tuple):
        return "per class [" + ", ".join(f"{t:.3f}" for t in temperature) + "]"
    return f"{temperature:.3f}"


def scale_log_probs(
    z: npt.NDArray[np.floating], temperature: Temperature, axis: int
) -> npt.NDArray[np.floating]:
    """Divide logits (or log-probabilities) along the class ``axis`` by the temperature."""
    if not isinstance(temperature, tuple):
        return z / temperature
    per_pixel = np.asarray(temperature, dtype=z.dtype)[z.argmax(axis=axis)]
    out: npt.NDArray[np.floating] = z / np.expand_dims(per_pixel, axis)
    return out
