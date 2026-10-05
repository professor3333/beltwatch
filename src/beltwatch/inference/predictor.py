"""The interface every BeltWatch model implements, baselines included.

A predictor takes an original-resolution RGB image and returns per-class
probabilities at the same resolution, so evaluation and serving treat
classical and neural models identically.
"""

from typing import Protocol

import numpy as np
import numpy.typing as npt


class Predictor(Protocol):
    name: str

    def predict_proba(self, rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """``H x W x 3`` uint8 RGB → ``C x H x W`` float32 probabilities at the same size."""
        ...


class TemperatureScaled:
    """Apply a fitted temperature to any predictor's probabilities.

    ``softmax(log p / T)`` equals ``softmax(logits / T)``, so this is exact for
    neural models and also calibrates the random forest.
    """

    def __init__(self, predictor: Predictor, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.predictor = predictor
        self.temperature = temperature
        self.name = predictor.name

    def predict_proba(self, rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        proba = self.predictor.predict_proba(rgb)
        if self.temperature == 1.0:
            return proba
        z = np.log(np.clip(proba, 1e-8, 1.0)) / self.temperature
        z -= z.max(axis=0, keepdims=True)
        e = np.exp(z)
        return (e / e.sum(axis=0, keepdims=True)).astype(np.float32)
