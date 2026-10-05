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
