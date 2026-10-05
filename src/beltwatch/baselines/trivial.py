"""All-background predictor: the sanity check that exposes pixel accuracy.

It scores high pixel accuracy on background-heavy conveyor images while finding
no material at all (foreground macro IoU 0), which is why pixel accuracy is
never a headline metric.
"""

import numpy as np
import numpy.typing as npt

from beltwatch.labels import BACKGROUND, NUM_CLASSES


class AllBackground:
    name = "all-background"

    def predict_proba(self, rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        proba = np.zeros((NUM_CLASSES, *rgb.shape[:2]), dtype=np.float32)
        proba[BACKGROUND] = 1.0
        return proba
