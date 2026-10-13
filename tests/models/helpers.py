import numpy as np

from beltwatch.labels import NUM_CLASSES

REMAP = {0: 0, 1: 3, 2: 1, 3: 4, 4: 2}  # release order -> BeltWatch IDs
INVERSE = {v: k for k, v in REMAP.items()}


class Oracle:
    """Predicts the ground truth it is handed (to test the runner, not a model)."""

    name = "oracle"

    def __init__(self, masks: dict[int, np.ndarray]) -> None:
        self.masks = masks

    def predict_proba(self, rgb: np.ndarray) -> np.ndarray:
        mask = self.masks[int(rgb[0, 0, 0])]
        return np.eye(NUM_CLASSES, dtype=np.float32)[mask].transpose(2, 0, 1)
