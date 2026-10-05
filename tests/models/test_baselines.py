from pathlib import Path

import numpy as np
import pytest

from beltwatch.baselines.features import FeatureConfig, pixel_features
from beltwatch.baselines.random_forest import (
    RandomForestConfig,
    RandomForestSegmenter,
    sample_pixels,
)
from beltwatch.baselines.trivial import AllBackground
from beltwatch.evaluation.segmentation import confusion_matrix, metrics_from_confusion
from beltwatch.labels import BACKGROUND, CARDBOARD, IGNORE_INDEX, METAL, NUM_CLASSES

SMALL = RandomForestConfig(
    long_side=48,
    pixels_per_class=150,
    n_estimators=20,
    max_depth=8,
    seed=0,
    features=FeatureConfig(sigma_max=4.0, num_sigma=3),
)


def scene(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Grey belt with an orange 'cardboard' patch and a bright 'metal' patch."""
    rng = np.random.default_rng(seed)
    rgb = rng.normal(70, 8, (54, 96, 3)).clip(0, 255).astype(np.uint8)
    mask = np.zeros((54, 96), dtype=np.uint8)
    y, x = rng.integers(5, 25), rng.integers(5, 50)
    rgb[y : y + 20, x : x + 30] = (200, 130, 50)
    mask[y : y + 20, x : x + 30] = CARDBOARD
    rgb[40:50, 80:92] = (235, 235, 240)
    mask[40:50, 80:92] = METAL
    return rgb, mask


def test_all_background_predicts_background_everywhere() -> None:
    proba = AllBackground().predict_proba(np.zeros((4, 6, 3), dtype=np.uint8))
    assert proba.shape == (NUM_CLASSES, 4, 6)
    assert np.all(proba.argmax(0) == BACKGROUND)


def test_pixel_features_shape_and_dtype() -> None:
    feats = pixel_features(
        np.zeros((8, 10, 3), dtype=np.uint8), FeatureConfig(sigma_max=4, num_sigma=3)
    )
    assert feats.shape[:2] == (8, 10)
    assert feats.dtype == np.float32
    assert feats.shape[2] == 3 * 3 * 4 + 3  # channels x sigmas x (intensity, edge, 2 texture) + Lab


def test_sample_pixels_is_class_stratified_and_skips_ignore() -> None:
    rgb, mask = scene(0)
    mask[:5] = IGNORE_INDEX

    x, y = sample_pixels(rgb, mask, SMALL, np.random.default_rng(0))

    counts = np.bincount(y, minlength=NUM_CLASSES)
    assert counts[BACKGROUND] == SMALL.pixels_per_class
    assert 0 < counts[METAL] <= SMALL.pixels_per_class
    assert IGNORE_INDEX not in y
    assert len(x) == len(y)


def test_random_forest_learns_a_separable_scene_and_round_trips(tmp_path: Path) -> None:
    xs, ys = zip(
        *(sample_pixels(*scene(s), SMALL, np.random.default_rng(s)) for s in range(6)), strict=True
    )
    rf = RandomForestSegmenter(SMALL).fit(np.concatenate(xs), np.concatenate(ys))

    rgb, mask = scene(99)
    proba = rf.predict_proba(rgb)

    assert proba.shape == (NUM_CLASSES, *mask.shape)
    assert np.allclose(proba.sum(0), 1.0, atol=1e-4)
    metrics = metrics_from_confusion(confusion_matrix(proba.argmax(0), mask))
    assert metrics.iou["cardboard"] > 0.7

    rf.save(tmp_path / "rf", {"note": "test"})
    loaded = RandomForestSegmenter.load(tmp_path / "rf")
    assert np.allclose(loaded.predict_proba(rgb), proba)


def test_random_forest_rejects_ignore_labels() -> None:
    with pytest.raises(ValueError, match="ignore"):
        RandomForestSegmenter(SMALL).fit(
            np.zeros((2, 3), np.float32), np.array([0, IGNORE_INDEX], np.uint8)
        )
