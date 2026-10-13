import json
from pathlib import Path

import numpy as np
import pytest

from beltwatch.baselines import random_forest
from beltwatch.baselines.features import FeatureConfig, pixel_features
from beltwatch.baselines.random_forest import (
    RandomForestConfig,
    RandomForestSegmenter,
    TrainingSet,
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


def fitted_forest() -> RandomForestSegmenter:
    xs, ys = zip(
        *(sample_pixels(*scene(s), SMALL, np.random.default_rng(s)) for s in range(6)), strict=True
    )
    y = np.concatenate(ys)
    # Pretend background is far more common in the training set than in the sample.
    totals = np.bincount(y, minlength=NUM_CLASSES) * np.array([50, 1, 1, 1, 1])
    data_set = TrainingSet(np.concatenate(xs), y, totals)
    return RandomForestSegmenter(SMALL, prior_ratio=data_set.prior_ratio()).fit(data_set.x, y)


def test_prior_ratio_is_training_share_over_sampled_share() -> None:
    y = np.array([0, 0, 1, 1], dtype=np.uint8)  # sampled: half background, half cardboard
    totals = np.array([90, 10, 0, 0, 0])  # training set: 90% background

    ratio = TrainingSet(np.zeros((4, 1), np.float32), y, totals).prior_ratio()

    assert ratio == pytest.approx([1.8, 0.2, 1.0, 1.0, 1.0])  # unsampled classes stay at 1


def test_prior_power_shifts_predictions_towards_frequent_classes(tmp_path: Path) -> None:
    rf = fitted_forest()
    rgb, _ = scene(99)

    base = rf.predict_proba(rgb)
    corrected = rf.with_prior_power(1.0).predict_proba(rgb)

    assert np.allclose(rf.with_prior_power(0.0).predict_proba(rgb), base)
    assert np.allclose(corrected.sum(0), 1.0, atol=1e-4)
    assert corrected[BACKGROUND].mean() > base[BACKGROUND].mean()
    rf.with_prior_power(1.0).save(tmp_path / "rf", {})
    loaded = RandomForestSegmenter.load(tmp_path / "rf")
    assert loaded.config.prior_power == 1.0
    assert np.allclose(loaded.predict_proba(rgb), corrected)


def test_prior_power_needs_a_recorded_ratio(tmp_path: Path) -> None:
    rf = fitted_forest()
    rf.save(tmp_path / "old", {})
    meta = json.loads((tmp_path / "old/metadata.json").read_text())
    del meta["prior_ratio"], meta["config"]["prior_power"]  # a forest saved before the option
    (tmp_path / "old/metadata.json").write_text(json.dumps(meta))

    old = RandomForestSegmenter.load(tmp_path / "old")

    assert old.config.prior_power == 0.0
    old.predict_proba(scene(1)[0])
    with pytest.raises(ValueError, match="prior ratio"):
        old.with_prior_power(0.5)


def test_random_forest_cli_records_the_prior(tiny_data_config: Path, tmp_path: Path) -> None:
    args = ["--config", str(tiny_data_config), "--out", str(tmp_path / "rf"), "--long-side", "48"]
    small = ["--pixels-per-class", "40", "--n-estimators", "5", "--max-depth", "6"]

    assert random_forest.main([*args, *small, "--prior-power", "0.5"]) == 0

    meta = json.loads((tmp_path / "rf/metadata.json").read_text())
    assert meta["config"]["prior_power"] == 0.5
    assert len(meta["prior_ratio"]) == NUM_CLASSES
    assert sum(meta["train_pixel_totals"]) > meta["train_pixels"]
