"""Random-forest pixel classifier: the tuned classical baseline.

Training samples a class-stratified set of pixels from every training image
(up to ``pixels_per_class`` per class per image), so rare materials are
represented without discarding ordinary background. Stratification also makes
the forest's class frequencies unlike the real ones (background is most of the
training pixels but about a third of the sample). ``prior_power`` corrects this
at inference by multiplying each class's probability by ``(training pixel share
/ sampled share) ** prior_power``: 0 leaves the forest as fitted, 1 restores the
training-set frequencies. It is tuned on val without refitting
(:mod:`beltwatch.baselines.tune_random_forest`).

Inference computes features on the whole frame at a reduced working resolution
and maps probabilities back to the original size with the same restoration code
the neural models use.

Usage::

    uv run python -m beltwatch.baselines.random_forest --out models/rf
"""

import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import numpy.typing as npt
import torch
from sklearn.ensemble import RandomForestClassifier

from beltwatch import __version__
from beltwatch.baselines.features import FeatureConfig, pixel_features
from beltwatch.data.config import load_data_config
from beltwatch.data.dataset import Sample, load_samples
from beltwatch.data.masks import load_mask, load_source_remap
from beltwatch.inference.preprocessing import (
    PREPROCESSING_VERSION,
    PreprocessConfig,
    compute_geometry,
    load_rgb,
    resize_image,
    resize_mask,
    restore_probabilities,
)
from beltwatch.labels import IGNORE_INDEX, NUM_CLASSES

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RandomForestConfig:
    long_side: int = 384
    """Working resolution for features. Lower than the neural models' 768 for speed."""
    pixels_per_class: int = 200
    n_estimators: int = 100
    max_depth: int | None = 20
    min_samples_leaf: int = 5
    class_weight: str | None = None
    prior_power: float = 0.0
    """Strength of the class-frequency correction applied at inference (0 = none)."""
    seed: int = 0
    features: FeatureConfig = field(default_factory=FeatureConfig)


def _working_pair(
    rgb: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8] | None, long_side: int
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8] | None]:
    geometry = compute_geometry(rgb.shape[1], rgb.shape[0], PreprocessConfig(long_side, 1))
    small = resize_image(rgb, geometry.resized_size)
    return small, None if mask is None else resize_mask(mask, geometry.resized_size)


def sample_pixels(
    rgb: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.uint8],
    config: RandomForestConfig,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Class-stratified feature/label sample from one image (ignore pixels skipped)."""
    small, small_mask = _working_pair(rgb, mask, config.long_side)
    assert small_mask is not None
    features = pixel_features(small, config.features)
    flat_labels = small_mask.reshape(-1)
    chosen: list[npt.NDArray[np.intp]] = []
    for class_id in range(NUM_CLASSES):
        idx = np.flatnonzero(flat_labels == class_id)
        if idx.size:
            take = min(idx.size, config.pixels_per_class)
            chosen.append(rng.choice(idx, size=take, replace=False))
    picked = np.concatenate(chosen) if chosen else np.empty(0, dtype=np.intp)
    return features.reshape(-1, features.shape[-1])[picked], flat_labels[picked]


@dataclass(frozen=True)
class TrainingSet:
    x: npt.NDArray[np.float32]
    y: npt.NDArray[np.uint8]
    pixel_totals: npt.NDArray[np.int64]
    """Valid pixels per class over all training images, at the working resolution."""

    def prior_ratio(self) -> npt.NDArray[np.float64]:
        """Training pixel share / sampled share per class (1 for classes never sampled)."""
        sampled = np.bincount(self.y, minlength=NUM_CLASSES).astype(np.float64)
        totals = self.pixel_totals.astype(np.float64)
        ratio = np.ones(NUM_CLASSES)
        seen = sampled > 0
        ratio[seen] = (totals[seen] / totals.sum()) / (sampled[seen] / sampled.sum())
        return ratio


def build_training_set(
    samples: Sequence[Sample], remap: dict[int, int], config: RandomForestConfig
) -> TrainingSet:
    rng = np.random.default_rng(config.seed)
    xs, ys = [], []
    totals = np.zeros(NUM_CLASSES, dtype=np.int64)
    for i, sample in enumerate(samples):
        rgb, mask = load_rgb(sample.image_path), load_mask(sample.mask_path, remap)
        x, y = sample_pixels(rgb, mask, config, rng)
        xs.append(x)
        ys.append(y)
        _, small_mask = _working_pair(rgb, mask, config.long_side)
        assert small_mask is not None
        valid = small_mask[small_mask != IGNORE_INDEX]
        totals += np.bincount(valid, minlength=NUM_CLASSES)[:NUM_CLASSES]
        if (i + 1) % 100 == 0:
            log.info("sampled pixels from %d / %d images", i + 1, len(samples))
    return TrainingSet(np.concatenate(xs), np.concatenate(ys), totals)


class RandomForestSegmenter:
    name = "random-forest"

    def __init__(
        self,
        config: RandomForestConfig,
        model: RandomForestClassifier | None = None,
        prior_ratio: Sequence[float] | npt.NDArray[np.float64] | None = None,
    ):
        self.config = config
        self.prior_ratio = None if prior_ratio is None else np.asarray(prior_ratio, np.float64)
        self.model = model or RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            class_weight=config.class_weight,
            n_jobs=-1,
            random_state=config.seed,
        )

    def fit(self, x: npt.NDArray[np.float32], y: npt.NDArray[np.uint8]) -> "RandomForestSegmenter":
        if np.any(y == IGNORE_INDEX):
            raise ValueError("training labels must not contain the ignore index")
        self.model.fit(x, y)
        return self

    def with_prior_power(self, power: float) -> "RandomForestSegmenter":
        """The same fitted forest with a different class-frequency correction."""
        if power and self.prior_ratio is None:
            raise ValueError("prior_power needs the prior ratio recorded at training time")
        return RandomForestSegmenter(
            replace(self.config, prior_power=power), self.model, self.prior_ratio
        )

    def predict_proba(self, rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        small, _ = _working_pair(rgb, None, self.config.long_side)
        features = pixel_features(small, self.config.features)
        h, w = small.shape[:2]
        partial = self.model.predict_proba(features.reshape(-1, features.shape[-1]))
        proba = np.zeros((h * w, NUM_CLASSES), dtype=np.float32)
        proba[:, self.model.classes_.astype(int)] = partial
        if self.config.prior_power:
            if self.prior_ratio is None:
                raise ValueError("prior_power needs the prior ratio recorded at training time")
            proba *= (self.prior_ratio**self.config.prior_power).astype(np.float32)
            proba /= np.maximum(proba.sum(axis=1, keepdims=True), 1e-12)
        log_proba = torch.from_numpy(np.log(proba.T.reshape(NUM_CLASSES, h, w) + 1e-6))
        geometry = compute_geometry(
            rgb.shape[1], rgb.shape[0], PreprocessConfig(self.config.long_side, 1)
        )
        return restore_probabilities(log_proba, geometry).numpy().astype(np.float32)

    def save(self, directory: Path, metadata: dict[str, Any]) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, directory / "model.joblib", compress=3)
        prior = None if self.prior_ratio is None else self.prior_ratio.tolist()
        record = {"config": asdict(self.config), "prior_ratio": prior, **metadata}
        (directory / "metadata.json").write_text(json.dumps(record, indent=2) + "\n")

    @classmethod
    def load(cls, directory: Path) -> "RandomForestSegmenter":
        meta = json.loads((directory / "metadata.json").read_text())
        cfg = meta["config"]
        config = RandomForestConfig(**{**cfg, "features": FeatureConfig(**cfg["features"])})
        return cls(config, joblib.load(directory / "model.joblib"), meta.get("prior_ratio"))


def train_and_save(
    samples: Sequence[Sample],
    remap: dict[int, int],
    config: RandomForestConfig,
    out: Path,
    split_manifest_id: str,
) -> RandomForestSegmenter:
    """Fit on the training samples and save the forest with its lineage metadata."""
    start = time.perf_counter()
    data_set = build_training_set(samples, remap, config)
    log.info("training on %d pixels from %d images", len(data_set.y), len(samples))
    segmenter = RandomForestSegmenter(config, prior_ratio=data_set.prior_ratio())
    segmenter.fit(data_set.x, data_set.y)
    segmenter.save(
        out,
        {
            "beltwatch_version": __version__,
            "preprocessing_version": PREPROCESSING_VERSION,
            "split_manifest_id": split_manifest_id,
            "train_images": len(samples),
            "train_pixels": len(data_set.y),
            "pixel_class_counts": np.bincount(data_set.y, minlength=NUM_CLASSES).tolist(),
            "train_pixel_totals": data_set.pixel_totals.tolist(),
            "fit_seconds": round(time.perf_counter() - start, 1),
        },
    )
    log.info("saved %s", out)
    return segmenter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the random-forest baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--out", type=Path, default=Path("models/random-forest"))
    parser.add_argument("--pixels-per-class", type=int, default=RandomForestConfig.pixels_per_class)
    parser.add_argument("--n-estimators", type=int, default=RandomForestConfig.n_estimators)
    parser.add_argument("--max-depth", type=int, default=RandomForestConfig.max_depth)
    parser.add_argument("--long-side", type=int, default=RandomForestConfig.long_side)
    parser.add_argument("--prior-power", type=float, default=RandomForestConfig.prior_power)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = load_data_config(args.config)
    name = data.source.name
    splits_json = data.paths.splits_dir / f"{name}-splits.json"
    samples = load_samples(
        data.paths.splits_dir / f"{name}-splits.csv",
        data.paths.manifests_dir / f"{name}-images.csv",
        data.paths.raw_dir,
        "train",
    )
    remap = load_source_remap(data.paths.manifests_dir / f"{name}-validation.json")
    config = RandomForestConfig(
        long_side=args.long_side,
        pixels_per_class=args.pixels_per_class,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        prior_power=args.prior_power,
    )
    split_manifest_id = json.loads(splits_json.read_text())["split_manifest_id"]
    train_and_save(samples, remap, config, args.out, split_manifest_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
