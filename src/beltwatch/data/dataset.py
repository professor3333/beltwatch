"""PyTorch datasets over the BeltWatch split manifest.

Training samples are augmented crops; evaluation samples are full frames
prepared exactly as the inference worker prepares uploads.
"""

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from beltwatch.data.augment import AugmentConfig, augment
from beltwatch.data.masks import load_mask
from beltwatch.inference.preprocessing import (
    PreprocessConfig,
    load_rgb,
    normalize,
    prepare_image,
    prepare_mask,
)


@dataclass(frozen=True)
class Sample:
    image_id: str
    image_path: Path
    mask_path: Path


def load_samples(splits_csv: Path, images_csv: Path, raw_dir: Path, split: str) -> list[Sample]:
    """Samples assigned to ``split`` (train/val/calibration/test) by the split stage."""
    with splits_csv.open(newline="", encoding="utf-8") as fh:
        wanted = {row["image_id"] for row in csv.DictReader(fh) if row["split"] == split}
    with images_csv.open(newline="", encoding="utf-8") as fh:
        rows = [row for row in csv.DictReader(fh) if row["image_id"] in wanted]
    if len(rows) != len(wanted):
        raise ValueError(f"{len(wanted) - len(rows)} {split} images missing from {images_csv}")
    return [
        Sample(row["image_id"], raw_dir / row["image_path"], raw_dir / row["mask_path"])
        for row in sorted(rows, key=lambda r: r["image_id"])
    ]


class SegmentationDataset(Dataset[dict[str, torch.Tensor | str]]):
    """Yields ``{"image": 3xHxW float32, "mask": HxW int64, "image_id": str}``.

    With ``augment_config`` set, samples are augmented training crops; otherwise
    they are full frames from :func:`prepare_image`. Augmentation randomness is
    drawn from torch's RNG, so ``torch.manual_seed`` and DataLoader worker
    seeding make it reproducible.
    """

    def __init__(
        self,
        samples: list[Sample],
        source_to_internal: dict[int, int],
        preprocess: PreprocessConfig,
        augment_config: AugmentConfig | None = None,
    ) -> None:
        self.samples = samples
        self.remap = source_to_internal
        self.preprocess = preprocess
        self.augment_config = augment_config

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        rgb = load_rgb(sample.image_path)
        mask = load_mask(sample.mask_path, self.remap)
        if mask.shape != rgb.shape[:2]:
            raise ValueError(f"{sample.image_id}: mask and image sizes differ")

        if self.augment_config is None:
            image, geometry = prepare_image(rgb, self.preprocess)
            target = prepare_mask(mask, geometry)
        else:
            rng = np.random.default_rng(int(torch.randint(0, 2**63 - 1, (1,)).item()))
            crop, target, valid = augment(
                rgb, mask, self.preprocess.long_side, self.augment_config, rng
            )
            image = normalize(crop, self.preprocess)
            image[:, ~valid] = 0.0  # padding looks like padding at inference time
        return {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(target.astype(np.int64)),
            "image_id": sample.image_id,
        }
