import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from beltwatch.data.augment import AugmentConfig
from beltwatch.data.dataset import SegmentationDataset, load_samples
from beltwatch.data.masks import load_mask, load_source_remap
from beltwatch.inference.preprocessing import PreprocessConfig
from beltwatch.labels import CARDBOARD, IGNORE_INDEX, RIGID_PLASTIC

# Source IDs as in the release: 1 rigid_plastic, 2 cardboard, 3 metal, 4 soft_plastic.
REMAP = {0: 0, 1: 3, 2: 1, 3: 4, 4: 2}
PREPROCESS = PreprocessConfig(long_side=96, pad_multiple=32)


@pytest.fixture
def tiny(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    (raw / "img").mkdir(parents=True)
    (raw / "msk").mkdir()
    images, splits = [], []
    for i, (split, value) in enumerate([("train", 2), ("train", 1), ("val", 2), ("test", 1)]):
        image_id = f"{split}/x{i}"
        Image.fromarray(np.full((54, 96, 3), 40 * i, dtype=np.uint8)).save(raw / f"img/{i}.png")
        mask = np.zeros((54, 96), dtype=np.uint8)
        mask[10:30, 10:50] = value
        Image.fromarray(mask).save(raw / f"msk/{i}.png")
        images.append(
            {"image_id": image_id, "image_path": f"img/{i}.png", "mask_path": f"msk/{i}.png"}
        )
        splits.append({"image_id": image_id, "split": split})
    for name, rows in (("images.csv", images), ("splits.csv", splits)):
        with (tmp_path / name).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    (tmp_path / "validation.json").write_text(
        json.dumps({"source_to_beltwatch": {str(k): v for k, v in REMAP.items()}})
    )
    return tmp_path


def samples(tiny: Path, split: str):  # type: ignore[no-untyped-def]
    return load_samples(tiny / "splits.csv", tiny / "images.csv", tiny / "raw", split)


def test_load_samples_filters_by_split(tiny: Path) -> None:
    assert [s.image_id for s in samples(tiny, "train")] == ["train/x0", "train/x1"]
    assert [s.image_id for s in samples(tiny, "test")] == ["test/x3"]


def test_load_mask_converts_source_ids(tiny: Path) -> None:
    remap = load_source_remap(tiny / "validation.json")

    mask = load_mask(tiny / "raw/msk/0.png", remap)

    assert remap == REMAP
    assert set(np.unique(mask)) == {0, CARDBOARD}
    assert set(np.unique(load_mask(tiny / "raw/msk/1.png", remap))) == {0, RIGID_PLASTIC}


def test_eval_dataset_returns_full_padded_frames(tiny: Path) -> None:
    ds = SegmentationDataset(samples(tiny, "val"), REMAP, PREPROCESS)

    item = ds[0]

    assert item["image_id"] == "val/x2"
    image, mask = item["image"], item["mask"]
    assert isinstance(image, torch.Tensor) and isinstance(mask, torch.Tensor)
    assert image.shape == (3, 64, 96) and image.dtype == torch.float32
    assert mask.shape == (64, 96) and mask.dtype == torch.int64
    assert set(mask.unique().tolist()) == {0, CARDBOARD, IGNORE_INDEX}


def test_train_dataset_returns_augmented_crops(tiny: Path) -> None:
    ds = SegmentationDataset(samples(tiny, "train"), REMAP, PREPROCESS, AugmentConfig(crop_size=48))
    torch.manual_seed(0)

    items = [ds[i % 2] for i in range(6)]

    for item in items:
        assert item["image"].shape == (3, 48, 48)
        assert item["mask"].shape == (48, 48)
        assert set(item["mask"].unique().tolist()) <= {0, 1, 3, IGNORE_INDEX}


def test_train_augmentation_is_reproducible_with_torch_seed(tiny: Path) -> None:
    ds = SegmentationDataset(samples(tiny, "train"), REMAP, PREPROCESS, AugmentConfig(crop_size=48))

    torch.manual_seed(7)
    first = ds[0]["image"]
    torch.manual_seed(7)
    second = ds[0]["image"]

    assert torch.equal(first, second)


def test_works_with_dataloader(tiny: Path) -> None:
    ds = SegmentationDataset(samples(tiny, "train"), REMAP, PREPROCESS, AugmentConfig(crop_size=48))

    batch = next(iter(torch.utils.data.DataLoader(ds, batch_size=2)))

    assert batch["image"].shape == (2, 3, 48, 48)
    assert batch["mask"].shape == (2, 48, 48)
