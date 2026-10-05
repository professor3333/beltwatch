import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from beltwatch import labels
from beltwatch.data import validate as v
from beltwatch.data.config import (
    DataConfig,
    DuplicatesConfig,
    LayoutConfig,
    PathsConfig,
    SourceConfig,
    load_data_config,
)

# Same category order as the real release: deliberately different from ours.
CATEGORIES = [
    {"id": 1, "name": "rigid_plastic"},
    {"id": 2, "name": "cardboard"},
    {"id": 3, "name": "metal"},
    {"id": 4, "name": "soft_plastic"},
]
LAYOUT = LayoutConfig(
    root_subdir="splits",
    splits=("train", "val"),
    images_subdir="data",
    masks_subdir="sem_seg",
    annotations_file="labels.json",
)
W, H = 16, 8
SQUARE = [[1.0, 1.0, 5.0, 1.0, 5.0, 5.0, 1.0, 5.0]]


class FakeDataset:
    """Builds a tiny dataset with the release's directory layout."""

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = raw_dir
        self.coco: dict[str, dict[str, Any]] = {
            split: {"categories": CATEGORIES, "images": [], "annotations": []}
            for split in LAYOUT.splits
        }

    def split_dir(self, split: str) -> Path:
        return self.raw_dir / LAYOUT.root_subdir / split

    def add(
        self,
        split: str,
        name: str,
        mask: np.ndarray | None = None,
        anns: list[dict[str, Any]] | None = None,
        *,
        write_image: bool = True,
        in_json: bool = True,
    ) -> None:
        mask = np.zeros((H, W), dtype=np.uint8) if mask is None else mask
        image_dir = self.split_dir(split) / LAYOUT.images_subdir
        mask_dir = self.split_dir(split) / LAYOUT.masks_subdir
        image_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        if write_image:
            Image.fromarray(np.full((H, W, 3), 120, dtype=np.uint8)).save(image_dir / name)
            Image.fromarray(mask).save(mask_dir / name)
        if in_json:
            coco = self.coco[split]
            image_id = len(coco["images"])
            coco["images"].append({"id": image_id, "file_name": name, "width": W, "height": H})
            for ann in anns or []:
                coco["annotations"].append(
                    {"id": len(coco["annotations"]), "image_id": image_id, "iscrowd": 0, **ann}
                )

    def write(self) -> None:
        for split, coco in self.coco.items():
            self.split_dir(split).mkdir(parents=True, exist_ok=True)
            (self.split_dir(split) / LAYOUT.annotations_file).write_text(json.dumps(coco))


def mask_with(value: int) -> np.ndarray:
    mask = np.zeros((H, W), dtype=np.uint8)
    mask[1:5, 1:5] = value
    return mask


@pytest.fixture
def ds(tmp_path: Path) -> FakeDataset:
    dataset = FakeDataset(tmp_path / "raw")
    dataset.add(
        "train", "01_frame_000010.PNG", mask_with(2), [{"category_id": 2, "segmentation": SQUARE}]
    )
    dataset.add(
        "train", "01_frame_000020.PNG", mask_with(1), [{"category_id": 1, "segmentation": SQUARE}]
    )
    dataset.add("val", "02_frame_000100.PNG")
    return dataset


def run(ds: FakeDataset) -> v.ValidationResult:
    ds.write()
    return v.validate_dataset(ds.raw_dir, LAYOUT)


def quarantined(result: v.ValidationResult) -> dict[str, str]:
    return {row["image_id"]: row["errors"] for row in result.quarantine}


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("01_frame_001160.PNG", v.FrameName("01", 1160, None)),
        ("09_frame_002000-2.PNG", v.FrameName("09", 2000, 2)),
        ("odd.png", v.FrameName(None, None, None)),
    ],
)
def test_parse_frame_name(name: str, expected: v.FrameName) -> None:
    assert v.parse_frame_name(name) == expected


def test_clean_dataset(ds: FakeDataset) -> None:
    result = run(ds)

    assert result.quarantine == []
    assert [row["image_id"] for row in result.manifest] == [
        "train/01_frame_000010",
        "train/01_frame_000020",
        "val/02_frame_000100",
    ]
    assert result.summary["source_to_beltwatch"] == {
        "0": labels.BACKGROUND,
        "1": labels.RIGID_PLASTIC,
        "2": labels.CARDBOARD,
        "3": labels.METAL,
        "4": labels.SOFT_PLASTIC,
    }
    assert result.summary["splits"]["train"]["valid"] == 2
    assert result.summary["sequences_by_split"] == {"01": {"train": 2}, "02": {"val": 1}}


def test_pixel_counts_use_beltwatch_classes(ds: FakeDataset) -> None:
    row = run(ds).manifest[0]  # mask value 2 = source "cardboard"

    assert row["px_cardboard"] == 16
    assert row["px_rigid_plastic"] == 0
    assert row["px_background"] == W * H - 16
    assert row["sequence_id"] == "01"
    assert row["frame_index"] == 10
    assert len(row["image_sha256"]) == 64


def test_empty_image_is_kept_with_warning(ds: FakeDataset) -> None:
    row = next(r for r in run(ds).manifest if r["split"] == "val")

    assert "no_annotations" in row["warnings"]
    assert row["px_background"] == W * H


def test_corrupt_image_is_quarantined(ds: FakeDataset) -> None:
    ds.add("train", "03_frame_000001.PNG")
    (ds.split_dir("train") / "data" / "03_frame_000001.PNG").write_bytes(b"not a png")

    errors = quarantined(run(ds))

    assert errors["train/03_frame_000001"].startswith("image_decode_failed")


def test_missing_mask_is_quarantined(ds: FakeDataset) -> None:
    ds.add("train", "03_frame_000001.PNG")
    (ds.split_dir("train") / "sem_seg" / "03_frame_000001.PNG").unlink()

    assert quarantined(run(ds))["train/03_frame_000001"] == "missing_mask"


def test_mask_size_mismatch_is_quarantined(ds: FakeDataset) -> None:
    ds.add("train", "03_frame_000001.PNG")
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(
        ds.split_dir("train") / "sem_seg" / "03_frame_000001.PNG"
    )

    assert quarantined(run(ds))["train/03_frame_000001"].startswith("mask_size_mismatch")


def test_unexpected_mask_value_is_quarantined(ds: FakeDataset) -> None:
    ds.add("train", "03_frame_000001.PNG", mask_with(7))

    assert quarantined(run(ds))["train/03_frame_000001"] == "unexpected_mask_values([7])"


def test_image_size_must_match_annotations(ds: FakeDataset) -> None:
    ds.add("train", "03_frame_000001.PNG")
    ds.coco["train"]["images"][-1]["width"] = W + 1

    assert quarantined(run(ds))["train/03_frame_000001"].startswith("image_size_mismatch")


def test_unannotated_file_and_missing_file_are_quarantined(ds: FakeDataset) -> None:
    ds.add("train", "03_frame_000001.PNG", in_json=False)
    ds.add("train", "03_frame_000002.PNG", write_image=False)

    errors = quarantined(run(ds))

    assert errors["train/03_frame_000001"] == "not_in_annotations"
    assert errors["train/03_frame_000002"] == "missing_image"


def test_unknown_category_is_quarantined(ds: FakeDataset) -> None:
    ds.add("train", "03_frame_000001.PNG", anns=[{"category_id": 9, "segmentation": SQUARE}])

    assert "unknown_category" in quarantined(run(ds))["train/03_frame_000001"]


def test_polygon_problems_are_warnings(ds: FakeDataset) -> None:
    ds.add(
        "train",
        "03_frame_000001.PNG",
        mask_with(2),
        [
            {"category_id": 2, "segmentation": [[1.0, 1.0, 2.0, 2.0]]},
            {"category_id": 2, "segmentation": [[0.0, 0.0, 40.0, 0.0, 40.0, 4.0]]},
            {"category_id": 2, "segmentation": [[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]]},
        ],
    )

    result = run(ds)

    assert "train/03_frame_000001" not in quarantined(result)
    assert result.summary["warning_counts"] == {
        "degenerate_polygon": 1,
        "no_annotations": 1,
        "polygon_out_of_bounds": 1,
        "zero_area_polygon": 1,
    }


def test_mask_and_annotation_category_disagreement_is_warned(ds: FakeDataset) -> None:
    ds.add(
        "train", "03_frame_000001.PNG", mask_with(3), [{"category_id": 2, "segmentation": SQUARE}]
    )

    row = next(r for r in run(ds).manifest if r["image_id"] == "train/03_frame_000001")

    assert "category_missing_from_mask(['cardboard'])" in row["warnings"]
    assert "category_missing_from_annotations(['metal'])" in row["warnings"]


def test_categories_must_match_across_splits(ds: FakeDataset) -> None:
    ds.coco["val"]["categories"] = [*CATEGORIES[:3], {"id": 4, "name": "glass"}]

    with pytest.raises(v.DatasetValidationError, match="categories differ"):
        run(ds)


def test_dangling_annotation_raises(ds: FakeDataset) -> None:
    ds.coco["train"]["annotations"].append({"id": 99, "image_id": 1234, "category_id": 1})

    with pytest.raises(v.DatasetValidationError, match="missing image"):
        run(ds)


def test_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(v.DatasetValidationError, match="download stage"):
        v.validate_dataset(tmp_path, LAYOUT)


def test_parallel_matches_serial(ds: FakeDataset) -> None:
    ds.write()
    serial = v.validate_dataset(ds.raw_dir, LAYOUT, workers=1)
    parallel = v.validate_dataset(ds.raw_dir, LAYOUT, workers=2)

    assert parallel.manifest == serial.manifest


def test_write_outputs(ds: FakeDataset, tmp_path: Path) -> None:
    ds.add("train", "03_frame_000001.PNG", mask_with(7))
    result = run(ds)
    repo = load_data_config(Path("configs/data.yaml"))
    config = DataConfig(
        source=SourceConfig(**{**repo.source.model_dump(), "name": "fixture"}),
        layout=LAYOUT,
        duplicates=DuplicatesConfig(hash_size=8, phash_max_hamming=6),
        paths=PathsConfig(
            downloads_dir=tmp_path / "dl", raw_dir=ds.raw_dir, manifests_dir=tmp_path / "m"
        ),
    )

    paths = v.write_outputs(result, config)

    with paths["manifest"].open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3
    assert list(rows[0]) == list(v.MANIFEST_FIELDS)
    with paths["quarantine"].open() as fh:
        assert [r["image_id"] for r in csv.DictReader(fh)] == ["train/03_frame_000001"]
    summary = json.loads(paths["summary"].read_text())
    assert summary["source"]["name"] == "fixture"
    assert summary["total_quarantined"] == 1
