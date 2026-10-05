import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
SOURCE_CARDBOARD, SOURCE_METAL = 2, 3  # release category IDs


def _scene(i: int) -> tuple[np.ndarray, np.ndarray]:
    """Grey belt, an orange 'cardboard' block, and a white 'metal' block (source IDs)."""
    rgb = np.full((54, 96, 3), 70, dtype=np.uint8)
    mask = np.zeros((54, 96), dtype=np.uint8)
    x = 8 + (i * 13) % 40
    rgb[10:34, x : x + 30] = (210, 130, 40)
    mask[10:34, x : x + 30] = SOURCE_CARDBOARD
    rgb[38:50, 70:90] = (240, 240, 245)
    mask[38:50, 70:90] = SOURCE_METAL
    return rgb, mask


@pytest.fixture
def tiny_data_config(tmp_path: Path) -> Path:
    """A complete, tiny dataset on disk with the pipeline's outputs and a data config."""
    raw, manifests, splits_dir = tmp_path / "raw", tmp_path / "manifests", tmp_path / "splits"
    for d in (raw / "img", raw / "msk", manifests, splits_dir):
        d.mkdir(parents=True)
    images, splits = [], []
    for i, split in enumerate(["train"] * 4 + ["val"] * 2):
        rgb, mask = _scene(i)
        Image.fromarray(rgb).save(raw / f"img/{i}.png")
        Image.fromarray(mask).save(raw / f"msk/{i}.png")
        image_id = f"{split}/01_frame_{i * 1000:06d}"
        images.append(
            {"image_id": image_id, "image_path": f"img/{i}.png", "mask_path": f"msk/{i}.png"}
        )
        splits.append({"image_id": image_id, "split": split, "unseen_sequence": "False"})
    for path, rows in (
        (manifests / "zerowaste-f-images.csv", images),
        (splits_dir / "zerowaste-f-splits.csv", splits),
    ):
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    remap = {"0": 0, "1": 3, "2": 1, "3": 4, "4": 2}
    (manifests / "zerowaste-f-validation.json").write_text(
        json.dumps({"source_to_beltwatch": remap})
    )
    (splits_dir / "zerowaste-f-splits.json").write_text(json.dumps({"split_manifest_id": "f" * 64}))

    config = yaml.safe_load((REPO / "configs/data.yaml").read_text())
    config["paths"] = {
        "downloads_dir": str(tmp_path / "dl"),
        "raw_dir": str(raw),
        "manifests_dir": str(manifests),
        "splits_dir": str(splits_dir),
    }
    path = tmp_path / "data.yaml"
    path.write_text(yaml.safe_dump(config))
    return path
