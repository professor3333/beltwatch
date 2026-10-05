import csv
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from beltwatch.baselines.trivial import AllBackground
from beltwatch.data.dataset import Sample
from beltwatch.evaluation import runner
from beltwatch.labels import CARDBOARD, NUM_CLASSES

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


@pytest.fixture
def images(tmp_path: Path) -> tuple[list[Sample], dict[int, np.ndarray]]:
    samples, masks = [], {}
    for i in range(6):
        internal = np.zeros((20, 30), dtype=np.uint8)
        internal[: 2 + 3 * i, :10] = CARDBOARD  # coverage grows with i
        source = np.vectorize(INVERSE.get)(internal).astype(np.uint8)
        rgb = np.full((20, 30, 3), i, dtype=np.uint8)
        Image.fromarray(rgb).save(tmp_path / f"{i}.png")
        Image.fromarray(source).save(tmp_path / f"{i}_m.png")
        image_id = f"val/0{i % 3 + 1}_frame_{i * 600:06d}"
        samples.append(Sample(image_id, tmp_path / f"{i}.png", tmp_path / f"{i}_m.png"))
        masks[i] = internal
    return samples, masks


def test_oracle_scores_perfectly(images: tuple[list[Sample], dict[int, np.ndarray]]) -> None:
    samples, masks = images
    results = runner.evaluate_images(Oracle(masks), samples, REMAP)

    report = runner.build_report(results, model="oracle", split="val", n_resamples=50)

    assert report["segmentation"]["foreground_macro_iou"] == pytest.approx(1.0)
    assert report["coverage"]["total_mae_pp"] == pytest.approx(0.0)
    workload = report["review_workload"]
    assert workload["n_positive"] == 5  # coverage above 5% for 5 of 6 images
    assert workload["recall_at"]["0.50"] == pytest.approx(3 / 5)  # best possible at 50%
    ci = report["confidence_intervals"]["foreground_macro_iou"]["groups_by_sequence"]
    assert ci["n_groups"] == 3
    assert ci["too_few_groups"] is True  # 3 sequences cannot support a meaningful interval


def test_all_background_scores_zero_with_high_pixel_accuracy(
    images: tuple[list[Sample], dict[int, np.ndarray]],
) -> None:
    samples, _ = images
    results = runner.evaluate_images(AllBackground(), samples, REMAP)

    report = runner.build_report(results, model="all-background", split="val", n_resamples=50)

    seg = report["segmentation"]
    assert seg["foreground_macro_iou"] == 0.0
    assert seg["pixel_accuracy"] > 0.8
    assert seg["iou"]["metal"] is None  # undefined classes serialize as null
    assert report["coverage"]["total_bias_pp"] < 0


def test_slices_and_per_image_csv(
    images: tuple[list[Sample], dict[int, np.ndarray]], tmp_path: Path
) -> None:
    samples, masks = images
    results = runner.evaluate_images(Oracle(masks), samples, REMAP)

    report = runner.build_report(
        results, model="oracle", split="val", slices={"seq01": {samples[0].image_id}}, n_resamples=20
    )
    runner.write_per_image(results, tmp_path / "per_image.csv")

    assert report["slices"]["seq01"]["n_images"] == 1
    with (tmp_path / "per_image.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 6
    assert rows[0]["sequence_id"] == "01"
    json.dumps(report)  # must be JSON-serializable (no NaN)


def test_test_split_requires_declared_release() -> None:
    with pytest.raises(SystemExit):
        runner.main(["--model", "all-background", "--split", "test"])
