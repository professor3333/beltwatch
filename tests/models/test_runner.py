import csv
import json
from pathlib import Path

import numpy as np
import pytest

from beltwatch.baselines.trivial import AllBackground
from beltwatch.data.dataset import Sample
from beltwatch.evaluation import runner
from tests.models.helpers import REMAP, Oracle


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
    calibration = report["calibration"]
    assert calibration["overall"]["ece"] == pytest.approx(0.0, abs=1e-6)  # one-hot and correct
    assert calibration["foreground"]["n_pixels"] > 0


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
        results,
        model="oracle",
        split="val",
        slices={"seq01": {samples[0].image_id}},
        n_resamples=20,
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
