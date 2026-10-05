import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from beltwatch.baselines.trivial import AllBackground
from beltwatch.data.config import load_data_config
from beltwatch.data.dataset import load_samples
from beltwatch.data.masks import load_source_remap
from beltwatch.evaluation import error_analysis as ea
from beltwatch.evaluation.benchmark import run_benchmark, synthetic_frames
from beltwatch.release.bundle import build_release
from beltwatch.review.policy import load_review_policy

REPO = Path(__file__).resolve().parents[2]


def val_samples(data_config: Path):  # type: ignore[no-untyped-def]
    data = load_data_config(data_config)
    samples = load_samples(
        data.paths.splits_dir / "zerowaste-f-splits.csv",
        data.paths.manifests_dir / "zerowaste-f-images.csv",
        data.paths.raw_dir,
        "train",
    )
    remap = load_source_remap(data.paths.manifests_dir / "zerowaste-f-validation.json")
    return samples, remap


def test_error_report_files_and_content(tiny_data_config: Path, tmp_path: Path) -> None:
    samples, remap = val_samples(tiny_data_config)

    summary = ea.build_report(
        AllBackground(), samples, remap, tmp_path / "report", split="train", k=3
    )

    out = tmp_path / "report"
    assert summary["segmentation"]["foreground_macro_iou"] == 0.0
    assert len(summary["galleries"]["False negatives: cardboard"]) == 3  # everything is missed
    assert summary["galleries"]["False positives: cardboard"] == []  # nothing is predicted
    assert set(summary["slices"]) == {"sequence", "true_coverage", "sharpness", "brightness"}
    page = (out / "report.html").read_text()
    assert "False negatives: metal" in page and "<script" not in page
    ET.fromstring((out / "confusion.svg").read_text())
    distinct = {image_id for ids in summary["galleries"].values() for image_id in ids}
    assert len(list((out / "img").glob("*.png"))) == len(distinct)  # each image rendered once
    assert json.loads((out / "error_analysis.json").read_text())["n_images"] == 4


def test_error_report_refuses_the_test_split() -> None:
    with pytest.raises(SystemExit):
        ea.main(["--model", "all-background", "--split", "test", "--out", "x"])


def test_benchmark_reports_latency_throughput_memory_and_targets(tmp_path: Path) -> None:
    policy = load_review_policy(REPO / "configs/review_policy.yaml")
    release = build_release(
        tmp_path, version="b1", kind="all-background", model_path=None, review_policy=policy
    )

    result = run_benchmark(
        release, synthetic_frames(2, size=(192, 108)), iterations=3, warmup=1, threads=1
    )

    assert result["iterations"] == 3 and result["image_size"] == [192, 108]
    assert result["full_pipeline"]["p95_seconds"] >= result["predict_only"]["p50_seconds"] >= 0
    assert result["images_per_second"] > 0 and result["peak_rss_mb"] > 0
    assert set(result["meets_targets"]) == {"p95_seconds", "images_per_second", "peak_rss_mb"}
    assert result["hardware"]["torch_threads"] == 1
