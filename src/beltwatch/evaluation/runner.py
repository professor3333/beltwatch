"""Score any :class:`~beltwatch.inference.predictor.Predictor` on a split.

Every model, classical or neural, goes through this one function. For each
image it restores the prediction to the original resolution and computes a
confusion matrix and the visible coverage of the prediction and the ground truth
(full frame as the inspection region). It then aggregates segmentation metrics,
coverage error, the review-workload curve (ordered by predicted target
coverage), and group-bootstrap confidence intervals.

Usage (via ``scripts/evaluate.py``)::

    uv run python scripts/evaluate.py --model random-forest \\
        --model-dir models/random-forest --split val

The locked test split requires ``--declared-release``: test results are produced
only for releases declared in advance, never during development.
"""

import argparse
import csv
import json
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from beltwatch import __version__
from beltwatch.baselines.random_forest import RandomForestSegmenter
from beltwatch.baselines.trivial import AllBackground
from beltwatch.data.config import load_data_config
from beltwatch.data.dataset import Sample, load_samples
from beltwatch.data.masks import load_mask, load_source_remap
from beltwatch.data.validate import parse_frame_name
from beltwatch.evaluation.calibration import calibration_summary, sample_pixels
from beltwatch.evaluation.coverage import coverage_errors, is_audit_positive
from beltwatch.evaluation.review_workload import workload_curve
from beltwatch.evaluation.segmentation import (
    confusion_matrix,
    foreground_macro_iou,
    metrics_from_confusion,
)
from beltwatch.evaluation.statistics import bootstrap_metric, time_block_groups
from beltwatch.inference.postprocessing import Coverage, compute_coverage
from beltwatch.inference.predictor import Predictor, TemperatureScaled
from beltwatch.inference.preprocessing import PREPROCESSING_VERSION, load_rgb
from beltwatch.labels import CLASS_NAMES, TARGET_CLASS_IDS

log = logging.getLogger(__name__)

TIME_BLOCK_FRAMES = 1000
MIN_GROUPS_FOR_INTERVAL = 5
"""Below this many groups an interval is flagged as not meaningful."""


@dataclass(frozen=True)
class ImageResult:
    image_id: str
    sequence_id: str | None
    frame_index: int | None
    confusion: np.ndarray
    predicted: Coverage
    truth: Coverage
    seconds: float
    calibration_log_probs: np.ndarray
    calibration_labels: np.ndarray


def evaluate_images(
    predictor: Predictor,
    samples: Sequence[Sample],
    remap: dict[int, int],
    *,
    calibration_pixels_per_image: int = 2000,
    seed: int = 0,
) -> list[ImageResult]:
    rng = np.random.default_rng(seed)
    results = []
    for i, sample in enumerate(samples):
        rgb = load_rgb(sample.image_path)
        truth_mask = load_mask(sample.mask_path, remap)
        start = time.perf_counter()
        proba = predictor.predict_proba(rgb)
        seconds = time.perf_counter() - start
        if proba.shape[1:] != truth_mask.shape:
            raise ValueError(f"{predictor.name} returned {proba.shape} for {truth_mask.shape}")
        pred = proba.argmax(axis=0).astype(np.uint8)
        region = np.ones(truth_mask.shape, dtype=bool)
        frame = parse_frame_name(sample.image_id.split("/", 1)[-1])
        log_probs, labels = sample_pixels(proba, truth_mask, calibration_pixels_per_image, rng)
        results.append(
            ImageResult(
                image_id=sample.image_id,
                sequence_id=frame.sequence_id,
                frame_index=frame.frame_index,
                confusion=confusion_matrix(pred, truth_mask),
                predicted=compute_coverage(pred, region, truth_mask != 255),
                truth=compute_coverage(truth_mask, region),
                seconds=seconds,
                calibration_log_probs=log_probs,
                calibration_labels=labels,
            )
        )
        if (i + 1) % 50 == 0:
            log.info("evaluated %d / %d images", i + 1, len(samples))
    return results


def _nan_to_none(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(v) for v in value]
    return value


def _calibration(results: Sequence[ImageResult]) -> dict[str, Any]:
    """ECE of the probabilities as the predictor outputs them (its temperature applied)."""
    log_probs = np.concatenate([r.calibration_log_probs for r in results])
    labels = np.concatenate([r.calibration_labels for r in results])
    if not len(labels):
        return {}
    summary = calibration_summary(log_probs, labels, 1.0)
    return {k: summary[k] for k in ("nll", "overall", "foreground")}


def _interval(per_image: np.ndarray, groups: list[str], n_resamples: int) -> dict[str, Any]:
    interval = bootstrap_metric(per_image, groups, foreground_macro_iou, n_resamples=n_resamples)
    return {
        **interval.as_dict(),
        "too_few_groups": interval.n_groups < MIN_GROUPS_FOR_INTERVAL,
    }


def build_report(
    results: Sequence[ImageResult],
    *,
    model: str,
    split: str,
    slices: Mapping[str, set[str]] | None = None,
    n_resamples: int = 2000,
) -> dict[str, Any]:
    if not results:
        raise ValueError("no images were evaluated")
    per_image = np.stack([r.confusion for r in results])
    metrics = metrics_from_confusion(per_image.sum(axis=0))
    curve = workload_curve(
        [r.predicted.total_target for r in results], [is_audit_positive(r.truth) for r in results]
    )
    sequences = [r.sequence_id for r in results]
    frames = [r.frame_index for r in results]
    by_sequence = [str(s) for s in sequences]
    by_block = time_block_groups(sequences, frames, TIME_BLOCK_FRAMES)

    slice_metrics: dict[str, Any] = {}
    for name, ids in (slices or {}).items():
        chosen = [r for r in results if r.image_id in ids]
        if chosen:
            conf = np.stack([r.confusion for r in chosen]).sum(axis=0)
            slice_metrics[name] = {
                "n_images": len(chosen),
                "segmentation": metrics_from_confusion(conf).as_dict(),
            }

    latency = np.array([r.seconds for r in results])
    report = {
        "model": model,
        "split": split,
        "n_images": len(results),
        "beltwatch_version": __version__,
        "preprocessing_version": PREPROCESSING_VERSION,
        "segmentation": metrics.as_dict(),
        "coverage": coverage_errors(
            [r.predicted for r in results], [r.truth for r in results]
        ).as_dict(),
        "review_workload": curve.summary(),
        "confidence_intervals": {
            "foreground_macro_iou": {
                "groups_by_sequence": _interval(per_image, by_sequence, n_resamples),
                f"groups_by_{TIME_BLOCK_FRAMES}_frame_block": _interval(
                    per_image, by_block, n_resamples
                ),
            }
        },
        "calibration": _calibration(results),
        "slices": slice_metrics,
        "prediction_seconds": {
            "p50": float(np.percentile(latency, 50)),
            "p95": float(np.percentile(latency, 95)),
            "note": "predict_proba only, on the evaluation machine; not a serving benchmark",
        },
    }
    return dict(_nan_to_none(report))


def write_per_image(results: Sequence[ImageResult], path: Path) -> None:
    targets = [CLASS_NAMES[c] for c in TARGET_CLASS_IDS]
    fields = [
        "image_id",
        "sequence_id",
        "frame_index",
        "audit_positive",
        "true_total",
        "pred_total",
    ]
    fields += [f"true_{c}" for c in targets] + [f"pred_{c}" for c in targets]
    fields += ["foreground_macro_iou"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for r in results:
            iou = foreground_macro_iou(r.confusion)
            writer.writerow(
                {
                    "image_id": r.image_id,
                    "sequence_id": r.sequence_id,
                    "frame_index": r.frame_index,
                    "audit_positive": is_audit_positive(r.truth),
                    "true_total": round(r.truth.total_target, 6),
                    "pred_total": round(r.predicted.total_target, 6),
                    **{f"true_{c}": round(r.truth.per_class[c], 6) for c in targets},
                    **{f"pred_{c}": round(r.predicted.per_class[c], 6) for c in targets},
                    "foreground_macro_iou": "" if math.isnan(iou) else round(iou, 6),
                }
            )


def load_predictor(kind: str, model_dir: Path | None, temperature: float = 1.0) -> Predictor:
    if kind == "all-background":
        return AllBackground()
    if model_dir is None:
        raise ValueError(f"--model-dir is required for {kind}")
    if kind == "random-forest":
        return TemperatureScaled(RandomForestSegmenter.load(model_dir), temperature)
    if kind in ("unet", "segformer"):
        from beltwatch.inference.neural import NeuralPredictor

        return NeuralPredictor.from_checkpoint(model_dir / "best.pt", temperature=temperature)
    raise ValueError(f"unknown model {kind!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate a BeltWatch model on one split.")
    parser.add_argument(
        "--model", required=True, choices=["all-background", "random-forest", "unet", "segformer"]
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--split", required=True, choices=["val", "calibration", "test"])
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--limit", type=int, help="evaluate only the first N images (smoke runs)")
    parser.add_argument("--temperature", type=float, default=1.0, help="calibration temperature")
    parser.add_argument(
        "--declared-release",
        help="release name; required for the locked test split and recorded in the report",
    )
    args = parser.parse_args(argv)

    if args.split == "test" and not args.declared_release:
        parser.error("the test split is locked: pass --declared-release <name> for a release")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = load_data_config(args.config)
    name = data.source.name
    splits_csv = data.paths.splits_dir / f"{name}-splits.csv"
    samples = load_samples(
        splits_csv, data.paths.manifests_dir / f"{name}-images.csv", data.paths.raw_dir, args.split
    )
    if args.limit:
        samples = samples[: args.limit]
    with splits_csv.open(newline="", encoding="utf-8") as fh:
        unseen = {r["image_id"] for r in csv.DictReader(fh) if r["unseen_sequence"] == "True"}
    remap = load_source_remap(data.paths.manifests_dir / f"{name}-validation.json")
    predictor = load_predictor(args.model, args.model_dir, args.temperature)

    results = evaluate_images(predictor, samples, remap)
    report = build_report(
        results, model=predictor.name, split=args.split, slices={"unseen_sequence": unseen}
    )
    splits_summary = json.loads((data.paths.splits_dir / f"{name}-splits.json").read_text())
    report["split_manifest_id"] = splits_summary["split_manifest_id"]
    report["declared_release"] = args.declared_release
    report["limited_to"] = args.limit
    report["temperature"] = args.temperature

    out = args.out_dir / f"{predictor.name}-{args.split}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_per_image(results, out / "per_image.csv")
    seg = report["segmentation"]
    log.info(
        "%s on %s (%d images): foreground macro IoU %s, coverage MAE %.2f pp -> %s",
        predictor.name,
        args.split,
        len(results),
        seg["foreground_macro_iou"],
        report["coverage"]["total_mae_pp"],
        out,
    )
    return 0
