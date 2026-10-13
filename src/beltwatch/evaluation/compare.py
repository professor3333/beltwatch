"""Paired comparison of two evaluation reports on the same images.

Both models must have been scored by :mod:`beltwatch.evaluation.runner` on the
same split of the same split manifest. The runner saves per-image confusion
matrices (``confusions.npz``). This module resamples whole recording groups
**jointly** for both models, so the difference ``A - B`` gets its own interval
rather than two overlapping per-model intervals being eyeballed.

Usage (via ``scripts/compare.py``)::

    uv run python scripts/compare.py reports/segformer_b0-val reports/unet_resnet18-val

The result is written next to the reports as
``compare-<A>-vs-<B>-<split>.json``. Intervals over fewer than
:data:`~beltwatch.evaluation.runner.MIN_GROUPS_FOR_INTERVAL` groups are flagged
``too_few_groups``.
"""

import argparse
import json
import math
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

from beltwatch.evaluation.runner import MIN_GROUPS_FOR_INTERVAL, load_confusions
from beltwatch.evaluation.segmentation import (
    Confusion,
    foreground_macro_iou,
    metrics_from_confusion,
)
from beltwatch.evaluation.statistics import paired_bootstrap_difference
from beltwatch.labels import CLASS_NAMES, TARGET_CLASS_IDS

MATCHED_FIELDS = ("split", "split_manifest_id", "n_images", "limited_to")
"""Report fields that must agree before two reports can be compared."""


def _class_iou(name: str) -> Callable[[Confusion], float]:
    def metric(confusion: Confusion) -> float:
        return metrics_from_confusion(confusion).iou[name]

    return metric


def _clean(value: float) -> float | None:
    return None if math.isnan(value) else value


def _fmt(value: float | None) -> str:
    return "undefined" if value is None else f"{value:+.4f}"


def _load(report_dir: Path) -> tuple[dict[str, Any], Any]:
    report = json.loads((report_dir / "report.json").read_text())
    confusions_path = report_dir / "confusions.npz"
    if not confusions_path.exists():
        raise FileNotFoundError(
            f"{confusions_path} is missing: re-run scripts/evaluate.py to save per-image confusions"
        )
    saved = load_confusions(confusions_path)
    recomputed = foreground_macro_iou(saved.confusions.sum(axis=0))
    if not math.isclose(recomputed, report["segmentation"]["foreground_macro_iou"], abs_tol=1e-9):
        raise ValueError(f"{report_dir}: confusions.npz does not match report.json")
    return report, saved


def compare_reports(dir_a: Path, dir_b: Path, *, n_resamples: int = 2000) -> dict[str, Any]:
    """Paired-bootstrap intervals for ``A - B`` on foreground macro IoU and per-class IoU."""
    report_a, saved_a = _load(dir_a)
    report_b, saved_b = _load(dir_b)
    for field in MATCHED_FIELDS:
        if report_a.get(field) != report_b.get(field):
            raise ValueError(
                f"reports differ in {field}: {report_a.get(field)!r} != {report_b.get(field)!r}"
            )
    if saved_a.image_ids != saved_b.image_ids:
        raise ValueError("reports were computed on different images or in a different order")

    metrics: dict[str, Callable[[Confusion], float]] = {
        "foreground_macro_iou": foreground_macro_iou
    }
    metrics |= {f"iou_{CLASS_NAMES[c]}": _class_iou(CLASS_NAMES[c]) for c in TARGET_CLASS_IDS}

    differences: dict[str, dict[str, Any]] = {}
    for metric_name, fn in metrics.items():
        differences[metric_name] = {}
        for group_name, labels in saved_a.groups().items():
            with warnings.catch_warnings():
                # A class absent from both models and the ground truth is reported as undefined.
                warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
                interval = paired_bootstrap_difference(
                    saved_a.confusions, saved_b.confusions, labels, fn, n_resamples=n_resamples
                )
            fields = interval.as_dict()
            differences[metric_name][group_name] = {
                **{k: _clean(v) if isinstance(v, float) else v for k, v in fields.items()},
                "too_few_groups": interval.n_groups < MIN_GROUPS_FOR_INTERVAL,
            }

    def summary(report: dict[str, Any], directory: Path) -> dict[str, Any]:
        seg = report["segmentation"]
        return {
            "model": report["model"],
            "report_dir": str(directory),
            "temperature": report.get("temperature"),
            "foreground_macro_iou": seg["foreground_macro_iou"],
            "iou": {CLASS_NAMES[c]: seg["iou"][CLASS_NAMES[c]] for c in TARGET_CLASS_IDS},
        }

    return {
        "a": summary(report_a, dir_a),
        "b": summary(report_b, dir_b),
        **{field: report_a.get(field) for field in MATCHED_FIELDS},
        "difference_a_minus_b": differences,
        "note": "paired bootstrap over recording groups; both models resampled on the same groups",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two evaluation reports (A - B).")
    parser.add_argument("report_a", type=Path, help="report directory of model A")
    parser.add_argument("report_b", type=Path, help="report directory of model B")
    parser.add_argument("--out", type=Path, help="output JSON (default: next to the reports)")
    parser.add_argument("--n-resamples", type=int, default=2000)
    args = parser.parse_args(argv)

    result = compare_reports(args.report_a, args.report_b, n_resamples=args.n_resamples)
    out = args.out or args.report_a.parent / (
        f"compare-{result['a']['model']}-vs-{result['b']['model']}-{result['split']}.json"
    )
    out.write_text(json.dumps(result, indent=2) + "\n")
    for metric_name, by_group in result["difference_a_minus_b"].items():
        for group_name, interval in by_group.items():
            low, estimate, high = (_fmt(interval[k]) for k in ("low", "estimate", "high"))
            flag = " (too few groups)" if interval["too_few_groups"] else ""
            print(f"{metric_name} {group_name}: {estimate} [{low}, {high}]{flag}")
    print(f"wrote {out}")
    return 0
