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

Several reports per side (for example one per training seed) compare the
seed-averaged metrics, with every resample applied to all of them::

    uv run python scripts/compare.py --a reports/x-seed1 reports/x-seed2 \
        --b reports/base-seed1 reports/base-seed2 --out reports/compare-x.json
"""

import argparse
import json
import math
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from beltwatch.evaluation.runner import MIN_GROUPS_FOR_INTERVAL, load_confusions
from beltwatch.evaluation.segmentation import (
    Confusion,
    foreground_macro_iou,
    metrics_from_confusion,
)
from beltwatch.evaluation.statistics import paired_bootstrap_mean_difference
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


def compare_reports(
    dir_a: Path | Sequence[Path], dir_b: Path | Sequence[Path], *, n_resamples: int = 2000
) -> dict[str, Any]:
    """Paired-bootstrap intervals for ``A - B`` on foreground macro IoU and per-class IoU.

    A side given as several report directories is the mean over them (for
    example over training seeds); a side given as one directory is that report.
    """
    dirs_a = [dir_a] if isinstance(dir_a, Path) else list(dir_a)
    dirs_b = [dir_b] if isinstance(dir_b, Path) else list(dir_b)
    if not dirs_a or not dirs_b:
        raise ValueError("each side needs at least one report")
    loaded_a = [_load(d) for d in dirs_a]
    loaded_b = [_load(d) for d in dirs_b]
    first_report, first_saved = loaded_a[0]
    for report, saved in [*loaded_a[1:], *loaded_b]:
        for field in MATCHED_FIELDS:
            if first_report.get(field) != report.get(field):
                raise ValueError(
                    f"reports differ in {field}: "
                    f"{first_report.get(field)!r} != {report.get(field)!r}"
                )
        if saved.image_ids != first_saved.image_ids:
            raise ValueError("reports were computed on different images or in a different order")

    metrics: dict[str, Callable[[Confusion], float]] = {
        "foreground_macro_iou": foreground_macro_iou
    }
    metrics |= {f"iou_{CLASS_NAMES[c]}": _class_iou(CLASS_NAMES[c]) for c in TARGET_CLASS_IDS}

    differences: dict[str, dict[str, Any]] = {}
    for metric_name, fn in metrics.items():
        differences[metric_name] = {}
        for group_name, labels in first_saved.groups().items():
            with warnings.catch_warnings():
                # A class absent from both models and the ground truth is reported as undefined.
                warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
                interval = paired_bootstrap_mean_difference(
                    [saved.confusions for _, saved in loaded_a],
                    [saved.confusions for _, saved in loaded_b],
                    labels,
                    fn,
                    n_resamples=n_resamples,
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
            "provenance": report.get("provenance"),
            "foreground_macro_iou": seg["foreground_macro_iou"],
            "iou": {CLASS_NAMES[c]: seg["iou"][CLASS_NAMES[c]] for c in TARGET_CLASS_IDS},
        }

    def side(loaded: list[tuple[dict[str, Any], Any]], dirs: list[Path]) -> dict[str, Any]:
        summaries = [summary(report, d) for (report, _), d in zip(loaded, dirs, strict=True)]
        if len(summaries) == 1:
            return summaries[0]
        names = ["foreground_macro_iou", *summaries[0]["iou"]]
        values = {
            name: [s[name] if name in s else s["iou"][name] for s in summaries] for name in names
        }

        def mean(name: str) -> float | None:  # undefined (null) in any report: undefined
            v = values[name]
            return None if any(x is None for x in v) else float(np.mean(v))

        return {
            "model": summaries[0]["model"],
            "mean_over_reports": len(summaries),
            "foreground_macro_iou": mean("foreground_macro_iou"),
            "iou": {name: mean(name) for name in names[1:]},
            "reports": summaries,
        }

    return {
        "a": side(loaded_a, dirs_a),
        "b": side(loaded_b, dirs_b),
        **{field: first_report.get(field) for field in MATCHED_FIELDS},
        "difference_a_minus_b": differences,
        "note": "paired bootstrap over recording groups; both models resampled on the same groups",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two evaluation reports (A - B).")
    parser.add_argument("report_a", type=Path, nargs="?", help="report directory of model A")
    parser.add_argument("report_b", type=Path, nargs="?", help="report directory of model B")
    parser.add_argument(
        "--a", type=Path, nargs="+", dest="set_a", help="several reports of A, averaged (seeds)"
    )
    parser.add_argument(
        "--b", type=Path, nargs="+", dest="set_b", help="several reports of B, averaged (seeds)"
    )
    parser.add_argument("--out", type=Path, help="output JSON (default: next to the reports)")
    parser.add_argument("--n-resamples", type=int, default=2000)
    args = parser.parse_args(argv)

    if args.set_a or args.set_b:
        if not (args.set_a and args.set_b) or args.report_a or args.report_b:
            parser.error("use --a and --b together, instead of the two positional reports")
        dirs_a, dirs_b = args.set_a, args.set_b
    elif args.report_a and args.report_b:
        dirs_a, dirs_b = [args.report_a], [args.report_b]
    else:
        parser.error("give two report directories, or --a and --b")
    result = compare_reports(dirs_a, dirs_b, n_resamples=args.n_resamples)
    out = args.out or dirs_a[0].parent / (
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
