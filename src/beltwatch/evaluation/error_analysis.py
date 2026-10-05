"""Automatically generated error-analysis report for a model on a split.

One pass over the split collects per-image statistics. The report then
contains:

* the pixel confusion matrix (counts and row-normalized), with an SVG heat map
* per-class galleries of the images with the most false-positive and
  false-negative pixels
* the images with the largest visible-coverage errors
* high-confidence mistakes (wrong pixels predicted with probability > 0.9)
* uncertain but correct predictions
* foreground macro IoU by slice: sequence, ground-truth coverage, sharpness,
  and brightness

Galleries show the original, the ground truth, and the prediction side by side.
Use **val** for iterative analysis; the locked test split is refused.

Usage::

    uv run python scripts/error_report.py --model unet --model-dir models/unet-resnet18-ce \\
        --split val --out reports/errors-unet
"""

import argparse
import html
import json
import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from beltwatch import __version__
from beltwatch.data.config import load_data_config
from beltwatch.data.dataset import Sample, load_samples
from beltwatch.data.masks import load_mask, load_source_remap
from beltwatch.data.validate import parse_frame_name
from beltwatch.evaluation.segmentation import (
    confusion_matrix,
    foreground_macro_iou,
    metrics_from_confusion,
)
from beltwatch.inference.postprocessing import (
    CLASS_COLORS,
    compute_coverage,
    normalized_entropy,
)
from beltwatch.inference.predictor import Predictor
from beltwatch.inference.preprocessing import load_rgb
from beltwatch.inference.quality import assess_quality
from beltwatch.labels import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, TARGET_CLASS_IDS
from beltwatch.review.policy import QualityPolicy

log = logging.getLogger(__name__)

CONFIDENT = 0.9
UNCERTAIN_ENTROPY = 0.5
THUMB_WIDTH = 360
QUALITY = QualityPolicy(
    min_sharpness=0.0, min_mean_brightness=0.0, max_mean_brightness=255.0, max_clipped_fraction=1.0
)


@dataclass
class ImageStats:
    image_id: str
    sequence_id: str | None
    confusion: npt.NDArray[np.int64]
    true_coverage: float
    pred_coverage: float
    confident_wrong: int
    uncertain_fraction: float
    error_fraction: float
    sharpness: float
    brightness: float


def collect(
    predictor: Predictor, samples: Sequence[Sample], remap: dict[int, int]
) -> list[ImageStats]:
    stats = []
    for i, sample in enumerate(samples):
        rgb = load_rgb(sample.image_path)
        truth = load_mask(sample.mask_path, remap)
        proba = predictor.predict_proba(rgb)
        pred = proba.argmax(axis=0).astype(np.uint8)
        valid = truth != IGNORE_INDEX
        wrong = (pred != truth) & valid
        region = np.ones(truth.shape, dtype=bool)
        quality = assess_quality(rgb, QUALITY)
        stats.append(
            ImageStats(
                image_id=sample.image_id,
                sequence_id=parse_frame_name(sample.image_id.split("/", 1)[-1]).sequence_id,
                confusion=confusion_matrix(pred, truth),
                true_coverage=compute_coverage(truth, region).total_target,
                pred_coverage=compute_coverage(pred, region, valid).total_target,
                confident_wrong=int((wrong & (proba.max(axis=0) > CONFIDENT)).sum()),
                uncertain_fraction=float(
                    (normalized_entropy(proba) > UNCERTAIN_ENTROPY)[valid].mean()
                ),
                error_fraction=float(wrong.sum() / max(valid.sum(), 1)),
                sharpness=quality.sharpness,
                brightness=quality.mean_brightness,
            )
        )
        if (i + 1) % 50 == 0:
            log.info("analysed %d / %d images", i + 1, len(samples))
    return stats


def _bins(values: Sequence[float], edges: Sequence[float], labels: Sequence[str]) -> list[str]:
    return [labels[int(np.searchsorted(edges, v, side="right"))] for v in values]


COVERAGE_BINS = ["<1%", "1-5%", "5-15%", ">=15%"]
SHARPNESS_BINS = ["low", "mid", "high"]
BRIGHTNESS_BINS = ["dark", "normal", "bright"]


def slice_table(stats: Sequence[ImageStats]) -> dict[str, dict[str, dict[str, float | int]]]:
    """Foreground macro IoU and image counts per slice value, in a meaningful order."""
    sharp_edges = list(np.quantile([s.sharpness for s in stats], [1 / 3, 2 / 3]))
    sequences = [str(s.sequence_id) for s in stats]
    slices: dict[str, tuple[list[str], list[str]]] = {
        "sequence": (sequences, sorted(set(sequences))),
        "true_coverage": (
            _bins([s.true_coverage for s in stats], [0.01, 0.05, 0.15], COVERAGE_BINS),
            COVERAGE_BINS,
        ),
        "sharpness": (
            _bins([s.sharpness for s in stats], sharp_edges, SHARPNESS_BINS),
            SHARPNESS_BINS,
        ),
        "brightness": (
            _bins([s.brightness for s in stats], [80, 140], BRIGHTNESS_BINS),
            BRIGHTNESS_BINS,
        ),
    }
    table: dict[str, dict[str, dict[str, float | int]]] = {}
    for name, (values, order) in slices.items():
        groups: dict[str, list[ImageStats]] = defaultdict(list)
        for s, v in zip(stats, values, strict=True):
            groups[v].append(s)
        table[name] = {
            value: {
                "n_images": len(groups[value]),
                "foreground_macro_iou": round(
                    foreground_macro_iou(np.stack([m.confusion for m in groups[value]]).sum(0)), 4
                ),
            }
            for value in order
            if groups[value]
        }
    return table


def top(
    stats: Sequence[ImageStats], key: Callable[[ImageStats], float], k: int
) -> list[ImageStats]:
    return sorted((s for s in stats if key(s) > 0), key=key, reverse=True)[:k]


def _colorize(labels: npt.NDArray[np.uint8], rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    out = (rgb.astype(np.float32) * 0.45).astype(np.uint8)
    for class_id, color in CLASS_COLORS.items():
        out[labels == class_id] = color
    out[labels == IGNORE_INDEX] = (60, 60, 60)
    return out


def write_triptych(sample: Sample, predictor: Predictor, remap: dict[int, int], path: Path) -> None:
    rgb = load_rgb(sample.image_path)
    truth = load_mask(sample.mask_path, remap)
    pred = predictor.predict_proba(rgb).argmax(axis=0).astype(np.uint8)
    panels = [rgb, _colorize(truth, rgb), _colorize(pred, rgb)]
    h = round(rgb.shape[0] * THUMB_WIDTH / rgb.shape[1])
    thumbs = [np.asarray(Image.fromarray(p).resize((THUMB_WIDTH, h))) for p in panels]
    gap = np.full((h, 6, 3), 255, dtype=np.uint8)
    Image.fromarray(np.concatenate([thumbs[0], gap, thumbs[1], gap, thumbs[2]], axis=1)).save(path)


def confusion_svg(confusion: npt.NDArray[np.int64]) -> str:
    names = [CLASS_NAMES[c] for c in range(NUM_CLASSES)]
    rows = confusion / np.maximum(confusion.sum(axis=1, keepdims=True), 1)
    cell, left, topm = 64, 110, 130
    size_w, size_h = left + cell * NUM_CLASSES + 10, topm + cell * NUM_CLASSES + 30
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size_w}" height="{size_h}" '
        'font-family="sans-serif" font-size="11">',
        f'<rect width="{size_w}" height="{size_h}" fill="white"/>',
        f'<text x="{left}" y="18" font-size="13">Row-normalized confusion (rows: truth)</text>',
    ]
    for j, name in enumerate(names):
        x, y = left + j * cell + cell / 2 - 6, topm - 6
        rotate = f"rotate(-45 {x} {y})"
        out.append(f'<text x="{x}" y="{y}" text-anchor="start" transform="{rotate}">{name}</text>')
    for i, name in enumerate(names):
        y = topm + i * cell
        out.append(f'<text x="{left - 6}" y="{y + cell / 2 + 4}" text-anchor="end">{name}</text>')
        for j in range(NUM_CLASSES):
            v = float(rows[i, j])
            shade = int(255 - 200 * v)
            fill = f"rgb({shade},{min(255, shade + 20)},255)"
            x = left + j * cell
            out.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" '
                f'fill="{fill}" stroke="#ccc"/>'
            )
            color = "white" if v > 0.6 else "black"
            out.append(
                f'<text x="{x + cell / 2}" y="{y + cell / 2 + 4}" text-anchor="middle" '
                f'fill="{color}">{v:.2f}</text>'
            )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _gallery_html(title: str, note: str, entries: Sequence[tuple[str, str]]) -> str:
    figures = [
        f'<figure><img src="{html.escape(src)}" loading="lazy" alt="">'
        f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        for src, caption in entries
    ]
    body = "".join(figures) or "<p>None.</p>"
    heading = f"<h2>{html.escape(title)}</h2><p>{html.escape(note)}</p>"
    return f"<section>{heading}<div class='grid'>{body}</div></section>"


PAGE_CSS = " ".join(
    [
        "body{font:14px system-ui,sans-serif;margin:24px;max-width:1200px}",
        "table{border-collapse:collapse}",
        "td,th{border:1px solid #ddd;padding:4px 8px;text-align:left}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(540px,1fr));gap:12px}",
        "figure{margin:0} img{width:100%;border:1px solid #ddd}",
        "figcaption{color:#555;font-size:12px}",
    ]
)
LEGEND = (
    "Colours: cardboard orange, soft plastic teal, rigid plastic magenta, metal yellow, "
    "grey = ignored. Verify any pattern against the images before acting on it."
)


def build_report(
    predictor: Predictor,
    samples: Sequence[Sample],
    remap: dict[int, int],
    out: Path,
    *,
    split: str,
    k: int = 6,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "img").mkdir(exist_ok=True)
    stats = collect(predictor, samples, remap)
    by_id = {s.image_id: s for s in samples}
    total = np.stack([s.confusion for s in stats]).sum(0)
    metrics = metrics_from_confusion(total)

    galleries: dict[str, list[tuple[ImageStats, str]]] = {}
    for c in TARGET_CLASS_IDS:
        name = CLASS_NAMES[c]

        def fp(s: ImageStats, c: int = c) -> float:
            return float(s.confusion[:, c].sum() - s.confusion[c, c])

        def fn(s: ImageStats, c: int = c) -> float:
            return float(s.confusion[c, :].sum() - s.confusion[c, c])

        galleries[f"False positives: {name}"] = [
            (s, f"{int(fp(s))} false-positive px") for s in top(stats, fp, k)
        ]
        galleries[f"False negatives: {name}"] = [
            (s, f"{int(fn(s))} missed px") for s in top(stats, fn, k)
        ]
    galleries["Largest coverage errors"] = [
        (s, f"predicted {s.pred_coverage:.1%} vs true {s.true_coverage:.1%}")
        for s in top(stats, lambda s: abs(s.pred_coverage - s.true_coverage), k)
    ]
    galleries["High-confidence mistakes"] = [
        (s, f"{s.confident_wrong} wrong px at p > {CONFIDENT}")
        for s in top(stats, lambda s: s.confident_wrong, k)
    ]
    galleries["Uncertain but correct"] = [
        (s, f"{s.uncertain_fraction:.1%} uncertain, {s.error_fraction:.1%} wrong")
        for s in top(
            stats, lambda s: s.uncertain_fraction * (1 - min(1.0, 10 * s.error_fraction)), k
        )
    ]

    written: dict[str, str] = {}
    sections = []
    for title, entries in galleries.items():
        rendered = []
        for s, caption in entries:
            if s.image_id not in written:
                fname = f"img/{s.image_id.replace('/', '__')}.png"
                write_triptych(by_id[s.image_id], predictor, remap, out / fname)
                written[s.image_id] = fname
            rendered.append((written[s.image_id], f"{s.image_id}: {caption}"))
        sections.append(_gallery_html(title, "original | ground truth | prediction", rendered))

    slices = slice_table(stats)
    (out / "confusion.svg").write_text(confusion_svg(total))
    summary = {
        "model": predictor.name,
        "split": split,
        "n_images": len(stats),
        "segmentation": metrics.as_dict(),
        "confusion_counts": total.tolist(),
        "slices": slices,
        "galleries": {t: [s.image_id for s, _ in e] for t, e in galleries.items()},
        "beltwatch_version": __version__,
    }
    (out / "error_analysis.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")

    slice_rows = []
    for slice_name, table in slices.items():
        slice_rows.append(f"<tr><th colspan=3>{html.escape(slice_name)}</th></tr>")
        for value, row in table.items():
            cells = (html.escape(value), row["n_images"], f"{row['foreground_macro_iou']:.3f}")
            slice_rows.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
    iou_rows = [
        f"<tr><td>{html.escape(n)}</td><td>{'' if v != v else f'{v:.3f}'}</td></tr>"
        for n, v in metrics.iou.items()
    ]
    title = f"Error analysis: {html.escape(predictor.name)} on {html.escape(split)}"
    page = "\n".join(
        [
            '<!doctype html><html lang="en"><head><meta charset="utf-8">',
            f"<title>{title}</title><style>{PAGE_CSS}</style></head><body>",
            f"<h1>{title} ({len(stats)} images)</h1>",
            f"<p>Foreground macro IoU {metrics.foreground_macro_iou:.3f}. {LEGEND}</p>",
            f"<h2>Per-class IoU</h2><table>{''.join(iou_rows)}</table>",
            '<h2>Confusion matrix</h2><img src="confusion.svg" alt="confusion matrix"'
            ' style="width:auto;border:0">',
            "<h2>Foreground macro IoU by slice</h2>",
            "<table><tr><th>slice</th><th>images</th><th>fg mIoU</th></tr>",
            "".join(slice_rows) + "</table>",
            "".join(sections),
            "</body></html>",
        ]
    )
    (out / "report.html").write_text(page)
    return summary


def main(argv: list[str] | None = None) -> int:
    from beltwatch.evaluation.runner import load_predictor

    parser = argparse.ArgumentParser(description="Generate an error-analysis report.")
    parser.add_argument(
        "--model", required=True, choices=["all-background", "random-forest", "unet", "segformer"]
    )
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--split", default="val", choices=["val", "calibration", "test"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--per-gallery", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.split == "test":
        parser.error("error analysis iterates on val; the locked test split is not inspected")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = load_data_config(args.config)
    name = data.source.name
    samples = load_samples(
        data.paths.splits_dir / f"{name}-splits.csv",
        data.paths.manifests_dir / f"{name}-images.csv",
        data.paths.raw_dir,
        args.split,
    )[: args.limit]
    remap = load_source_remap(data.paths.manifests_dir / f"{name}-validation.json")
    predictor = load_predictor(args.model, args.model_dir, args.temperature)
    summary = build_report(
        predictor, samples, remap, args.out, split=args.split, k=args.per_gallery
    )
    log.info(
        "foreground macro IoU %.4f; report at %s",
        summary["segmentation"]["foreground_macro_iou"],
        args.out / "report.html",
    )
    return 0
