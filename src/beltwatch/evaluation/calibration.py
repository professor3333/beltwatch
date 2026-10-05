"""Confidence calibration: temperature scaling and expected calibration error.

Temperature scaling divides logits by a single scalar T fitted to minimize the
negative log-likelihood on the **calibration split only**. It works from
probabilities, since ``softmax(log p / T)`` equals ``softmax(logits / T)``, so
the same code calibrates the random forest and the neural models.

Expected calibration error (ECE) is reported overall and on foreground pixels
(ground truth or prediction is one of the four materials). Background
dominates the pixel count and can hide poor confidence on the classes that
matter.

Usage::

    uv run python -m beltwatch.evaluation.calibration --model unet \\
        --model-dir models/unet-resnet18-ce --out reports/calibration-unet

The fitted temperature goes into a release with
``python -m beltwatch.release.bundle ... --calibration reports/calibration-unet/calibration.json``.
"""

import argparse
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from beltwatch import __version__
from beltwatch.data.config import load_data_config
from beltwatch.data.dataset import Sample, load_samples
from beltwatch.data.masks import load_mask, load_source_remap
from beltwatch.inference.predictor import Predictor
from beltwatch.inference.preprocessing import load_rgb
from beltwatch.labels import IGNORE_INDEX, TARGET_CLASS_IDS

log = logging.getLogger(__name__)

DEFAULT_BINS = 15


def softmax_with_temperature(
    log_probs: npt.NDArray[np.float32], temperature: float
) -> npt.NDArray[np.float64]:
    z = log_probs.astype(np.float64) / temperature
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out: npt.NDArray[np.float64] = e / e.sum(axis=1, keepdims=True)
    return out


def nll(
    log_probs: npt.NDArray[np.float32], labels: npt.NDArray[np.integer], temperature: float
) -> float:
    probs = softmax_with_temperature(log_probs, temperature)
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def fit_temperature(
    log_probs: npt.NDArray[np.float32],
    labels: npt.NDArray[np.integer],
    *,
    bounds: tuple[float, float] = (0.05, 20.0),
    iterations: int = 60,
) -> float:
    """Golden-section search for the NLL-minimizing temperature (on log T)."""
    if len(labels) == 0:
        raise ValueError("no calibration pixels")
    lo, hi = math.log(bounds[0]), math.log(bounds[1])
    ratio = (math.sqrt(5) - 1) / 2
    a, b = hi - ratio * (hi - lo), lo + ratio * (hi - lo)
    fa, fb = nll(log_probs, labels, math.exp(a)), nll(log_probs, labels, math.exp(b))
    for _ in range(iterations):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - ratio * (hi - lo)
            fa = nll(log_probs, labels, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + ratio * (hi - lo)
            fb = nll(log_probs, labels, math.exp(b))
    return round(math.exp((lo + hi) / 2), 4)


@dataclass(frozen=True)
class CalibrationBins:
    ece: float
    n_pixels: int
    bin_edges: list[float]
    confidence: list[float | None]
    accuracy: list[float | None]
    count: list[int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ece": self.ece,
            "n_pixels": self.n_pixels,
            "bin_edges": self.bin_edges,
            "confidence": self.confidence,
            "accuracy": self.accuracy,
            "count": self.count,
        }


def expected_calibration_error(
    probs: npt.NDArray[np.floating], labels: npt.NDArray[np.integer], n_bins: int = DEFAULT_BINS
) -> CalibrationBins:
    """Top-label ECE: count-weighted |accuracy - confidence| over equal-width bins."""
    confidence = probs.max(axis=1)
    correct = probs.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    conf_bins: list[float | None] = []
    acc_bins: list[float | None] = []
    counts: list[int] = []
    ece = 0.0
    for b in range(n_bins):
        sel = index == b
        n = int(sel.sum())
        counts.append(n)
        if n == 0:
            conf_bins.append(None)
            acc_bins.append(None)
            continue
        c, a = float(confidence[sel].mean()), float(correct[sel].mean())
        conf_bins.append(round(c, 6))
        acc_bins.append(round(a, 6))
        ece += n * abs(a - c)
    total = len(labels)
    return CalibrationBins(
        ece=round(ece / total, 6) if total else math.nan,
        n_pixels=total,
        bin_edges=[round(float(e), 6) for e in edges],
        confidence=conf_bins,
        accuracy=acc_bins,
        count=counts,
    )


def foreground_mask(
    probs: npt.NDArray[np.floating], labels: npt.NDArray[np.integer]
) -> npt.NDArray[np.bool_]:
    targets = np.asarray(TARGET_CLASS_IDS)
    return np.isin(labels, targets) | np.isin(probs.argmax(axis=1), targets)


def sample_pixels(
    proba: npt.NDArray[np.float32],
    truth: npt.NDArray[np.uint8],
    n: int,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    """Uniformly sample ``n`` valid pixels: (log-probabilities ``n x C``, labels)."""
    valid = np.flatnonzero(truth.reshape(-1) != IGNORE_INDEX)
    if valid.size == 0:
        return np.empty((0, proba.shape[0]), np.float32), np.empty(0, np.uint8)
    picked = rng.choice(valid, size=min(n, valid.size), replace=False)
    flat = proba.reshape(proba.shape[0], -1)[:, picked].T
    return np.log(np.clip(flat, 1e-8, 1.0)).astype(np.float32), truth.reshape(-1)[picked]


def collect_pixels(
    predictor: Predictor,
    samples: Sequence[Sample],
    remap: dict[int, int],
    pixels_per_image: int,
    seed: int = 0,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8]]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i, sample in enumerate(samples):
        proba = predictor.predict_proba(load_rgb(sample.image_path))
        x, y = sample_pixels(proba, load_mask(sample.mask_path, remap), pixels_per_image, rng)
        xs.append(x)
        ys.append(y)
        if (i + 1) % 50 == 0:
            log.info("sampled %d / %d images", i + 1, len(samples))
    return np.concatenate(xs), np.concatenate(ys)


def calibration_summary(
    log_probs: npt.NDArray[np.float32], labels: npt.NDArray[np.integer], temperature: float
) -> dict[str, Any]:
    probs = softmax_with_temperature(log_probs, temperature)
    fg = foreground_mask(probs, labels)
    return {
        "temperature": temperature,
        "nll": round(nll(log_probs, labels, temperature), 6),
        "overall": expected_calibration_error(probs, labels).as_dict(),
        "foreground": expected_calibration_error(probs[fg], labels[fg]).as_dict(),
    }


def reliability_svg(before: dict[str, Any], after: dict[str, Any] | None, title: str) -> str:
    """A dependency-free reliability diagram (accuracy vs confidence per bin)."""
    size, pad = 360, 48
    plot = size - 2 * pad

    def xy(c: float, a: float) -> tuple[float, float]:
        return pad + c * plot, size - pad - a * plot

    def polyline(bins: dict[str, Any], color: str) -> str:
        pts = [
            xy(c, a)
            for c, a in zip(bins["confidence"], bins["accuracy"], strict=True)
            if c is not None
        ]
        coords = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color}"/>' for x, y in pts)
        return f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>{dots}'

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        'font-family="sans-serif" font-size="11">',
        f'<rect width="{size}" height="{size}" fill="white"/>',
        f'<text x="{size / 2}" y="20" text-anchor="middle" font-size="13">{title}</text>',
        f'<line x1="{pad}" y1="{size - pad}" x2="{size - pad}" y2="{pad}" '
        'stroke="#999" stroke-dasharray="4 3"/>',
        f'<rect x="{pad}" y="{pad}" width="{plot}" height="{plot}" fill="none" stroke="#333"/>',
        f'<text x="{size / 2}" y="{size - 12}" text-anchor="middle">confidence</text>',
        f'<text x="14" y="{size / 2}" text-anchor="middle" '
        f'transform="rotate(-90 14 {size / 2})">accuracy</text>',
        polyline(before, "#d1495b"),
        f'<text x="{pad + 6}" y="{pad + 14}" fill="#d1495b">before: ECE {before["ece"]:.4f}</text>',
    ]
    if after is not None:
        parts += [
            polyline(after, "#1f6f5c"),
            f'<text x="{pad + 6}" y="{pad + 28}" fill="#1f6f5c">'
            f"after: ECE {after['ece']:.4f}</text>",
        ]
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> int:
    from beltwatch.evaluation.runner import load_predictor

    parser = argparse.ArgumentParser(description="Fit or check temperature scaling.")
    parser.add_argument("--model", required=True, choices=["random-forest", "unet", "segformer"])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--split", default="calibration", choices=["calibration", "val"])
    parser.add_argument(
        "--temperature", type=float, help="check a given temperature (val) instead of fitting"
    )
    parser.add_argument("--pixels-per-image", type=int, default=20000)
    parser.add_argument(
        "--fit-scope",
        choices=["all", "foreground"],
        default="all",
        help="pixels the temperature is fitted on; 'foreground' targets the four materials",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.temperature is None and args.split != "calibration":
        parser.error("temperatures are fitted on the calibration split only")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    data = load_data_config(args.config)
    name = data.source.name
    samples = load_samples(
        data.paths.splits_dir / f"{name}-splits.csv",
        data.paths.manifests_dir / f"{name}-images.csv",
        data.paths.raw_dir,
        args.split,
    )
    remap = load_source_remap(data.paths.manifests_dir / f"{name}-validation.json")
    predictor = load_predictor(args.model, args.model_dir)
    log_probs, labels = collect_pixels(predictor, samples, remap, args.pixels_per_image)

    if args.temperature is not None:
        temperature = args.temperature
    elif args.fit_scope == "foreground":
        fg = foreground_mask(softmax_with_temperature(log_probs, 1.0), labels)
        temperature = fit_temperature(log_probs[fg], labels[fg])
    else:
        temperature = fit_temperature(log_probs, labels)
    before = calibration_summary(log_probs, labels, 1.0)
    after = calibration_summary(log_probs, labels, temperature)
    splits_summary = json.loads((data.paths.splits_dir / f"{name}-splits.json").read_text())
    report = {
        "model": predictor.name,
        "model_dir": str(args.model_dir),
        "split": args.split,
        "fitted": args.temperature is None,
        "fit_scope": args.fit_scope,
        "temperature": temperature,
        "n_images": len(samples),
        "n_pixels": len(labels),
        "split_manifest_id": splits_summary["split_manifest_id"],
        "before": before,
        "after": after,
        "beltwatch_version": __version__,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "calibration.json").write_text(json.dumps(report, indent=2) + "\n")
    for scope in ("overall", "foreground"):
        svg = reliability_svg(
            before[scope], after[scope], f"{predictor.name} ({scope}, {args.split})"
        )
        (args.out / f"reliability-{scope}.svg").write_text(svg)
    log.info(
        "T=%.3f  ECE overall %.4f -> %.4f, foreground %.4f -> %.4f",
        temperature,
        before["overall"]["ece"],
        after["overall"]["ece"],
        before["foreground"]["ece"],
        after["foreground"]["ece"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
