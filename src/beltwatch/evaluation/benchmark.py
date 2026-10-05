"""CPU benchmark of the serving pipeline against the engineering targets.

Measures, for a model release, the warm latency of model prediction alone and
of the full per-image worker pipeline (quality checks, prediction, coverage,
uncertainty, review routing, and PNG encoding of the mask, overlay, and
uncertainty images), sequential throughput, and the process's peak resident
memory. Run it in a fresh process on the target hardware: the planning target
is 4 vCPU and 8 GB RAM.

Usage::

    uv run python scripts/benchmark.py --release model_releases/beltwatch-0.1.0 --threads 4 \\
        --out reports/benchmark.json

Targets (planning values from the design, not guarantees): warm p95 per image
of at most 3 s, at least 0.5 images per second sustained, and a worker peak of
at most 2 GB.
"""

import argparse
import io
import json
import os
import platform
import resource
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from beltwatch import __version__
from beltwatch.inference.postprocessing import FULL_FRAME
from beltwatch.inference.preprocessing import load_rgb
from beltwatch.jobs.pipeline import process_image
from beltwatch.release.bundle import Release, load_release

TARGETS = {"p95_seconds": 3.0, "images_per_second": 0.5, "peak_rss_mb": 2048.0}


def peak_rss_mb() -> float:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 1e6 if sys.platform == "darwin" else peak / 1024  # bytes on macOS, KiB on Linux


def synthetic_frames(
    n: int, size: tuple[int, int] = (1920, 1080), seed: int = 0
) -> list[npt.NDArray[np.uint8]]:
    rng = np.random.default_rng(seed)
    w, h = size
    frames = []
    for _ in range(n):
        small = rng.integers(30, 225, (h // 40, w // 40, 3), dtype=np.uint8)
        base = np.asarray(Image.fromarray(small).resize((w, h), Image.Resampling.BICUBIC))
        noise = rng.integers(-12, 12, (h, w, 3))
        frames.append(np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    return frames


def _stats(seconds: Sequence[float]) -> dict[str, float]:
    a = np.asarray(seconds)
    return {
        "p50_seconds": round(float(np.percentile(a, 50)), 4),
        "p95_seconds": round(float(np.percentile(a, 95)), 4),
        "max_seconds": round(float(a.max()), 4),
        "mean_seconds": round(float(a.mean()), 4),
    }


def run_benchmark(
    release: Release,
    frames: Sequence[npt.NDArray[np.uint8]],
    *,
    iterations: int = 20,
    warmup: int = 3,
    threads: int | None = None,
) -> dict[str, Any]:
    if threads:
        torch.set_num_threads(threads)
    predictor = release.load_predictor("cpu")
    policy = release.manifest.review_policy
    for i in range(warmup):
        predictor.predict_proba(frames[i % len(frames)])

    predict, full = [], []
    started = time.perf_counter()
    for i in range(iterations):
        rgb = frames[i % len(frames)]
        t0 = time.perf_counter()
        predictor.predict_proba(rgb)
        predict.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        outcome = process_image(
            rgb,
            image_sha256=f"benchmark-{i}",
            region_polygon=FULL_FRAME,
            predictor=predictor,
            policy=policy,
            model_version=release.version,
        )
        for array in (outcome.mask, outcome.overlay, outcome.uncertainty):
            Image.fromarray(array).save(io.BytesIO(), format="PNG")
        full.append(time.perf_counter() - t0)
    wall = time.perf_counter() - started

    full_stats = _stats(full)
    throughput = round(iterations / sum(full), 4)
    peak = round(peak_rss_mb(), 1)
    return {
        "release": release.version,
        "model_kind": release.manifest.model.kind,
        "image_size": list(frames[0].shape[1::-1]),
        "iterations": iterations,
        "predict_only": _stats(predict),
        "full_pipeline": full_stats,
        "images_per_second": throughput,
        "peak_rss_mb": peak,
        "wall_seconds": round(wall, 2),
        "targets": TARGETS,
        "meets_targets": {
            "p95_seconds": full_stats["p95_seconds"] <= TARGETS["p95_seconds"],
            "images_per_second": throughput >= TARGETS["images_per_second"],
            "peak_rss_mb": peak <= TARGETS["peak_rss_mb"],
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "torch_version": str(torch.__version__),
        },
        "note": "planning targets from the design; meaningful only on the target hardware",
        "beltwatch_version": __version__,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark a release on CPU.")
    parser.add_argument("--release", type=Path, required=True, help="release bundle directory")
    parser.add_argument(
        "--images", type=Path, help="directory of images (default: synthetic 1920x1080)"
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    if args.images:
        paths = sorted(
            p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
        frames = [load_rgb(p) for p in paths[: max(args.iterations, 1)]]
    else:
        frames = synthetic_frames(4)
    result = run_benchmark(
        load_release(args.release),
        frames,
        iterations=args.iterations,
        warmup=args.warmup,
        threads=args.threads,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    return 0
