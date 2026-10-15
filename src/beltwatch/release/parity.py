"""Runtime parity: the served bundle predicts what the evaluated model predicted.

Serving has no exported runtime (ONNX) yet, so the worker and the evaluation
runner use the same PyTorch predictor class. This check still loads both the
way each path does (the bundle as the worker loads it, the checkpoint as
``scripts/evaluate.py`` loads it) and compares their probabilities on real
frames. A bundle built with other weights, another temperature or other
preprocessing fails it before the test split is read.

Usage::

    uv run python -m beltwatch.release.parity --release model_releases/<v> \\
        --model segformer --model-dir models/segformer-b0-ce --temperature 1.04 --images <dir>
"""

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from beltwatch.evaluation.runner import load_predictor
from beltwatch.inference.predictor import Predictor
from beltwatch.inference.preprocessing import load_rgb
from beltwatch.inference.temperature import parse_temperature
from beltwatch.release.bundle import load_release

TOLERANCE = 1e-5
"""Largest allowed per-pixel probability difference."""


def max_differences(
    served: Predictor, evaluated: Predictor, frames: Sequence[npt.NDArray[np.uint8]]
) -> list[float]:
    """Largest absolute probability difference per frame."""
    return [
        float(np.abs(served.predict_proba(rgb) - evaluated.predict_proba(rgb)).max())
        for rgb in frames
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check served vs evaluated predictions.")
    parser.add_argument("--release", type=Path, required=True, help="release bundle directory")
    parser.add_argument("--model", required=True, choices=["unet", "segformer", "random-forest"])
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=parse_temperature, default=1.0)
    parser.add_argument("--images", type=Path, required=True, help="directory of frames")
    parser.add_argument("--n", type=int, default=3, help="number of frames")
    args = parser.parse_args(argv)

    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg"})
    paths = paths[: args.n]
    if not paths:
        parser.error(f"no frames in {args.images}")
    served = load_release(args.release).load_predictor("cpu")
    evaluated = load_predictor(args.model, args.model_dir, args.temperature)
    differences = max_differences(served, evaluated, [load_rgb(p) for p in paths])
    for path, difference in zip(paths, differences, strict=True):
        print(f"{path.name}: max |served - evaluated| = {difference:.2e}")
    ok = max(differences) <= TOLERANCE
    print("parity" if ok else f"MISMATCH: above {TOLERANCE}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
