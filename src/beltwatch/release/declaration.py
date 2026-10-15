"""Release declarations: what a release is, fixed before the locked test split is read.

A declaration (``configs/releases/<version>.yaml``) is committed before the
test split is scored. It names the weights by SHA-256, the calibration
temperature, the review policy and preprocessing versions, the frozen
baselines scored next to the release, and the val scores the release session
must reproduce. :func:`check` compares what a session built against it, so the
test split is only read for exactly the declared release, and only once: a
test report directory holding :data:`TEST_COMPLETE` is refused. A session that
stopped part-way may finish the declared evaluations; nothing is tuned between
them.

Usage::

    uv run python -m beltwatch.release.declaration configs/releases/beltwatch-0.1.0.yaml \\
        --model <best.pt> --calibration <calibration.json> --val-report <report.json> \\
        --release-dir <bundle> --test-dir <test reports>

It prints every mismatch and exits with status 1 if there is any.
"""

import argparse
import json
import math
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from beltwatch.inference.preprocessing import PREPROCESSING_VERSION
from beltwatch.provenance import file_digest
from beltwatch.review.policy import load_review_policy

TEST_COMPLETE = "complete.json"
"""Written into the test report directory once every declared test evaluation is done."""

SCORE_TOLERANCE = 1e-5
"""Allowed difference when a session re-scores val (a few pixels may flip on another CPU)."""

TestEvaluation = Literal["release", "release-uncalibrated", "all-background", "random-forest"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelDeclaration(_Strict):
    kind: Literal["unet", "segformer"]
    checkpoint: str
    mlflow_run_id: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessing_version: str
    val_foreground_macro_iou: float


class CalibrationDeclaration(_Strict):
    fit_scope: Literal["all", "foreground", "per-class"]
    temperature: float | tuple[float, ...]


class ForestDeclaration(_Strict):
    pixels_per_class: int
    prior_power: float
    val_foreground_macro_iou: float


class BaselinesDeclaration(_Strict):
    random_forest: ForestDeclaration


class TargetsDeclaration(_Strict):
    margin_over_random_forest: float
    foreground_macro_iou: float
    coverage_mae_pp: float


class ReleaseDeclaration(_Strict):
    version: str
    declared_on: date
    replaces: str | None
    reason: str
    model: ModelDeclaration
    calibration: CalibrationDeclaration
    review_policy_version: str
    baselines: BaselinesDeclaration
    test_evaluations: tuple[TestEvaluation, ...]
    test_comparisons: tuple[tuple[TestEvaluation, TestEvaluation], ...]
    targets: TargetsDeclaration


def load_declaration(path: Path) -> ReleaseDeclaration:
    return ReleaseDeclaration.model_validate(yaml.safe_load(path.read_text()))


def _same_temperature(found: Any, declared: float | tuple[float, ...]) -> bool:
    found_values = found if isinstance(found, list | tuple) else [found]
    declared_values = declared if isinstance(declared, tuple) else (declared,)
    return len(found_values) == len(declared_values) and all(
        math.isclose(a, b, abs_tol=1e-9) for a, b in zip(found_values, declared_values, strict=True)
    )


def _score_problem(name: str, found: float | None, declared: float) -> list[str]:
    if found is None or not math.isclose(found, declared, abs_tol=SCORE_TOLERANCE):
        return [f"{name}: val foreground macro IoU {found} does not reproduce {declared}"]
    return []


def check(
    declaration: ReleaseDeclaration,
    *,
    review_policy: Path = Path("configs/review_policy.yaml"),
    model: Path | None = None,
    calibration: Path | None = None,
    val_report: Path | None = None,
    release_dir: Path | None = None,
    forest_val_report: Path | None = None,
    test_dir: Path | None = None,
) -> list[str]:
    """Every way the given artifacts differ from the declaration; empty if none."""
    d = declaration
    problems = []
    if d.model.preprocessing_version != PREPROCESSING_VERSION:
        problems.append(
            f"preprocessing {PREPROCESSING_VERSION} in the code, "
            f"{d.model.preprocessing_version} declared"
        )
    policy_version = load_review_policy(review_policy).version
    if policy_version != d.review_policy_version:
        problems.append(f"review policy {policy_version}, {d.review_policy_version} declared")

    if model is not None and file_digest(model) != d.model.sha256:
        problems.append(f"{model} is not the declared weights {d.model.sha256[:12]}")

    if calibration is not None:
        fit = json.loads(calibration.read_text())
        if not fit.get("fitted") or fit.get("split") != "calibration":
            problems.append("calibration: not a fit on the calibration split")
        if fit.get("fit_scope") != d.calibration.fit_scope:
            problems.append(f"calibration: fit scope {fit.get('fit_scope')!r} is not declared")
        if not _same_temperature(fit.get("temperature"), d.calibration.temperature):
            problems.append(
                f"calibration: temperature {fit.get('temperature')} does not reproduce "
                f"{d.calibration.temperature}"
            )
        if (fit.get("provenance") or {}).get("model_sha256") != d.model.sha256:
            problems.append("calibration: fitted on other weights")

    if val_report is not None:
        report = json.loads(val_report.read_text())
        if report.get("split") != "val" or report.get("limited_to") is not None:
            problems.append("val report: not a full val evaluation")
        if not _same_temperature(report.get("temperature"), d.calibration.temperature):
            problems.append(f"val report: temperature {report.get('temperature')} not declared")
        if (report.get("provenance") or {}).get("model_sha256") != d.model.sha256:
            problems.append("val report: other weights")
        problems += _score_problem(
            "release",
            report.get("segmentation", {}).get("foreground_macro_iou"),
            d.model.val_foreground_macro_iou,
        )

    if release_dir is not None:
        manifest = json.loads((release_dir / "manifest.json").read_text())
        if manifest.get("version") != d.version:
            problems.append(f"bundle: version {manifest.get('version')!r}")
        if manifest.get("model", {}).get("sha256") != d.model.sha256:
            problems.append("bundle: other weights")
        if manifest.get("preprocessing_version") != d.model.preprocessing_version:
            problems.append("bundle: other preprocessing version")
        if not _same_temperature(manifest.get("temperature"), d.calibration.temperature):
            problems.append(f"bundle: temperature {manifest.get('temperature')}")
        if manifest.get("review_policy", {}).get("version") != d.review_policy_version:
            problems.append("bundle: other review policy")

    if forest_val_report is not None:
        report = json.loads(forest_val_report.read_text())
        if report.get("split") != "val" or report.get("limited_to") is not None:
            problems.append("forest val report: not a full val evaluation")
        problems += _score_problem(
            "random forest",
            report.get("segmentation", {}).get("foreground_macro_iou"),
            d.baselines.random_forest.val_foreground_macro_iou,
        )

    if test_dir is not None and (test_dir / TEST_COMPLETE).exists():
        problems.append(f"{test_dir}: the test split was already scored for this release")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check a release session against its declaration.")
    parser.add_argument("declaration", type=Path)
    parser.add_argument("--review-policy", type=Path, default=Path("configs/review_policy.yaml"))
    parser.add_argument("--model", type=Path, help="checkpoint file")
    parser.add_argument("--calibration", type=Path, help="calibration.json")
    parser.add_argument("--val-report", type=Path, help="report.json of the release on val")
    parser.add_argument("--release-dir", type=Path, help="built bundle directory")
    parser.add_argument("--forest-val-report", type=Path, help="report.json of the refit forest")
    parser.add_argument(
        "--test-dir", type=Path, help=f"test report directory; refused if it has {TEST_COMPLETE}"
    )
    args = parser.parse_args(argv)

    declaration = load_declaration(args.declaration)
    problems = check(
        declaration,
        review_policy=args.review_policy,
        model=args.model,
        calibration=args.calibration,
        val_report=args.val_report,
        release_dir=args.release_dir,
        forest_val_report=args.forest_val_report,
        test_dir=args.test_dir,
    )
    for problem in problems:
        print(f"MISMATCH {declaration.version}: {problem}")
    if not problems:
        print(f"{declaration.version}: matches the declaration")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
