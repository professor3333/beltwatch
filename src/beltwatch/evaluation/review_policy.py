"""Measure and tune the review policy's routing on val (docs/evaluation.md).

The review-workload curve ranks images; this module measures the routing
decision itself: which images a policy sends to a person. Each val image is
scored once through the serving pipeline (:func:`process_image`, full frame
as the inspection region) with a release bundle, and its predicted coverage,
uncertain fraction and quality issues are kept. Every policy in the declared
grid (``configs/review_policy_tuning.yaml``) is then applied to those values
with :func:`decide`, so the candidates differ only in their thresholds.

Usage::

    uv run python -m beltwatch.evaluation.review_policy \\
        --release model_releases/beltwatch-0.1.0 --split val --out reports/review-policy

Writes ``per_image.csv`` (the scored values; regenerated only if missing) and
``review_policy.json`` (every candidate, review-policy-1, the selection and the
adoption decision). The calibration and test splits are refused.
"""

import argparse
import csv
import hashlib
import itertools
import json
import logging
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from beltwatch import __version__
from beltwatch.data.config import load_data_config
from beltwatch.data.dataset import Sample, load_samples
from beltwatch.data.masks import load_mask, load_source_remap
from beltwatch.evaluation.coverage import is_audit_positive
from beltwatch.inference.postprocessing import FULL_FRAME, compute_coverage
from beltwatch.inference.preprocessing import load_rgb
from beltwatch.jobs.pipeline import process_image
from beltwatch.provenance import git_commit
from beltwatch.release.bundle import Release, load_release
from beltwatch.review.policy import ReviewPolicy, decide, load_review_policy

log = logging.getLogger(__name__)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Grid(_Frozen):
    coverage_threshold: tuple[float, ...] = Field(min_length=1)
    uncertain_fraction_threshold: tuple[float, ...] = Field(min_length=1)


class Selection(_Frozen):
    recall_floor: float = Field(gt=0, le=1)


class Adoption(_Frozen):
    min_review_rate_reduction: float = Field(ge=0, le=1)


class TuningConfig(_Frozen):
    version: str
    release: str
    base_policy: Path
    candidate_version: str
    grid: Grid
    selection: Selection
    adoption: Adoption


def load_tuning_config(path: Path) -> TuningConfig:
    with path.open(encoding="utf-8") as fh:
        return TuningConfig.model_validate(yaml.safe_load(fh))


@dataclass(frozen=True)
class ScoredImage:
    image_id: str
    image_sha256: str
    audit_positive: bool
    true_total: float
    pred_total: float
    uncertain_fraction: float
    quality_issues: tuple[str, ...]


def score_images(
    release: Release, samples: Sequence[Sample], remap: dict[int, int]
) -> list[ScoredImage]:
    """Run each image through the serving pipeline once and keep the routing inputs."""
    predictor = release.load_predictor("cpu")
    scored = []
    for i, sample in enumerate(samples):
        digest = hashlib.sha256(sample.image_path.read_bytes()).hexdigest()
        rgb = load_rgb(sample.image_path)
        outcome = process_image(
            rgb,
            image_sha256=digest,
            region_polygon=FULL_FRAME,
            predictor=predictor,
            policy=release.manifest.review_policy,
            model_version=release.version,
        )
        truth = compute_coverage(
            load_mask(sample.mask_path, remap), np.ones(rgb.shape[:2], dtype=bool)
        )
        result = outcome.result
        scored.append(
            ScoredImage(
                image_id=sample.image_id,
                image_sha256=digest,
                audit_positive=is_audit_positive(truth),
                true_total=round(truth.total_target, 6),
                pred_total=result["total_target_coverage"],
                uncertain_fraction=result["uncertain_pixel_fraction"],
                quality_issues=tuple(result["quality"]["issues"]),
            )
        )
        if (i + 1) % 50 == 0:
            log.info("scored %d / %d images", i + 1, len(samples))
    return scored


FIELDS = [
    "image_id",
    "image_sha256",
    "audit_positive",
    "true_total",
    "pred_total",
    "uncertain_fraction",
    "quality_issues",
]


def write_scored(scored: Sequence[ScoredImage], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for s in scored:
            writer.writerow(
                {
                    **{f: getattr(s, f) for f in FIELDS if f != "quality_issues"},
                    "quality_issues": ";".join(s.quality_issues),
                }
            )


def read_scored(path: Path) -> list[ScoredImage]:
    with path.open(newline="", encoding="utf-8") as fh:
        return [
            ScoredImage(
                image_id=row["image_id"],
                image_sha256=row["image_sha256"],
                audit_positive=row["audit_positive"] == "True",
                true_total=float(row["true_total"]),
                pred_total=float(row["pred_total"]),
                uncertain_fraction=float(row["uncertain_fraction"]),
                quality_issues=tuple(i for i in row["quality_issues"].split(";") if i),
            )
            for row in csv.DictReader(fh)
        ]


def evaluate_policy(scored: Sequence[ScoredImage], policy: ReviewPolicy) -> dict[str, Any]:
    """Routing metrics of ``policy`` on the scored images."""
    routes: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    flagged = positives = found = false_alarms = 0
    for s in scored:
        decision = decide(
            total_target_coverage=s.pred_total,
            uncertain_fraction=s.uncertain_fraction,
            quality_issues=s.quality_issues,
            image_sha256=s.image_sha256,
            policy=policy,
        )
        routes[decision.route] += 1
        reasons.update(decision.reasons)
        flagged += decision.review_required
        positives += s.audit_positive
        found += decision.review_required and s.audit_positive
        false_alarms += decision.review_required and not s.audit_positive
    n = len(scored)
    return {
        "coverage_threshold": policy.coverage_threshold,
        "uncertain_fraction_threshold": policy.uncertainty.uncertain_fraction_threshold,
        "review_rate": flagged / n,
        "recall": found / positives if positives else None,
        "false_alarm_share": false_alarms / flagged if flagged else None,
        "missed_audit_positive": positives - found,
        "flagged": flagged,
        "routes": dict(sorted(routes.items())),
        "reasons": dict(sorted(reasons.items())),
    }


def candidates(base: ReviewPolicy, config: TuningConfig) -> list[ReviewPolicy]:
    """Every grid point, in declared order, with the candidate version (same random audits)."""
    out = []
    for coverage, uncertain in itertools.product(
        config.grid.coverage_threshold, config.grid.uncertain_fraction_threshold
    ):
        out.append(
            base.model_copy(
                update={
                    "version": config.candidate_version,
                    "coverage_threshold": coverage,
                    "uncertainty": base.uncertainty.model_copy(
                        update={"uncertain_fraction_threshold": uncertain}
                    ),
                }
            )
        )
    return out


def select(results: Sequence[dict[str, Any]], recall_floor: float) -> int | None:
    """Index of the lowest review rate with recall at or above the floor, or None."""
    eligible = [
        i for i, r in enumerate(results) if r["recall"] is not None and r["recall"] >= recall_floor
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda i: (results[i]["review_rate"], results[i]["false_alarm_share"]))


def tune(scored: Sequence[ScoredImage], base: ReviewPolicy, config: TuningConfig) -> dict[str, Any]:
    current = evaluate_policy(scored, base)
    results = [evaluate_policy(scored, p) for p in candidates(base, config)]
    chosen = select(results, config.selection.recall_floor)
    floor = config.selection.recall_floor
    if chosen is None:
        adopted, reason = False, f"no candidate reaches recall {floor}"
    elif current["recall"] is None or current["recall"] < floor:
        adopted, reason = True, f"{base.version} misses recall {floor}; the candidate meets it"
    else:
        reduction = current["review_rate"] - results[chosen]["review_rate"]
        adopted = reduction >= config.adoption.min_review_rate_reduction
        reason = (
            f"review rate {reduction:+.4f} below {base.version} "
            f"(adoption needs {config.adoption.min_review_rate_reduction})"
        )
    positives = sum(s.audit_positive for s in scored)
    return {
        "tuning_version": config.version,
        "n_images": len(scored),
        "n_audit_positive": positives,
        "prevalence": positives / len(scored) if scored else None,
        "current": {"version": base.version, **current},
        "candidates": results,
        "selected": None if chosen is None else {"index": chosen, **results[chosen]},
        "adopted": adopted,
        "adoption_reason": reason,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure and tune the review policy on val.")
    parser.add_argument("--release", type=Path, required=True, help="release bundle directory")
    parser.add_argument("--split", choices=["val"], default="val")
    parser.add_argument("--tuning", type=Path, default=Path("configs/review_policy_tuning.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tuning = load_tuning_config(args.tuning)
    release = load_release(args.release)
    if release.version != tuning.release:
        parser.error(f"{args.tuning} declares {tuning.release}, not {release.version}")
    base = load_review_policy(tuning.base_policy)
    if base.version != release.manifest.review_policy.version:
        parser.error(f"{tuning.base_policy} is not the release's policy")

    args.out.mkdir(parents=True, exist_ok=True)
    per_image = args.out / "per_image.csv"
    if not per_image.exists():
        data = load_data_config(args.config)
        name = data.source.name
        samples = load_samples(
            data.paths.splits_dir / f"{name}-splits.csv",
            data.paths.manifests_dir / f"{name}-images.csv",
            data.paths.raw_dir,
            args.split,
        )
        remap = load_source_remap(data.paths.manifests_dir / f"{name}-validation.json")
        write_scored(score_images(release, samples, remap), per_image)
    result = {
        "split": args.split,
        "release": release.version,
        "model_sha256": release.manifest.model.sha256,
        "git_commit": git_commit(),
        "beltwatch_version": __version__,
        **tune(read_scored(per_image), base, tuning),
    }
    (args.out / "review_policy.json").write_text(json.dumps(result, indent=2) + "\n")
    current, selected = result["current"], result["selected"]
    log.info(
        "%s: review rate %.3f, recall %s; selected %s; adopted %s (%s)",
        current["version"],
        current["review_rate"],
        current["recall"],
        None
        if selected is None
        else (selected["coverage_threshold"], selected["uncertain_fraction_threshold"]),
        result["adopted"],
        result["adoption_reason"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
