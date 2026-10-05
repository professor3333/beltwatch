"""Decide whether a result needs a person, why, and with what priority.

Routes:

* ``retake_image``: the input is unusable (blurred, badly exposed). Its
  prediction is not trusted.
* ``manual_review``: a large share of the inspection region is uncertain.
* ``review``: suspected target material at or above the coverage threshold,
  or a random audit of an apparently low-risk image.
* ``none``: no review is required, which is **not** a certification that the
  belt is clean.

Random audits are deterministic per image and policy version (a hash of the
image digest), so re-running an audit gives the same routing.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Route = Literal["retake_image", "manual_review", "review", "none"]


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class UncertaintyPolicy(_Strict):
    entropy_threshold: float = Field(gt=0, lt=1)
    uncertain_fraction_threshold: float = Field(gt=0, le=1)
    priority_weight: float = Field(ge=0)


class QualityPolicy(_Strict):
    min_sharpness: float = Field(ge=0)
    min_mean_brightness: float = Field(ge=0, le=255)
    max_mean_brightness: float = Field(ge=0, le=255)
    max_clipped_fraction: float = Field(gt=0, le=1)


class ReviewPolicy(_Strict):
    version: str
    coverage_threshold: float = Field(gt=0, lt=1)
    uncertainty: UncertaintyPolicy
    random_audit_rate: float = Field(ge=0, le=1)
    quality: QualityPolicy


def load_review_policy(path: Path) -> ReviewPolicy:
    with path.open(encoding="utf-8") as fh:
        return ReviewPolicy.model_validate(yaml.safe_load(fh))


@dataclass(frozen=True)
class Decision:
    review_required: bool
    route: Route
    reasons: tuple[str, ...]
    priority: float


def in_random_audit(image_sha256: str, policy: ReviewPolicy) -> bool:
    digest = hashlib.sha256(f"{policy.version}:{image_sha256}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64 < policy.random_audit_rate


def decide(
    *,
    total_target_coverage: float,
    uncertain_fraction: float,
    quality_issues: tuple[str, ...],
    image_sha256: str,
    policy: ReviewPolicy,
) -> Decision:
    if quality_issues:
        return Decision(True, "retake_image", quality_issues, priority=1.0)

    reasons: list[str] = []
    if total_target_coverage >= policy.coverage_threshold:
        reasons.append("target_coverage")
    uncertain = uncertain_fraction >= policy.uncertainty.uncertain_fraction_threshold
    if uncertain:
        reasons.append("uncertain_regions")
    if not reasons and in_random_audit(image_sha256, policy):
        reasons.append("random_audit")

    priority = total_target_coverage + policy.uncertainty.priority_weight * uncertain_fraction
    route: Route = "manual_review" if uncertain else ("review" if reasons else "none")
    return Decision(bool(reasons), route, tuple(reasons), round(priority, 6))
