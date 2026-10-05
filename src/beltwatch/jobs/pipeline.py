"""Process one uploaded image into a result and its artifacts.

Steps: decode → input-quality checks → prediction at original resolution →
coverage inside the inspection region → uncertainty → review decision →
overlay, mask, and uncertainty images. Retakes are still segmented, so a
reviewer can see what the model made of a poor image, but they are routed to
``retake_image``.
"""

import hashlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from beltwatch.inference.postprocessing import (
    compute_coverage,
    normalized_entropy,
    rasterize_region,
    render_overlay,
    render_uncertainty,
)
from beltwatch.inference.predictor import Predictor
from beltwatch.inference.preprocessing import PREPROCESSING_VERSION
from beltwatch.inference.quality import assess_quality
from beltwatch.review.policy import ReviewPolicy, decide


def cache_key(
    image_sha256: str,
    region: Sequence[Sequence[float]],
    model_version: str,
    review_policy_version: str,
) -> str:
    """Identical inputs under identical versions give identical results."""
    payload = json.dumps(
        {
            "image": image_sha256,
            "region": [[round(float(x), 6), round(float(y), 6)] for x, y in region],
            "preprocessing": PREPROCESSING_VERSION,
            "model": model_version,
            "policy": review_policy_version,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True)
class ImageOutcome:
    result: dict[str, Any]
    mask: npt.NDArray[np.uint8]
    overlay: npt.NDArray[np.uint8]
    uncertainty: npt.NDArray[np.uint8]


def process_image(
    rgb: npt.NDArray[np.uint8],
    *,
    image_sha256: str,
    region_polygon: Sequence[Sequence[float]],
    predictor: Predictor,
    policy: ReviewPolicy,
    model_version: str,
) -> ImageOutcome:
    started = time.perf_counter()
    height, width = rgb.shape[:2]
    region = rasterize_region(region_polygon, width, height)
    quality = assess_quality(rgb, policy.quality)

    proba = predictor.predict_proba(rgb)
    labels = proba.argmax(axis=0).astype(np.uint8)
    entropy = normalized_entropy(proba)
    uncertain = entropy > policy.uncertainty.entropy_threshold

    coverage = compute_coverage(labels, region)
    uncertain_fraction = float((uncertain & region).sum()) / coverage.valid_pixels
    decision = decide(
        total_target_coverage=coverage.total_target,
        uncertain_fraction=uncertain_fraction,
        quality_issues=quality.issues,
        image_sha256=image_sha256,
        policy=policy,
    )
    result = {
        "status": "complete",
        "model_version": model_version,
        "preprocessing_version": PREPROCESSING_VERSION,
        "review_policy_version": policy.version,
        "visible_coverage": {k: round(v, 6) for k, v in coverage.per_class.items()},
        "total_target_coverage": round(coverage.total_target, 6),
        "uncertain_pixel_fraction": round(uncertain_fraction, 6),
        "inspection_region_pixels": coverage.valid_pixels,
        "review_required": decision.review_required,
        "review_reasons": list(decision.reasons),
        "route": decision.route,
        "priority": decision.priority,
        "quality": quality.as_dict(),
        "processing_seconds": round(time.perf_counter() - started, 3),
    }
    return ImageOutcome(
        result=result,
        mask=labels,
        overlay=render_overlay(rgb, labels, uncertain, region),
        uncertainty=render_uncertainty(entropy),
    )
