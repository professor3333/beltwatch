from pathlib import Path

import numpy as np
import pytest

from beltwatch.inference.quality import assess_quality
from beltwatch.review.policy import decide, in_random_audit, load_review_policy

POLICY = load_review_policy(Path("configs/review_policy.yaml"))


def run(
    coverage: float = 0.0, uncertain: float = 0.0, issues: tuple[str, ...] = (), sha: str = "a"
) -> object:
    return decide(
        total_target_coverage=coverage,
        uncertain_fraction=uncertain,
        quality_issues=issues,
        image_sha256=sha,
        policy=POLICY,
    )


def test_unusable_input_goes_to_retake() -> None:
    d = run(coverage=0.5, issues=("blurred",))
    assert (d.route, d.reasons, d.review_required) == ("retake_image", ("blurred",), True)


def test_high_coverage_goes_to_review() -> None:
    d = run(coverage=0.08)
    assert d.route == "review" and d.reasons == ("target_coverage",)
    assert d.priority == pytest.approx(0.08)


def test_uncertainty_goes_to_manual_review_and_raises_priority() -> None:
    d = run(coverage=0.08, uncertain=0.3)
    assert d.route == "manual_review"
    assert d.reasons == ("target_coverage", "uncertain_regions")
    assert d.priority == pytest.approx(0.08 + 0.5 * 0.3)


def test_low_risk_images_are_randomly_audited_at_the_configured_rate() -> None:
    shas = [f"{i:064x}" for i in range(4000)]
    audited = [s for s in shas if in_random_audit(s, POLICY)]
    assert 0.08 < len(audited) / len(shas) < 0.12
    assert run(sha=audited[0]).reasons == ("random_audit",)
    not_audited = next(s for s in shas if s not in audited)
    d = run(sha=not_audited)
    assert (d.route, d.review_required) == ("none", False)


def test_random_audit_is_deterministic() -> None:
    assert in_random_audit("abc", POLICY) == in_random_audit("abc", POLICY)


def test_quality_checks() -> None:
    rng = np.random.default_rng(0)
    sharp = rng.integers(40, 216, (120, 160, 3), dtype=np.uint8)
    assert assess_quality(sharp, POLICY.quality).issues == ()
    flat = np.full((120, 160, 3), 128, dtype=np.uint8)
    assert "blurred" in assess_quality(flat, POLICY.quality).issues
    dark = (sharp // 12).astype(np.uint8)
    assert "underexposed" in assess_quality(dark, POLICY.quality).issues
    bright = np.full((120, 160, 3), 255, dtype=np.uint8)
    issues = assess_quality(bright, POLICY.quality).issues
    assert "overexposed" in issues and "clipped_exposure" in issues
