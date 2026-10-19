from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from beltwatch.data.dataset import Sample
from beltwatch.evaluation import review_policy as rp
from beltwatch.release.bundle import build_release
from beltwatch.review.policy import in_random_audit, load_review_policy

REPO = Path(__file__).resolve().parents[2]
BASE = load_review_policy(REPO / "configs/review_policy.yaml")
TUNING = rp.load_tuning_config(REPO / "configs/review_policy_tuning.yaml")


def scored(pred: float, positive: bool, uncertain: float = 0.0, n: int = 0) -> rp.ScoredImage:
    return rp.ScoredImage(
        image_id=f"val/{n}",
        image_sha256=f"{n:064x}",
        audit_positive=positive,
        true_total=0.1 if positive else 0.0,
        pred_total=pred,
        uncertain_fraction=uncertain,
        quality_issues=(),
    )


def test_the_declared_tuning_starts_from_the_release_policy() -> None:
    assert BASE.version == "review-policy-1"
    assert TUNING.base_policy == Path("configs/review_policy.yaml")
    assert TUNING.candidate_version != BASE.version


def test_policy_metrics_count_routes_recall_and_false_alarms() -> None:
    policy = BASE.model_copy(update={"random_audit_rate": 0.0})
    images = [
        scored(0.20, True, n=0),  # found by coverage
        scored(0.01, True, uncertain=0.5, n=1),  # found by uncertainty
        scored(0.01, True, n=2),  # missed
        scored(0.20, False, n=3),  # false alarm
        rp.ScoredImage("val/4", "4" * 64, False, 0.0, 0.0, 0.0, ("too_blurry",)),
    ]

    m = rp.evaluate_policy(images, policy)

    assert m["review_rate"] == 4 / 5
    assert m["recall"] == 2 / 3 and m["missed_audit_positive"] == 1
    assert m["false_alarm_share"] == 2 / 4  # the false alarm and the retake
    assert m["routes"] == {"manual_review": 1, "none": 1, "retake_image": 1, "review": 2}


def test_candidates_cover_the_grid_and_share_one_random_audit_sample() -> None:
    policies = rp.candidates(BASE, TUNING)

    grid = TUNING.grid
    assert len(policies) == len(grid.coverage_threshold) * len(grid.uncertain_fraction_threshold)
    assert {p.version for p in policies} == {TUNING.candidate_version}
    sha = "ab" * 32
    assert len({in_random_audit(sha, p) for p in policies}) == 1


def test_selection_takes_the_lowest_review_rate_above_the_recall_floor() -> None:
    results = [
        {"review_rate": 0.9, "recall": 0.99, "false_alarm_share": 0.1},
        {"review_rate": 0.7, "recall": 0.90, "false_alarm_share": 0.1},  # below the floor
        {"review_rate": 0.8, "recall": 0.96, "false_alarm_share": 0.2},
        {"review_rate": 0.8, "recall": 0.97, "false_alarm_share": 0.1},  # tie, fewer false alarms
    ]

    assert rp.select(results, 0.95) == 3
    assert rp.select(results, 0.999) is None


def test_adoption_needs_a_large_enough_review_rate_reduction() -> None:
    # Positives are flagged at any threshold; negatives only at the lowest ones.
    images = [scored(0.20, True, n=i) for i in range(18)]
    images += [scored(0.025, False, n=100 + i) for i in range(2)]

    result = rp.tune(images, BASE, TUNING)

    assert result["prevalence"] == 0.9
    assert result["current"]["recall"] == 1.0
    assert result["selected"]["recall"] == 1.0
    reduction = result["current"]["review_rate"] - result["selected"]["review_rate"]
    assert result["adopted"] == (reduction >= TUNING.adoption.min_review_rate_reduction)


def test_a_policy_missing_the_floor_is_replaced_by_one_that_meets_it() -> None:
    images = [scored(0.03, True, n=i) for i in range(10)]  # below the 5% threshold

    result = rp.tune(images, BASE, TUNING)

    assert result["current"]["recall"] < TUNING.selection.recall_floor
    assert result["adopted"] and result["selected"]["coverage_threshold"] <= 0.03


def test_scoring_runs_the_serving_pipeline_and_round_trips(tmp_path: Path) -> None:
    release = build_release(
        tmp_path / "releases", version="t1", kind="all-background", model_path=None,
        review_policy=BASE,
    )  # fmt: skip
    rng = np.random.default_rng(0)
    samples = []
    for i, material in enumerate((0, 2)):
        image = tmp_path / f"{i}.png"
        Image.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(image)
        mask = np.zeros((60, 80), dtype=np.uint8)
        mask[:30, :] = material  # half the frame
        Image.fromarray(mask).save(tmp_path / f"{i}-mask.png")
        samples.append(Sample(f"val/{i}", image, tmp_path / f"{i}-mask.png"))

    images = rp.score_images(release, samples, {0: 0, 2: 2})
    rp.write_scored(images, tmp_path / "per_image.csv")

    assert [s.audit_positive for s in images] == [False, True]
    assert images[1].true_total == pytest.approx(0.5)
    assert all(s.pred_total == 0.0 for s in images)  # all-background predicts no material
    assert rp.read_scored(tmp_path / "per_image.csv") == images


def test_only_val_is_accepted() -> None:
    with pytest.raises(SystemExit):
        rp.main(["--release", "x", "--split", "test", "--out", "y"])
