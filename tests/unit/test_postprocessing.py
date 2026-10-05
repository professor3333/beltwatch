import numpy as np
import pytest

from beltwatch.inference.postprocessing import FULL_FRAME, compute_coverage, rasterize_region
from beltwatch.labels import BACKGROUND, CARDBOARD, IGNORE_INDEX, METAL


def test_full_frame_region_covers_everything() -> None:
    assert rasterize_region(FULL_FRAME, 20, 10).all()


def test_rectangle_region() -> None:
    region = rasterize_region([[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]], 20, 10)

    assert region[:, :9].all()
    assert not region[:, 12:].any()


@pytest.mark.parametrize(
    ("polygon", "message"),
    [
        ([[0.0, 0.0], [1.0, 1.0]], "at least 3"),
        ([[0.0, 0.0], [1.5, 0.0], [0.0, 1.0]], "outside"),
        ([[0.0, 0.0, 1.0], [1.0, 0.0], [0.0, 1.0]], "pair"),
    ],
)
def test_region_validation(polygon: list[list[float]], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        rasterize_region(polygon, 10, 10)


def test_coverage_counts_region_pixels_only() -> None:
    labels = np.zeros((10, 10), dtype=np.uint8)
    labels[:, :5] = CARDBOARD  # left half
    labels[0, 9] = METAL
    right = np.zeros((10, 10), dtype=bool)
    right[:, 5:] = True

    cov = compute_coverage(labels, right)

    assert cov.valid_pixels == 50
    assert cov.per_class["cardboard"] == 0.0
    assert cov.per_class["metal"] == pytest.approx(1 / 50)
    assert cov.total_target == pytest.approx(1 / 50)


def test_coverage_excludes_ignore_and_invalid_pixels() -> None:
    labels = np.full((4, 4), CARDBOARD, dtype=np.uint8)
    labels[0] = IGNORE_INDEX
    valid = np.ones((4, 4), dtype=bool)
    valid[1] = False
    labels[2] = BACKGROUND

    cov = compute_coverage(labels, np.ones((4, 4), dtype=bool), valid)

    assert cov.valid_pixels == 8
    assert cov.per_class["cardboard"] == pytest.approx(0.5)


def test_coverage_of_empty_region_raises() -> None:
    with pytest.raises(ValueError, match="no valid pixels"):
        compute_coverage(np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=bool))


def test_coverage_is_background_free() -> None:
    cov = compute_coverage(np.zeros((4, 4), dtype=np.uint8), np.ones((4, 4), dtype=bool))
    assert set(cov.per_class) == {"cardboard", "soft_plastic", "rigid_plastic", "metal"}
    assert cov.total_target == 0.0
