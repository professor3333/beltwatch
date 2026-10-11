from collections.abc import Sequence

import pytest

from beltwatch.jobs import pipeline

REGION = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


def key(
    image: str = "a" * 64,
    region: Sequence[Sequence[float]] = REGION,
    model: str = "unet-1",
    policy: str = "review-policy-1",
) -> str:
    return pipeline.cache_key(image, region, model, policy)


def test_identical_inputs_give_identical_keys() -> None:
    assert key() == key(region=[tuple(v) for v in REGION])


@pytest.mark.parametrize(
    "changed",
    [
        {"image": "b" * 64},
        {"region": [[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]]},
        {"model": "unet-2"},
        {"policy": "review-policy-2"},
    ],
)
def test_every_component_changes_the_key(changed: dict[str, object]) -> None:
    assert key(**changed) != key()  # type: ignore[arg-type]


def test_preprocessing_version_changes_the_key(monkeypatch: pytest.MonkeyPatch) -> None:
    before = key()
    monkeypatch.setattr(pipeline, "PREPROCESSING_VERSION", "pp-test")

    assert key() != before


def test_region_float_noise_below_rounding_shares_a_key() -> None:
    assert key(region=[[x + 1e-9, y] for x, y in REGION]) == key()


def test_vertex_order_is_part_of_the_key() -> None:
    # The same polygon traced from another vertex is not recognized as identical;
    # that costs a cache miss, never a wrong result.
    assert key(region=REGION[1:] + REGION[:1]) != key()
