import numpy as np
import pytest

from beltwatch import labels

# A source taxonomy deliberately numbered in a different order from ours.
SOURCE_CATEGORIES = {1: "rigid_plastic", 2: "cardboard", 3: "metal", 4: "soft_plastic"}


def test_class_table_is_consistent() -> None:
    assert labels.NUM_CLASSES == 5
    assert labels.CLASS_NAMES[labels.BACKGROUND] == "background"
    assert labels.BACKGROUND not in labels.TARGET_CLASS_IDS
    assert len(labels.TARGET_CLASS_IDS) == 4
    assert labels.IGNORE_INDEX not in labels.CLASS_NAMES


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Soft Plastic", "soft_plastic"), ("soft-plastic", "soft_plastic"), (" Metal ", "metal")],
)
def test_normalize_name(raw: str, expected: str) -> None:
    assert labels.normalize_name(raw) == expected


def test_remap_is_by_name_not_by_id() -> None:
    remap = labels.build_source_remap(SOURCE_CATEGORIES)
    assert remap == {
        1: labels.RIGID_PLASTIC,
        2: labels.CARDBOARD,
        3: labels.METAL,
        4: labels.SOFT_PLASTIC,
    }


def test_remap_accepts_source_background() -> None:
    remap = labels.build_source_remap({0: "Background", **SOURCE_CATEGORIES})
    assert remap[0] == labels.BACKGROUND


def test_remap_rejects_unknown_category() -> None:
    with pytest.raises(ValueError, match="unknown source category"):
        labels.build_source_remap({**SOURCE_CATEGORIES, 5: "glass"})


def test_remap_rejects_duplicate_class() -> None:
    with pytest.raises(ValueError, match="both map to"):
        labels.build_source_remap({**SOURCE_CATEGORIES, 5: "Cardboard"})


def test_remap_rejects_missing_target_class() -> None:
    partial = {k: v for k, v in SOURCE_CATEGORIES.items() if v != "metal"}
    with pytest.raises(ValueError, match="missing target classes"):
        labels.build_source_remap(partial)


def test_remap_mask_preserves_background_and_ignore() -> None:
    mask_map = {0: labels.BACKGROUND, **labels.build_source_remap(SOURCE_CATEGORIES)}
    source = np.array([[0, 1, 2], [3, 4, 255]], dtype=np.uint8)

    out = labels.remap_mask(source, mask_map)

    assert out.dtype == np.uint8
    expected = [
        [labels.BACKGROUND, labels.RIGID_PLASTIC, labels.CARDBOARD],
        [labels.METAL, labels.SOFT_PLASTIC, labels.IGNORE_INDEX],
    ]
    np.testing.assert_array_equal(out, np.array(expected, dtype=np.uint8))


def test_remap_mask_all_background_survives() -> None:
    # Guards against `mask > 0` style bugs that drop label 0.
    out = labels.remap_mask(np.zeros((4, 4), dtype=np.uint8), {0: labels.BACKGROUND})
    assert out.shape == (4, 4)
    assert set(np.unique(out)) == {labels.BACKGROUND}


def test_remap_mask_rejects_unmapped_values() -> None:
    with pytest.raises(ValueError, match=r"unmapped source values: \[7\]"):
        labels.remap_mask(np.array([[0, 7]], dtype=np.uint8), {0: labels.BACKGROUND})


def test_remap_mask_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match=r"0\.\.255"):
        labels.remap_mask(np.array([[0, 300]], dtype=np.int32), {0: labels.BACKGROUND})
