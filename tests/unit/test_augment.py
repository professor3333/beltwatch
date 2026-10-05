import numpy as np

from beltwatch.data import augment as aug
from beltwatch.labels import IGNORE_INDEX

CONFIG = aug.AugmentConfig(crop_size=64, scale_range=(0.8, 1.2))


def coded_pair(h: int = 64, w: int = 96, block: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """Image whose red channel encodes the mask, so alignment is checkable."""
    rng = np.random.default_rng(1)
    blocks = rng.integers(0, 5, (-(-h // block), -(-w // block)))
    mask = np.repeat(np.repeat(blocks, block, 0), block, 1)[:h, :w].astype(np.uint8)
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[..., 0] = mask * 50
    return rgb, mask


def test_geometric_ops_keep_image_and_mask_aligned() -> None:
    rgb, mask = coded_pair()
    for seed in range(20):
        out_rgb, out_mask, valid = aug.augment(
            rgb, mask, 96, CONFIG, np.random.default_rng(seed), photometric_ops=False
        )
        assert out_rgb.shape == (64, 64, 3)
        assert out_mask.shape == valid.shape == (64, 64)
        decoded = np.rint(out_rgb[..., 0] / 50).astype(np.uint8)
        inside = valid & (out_mask != IGNORE_INDEX)
        # Bilinear image vs nearest mask may disagree only on block borders...
        assert (decoded[inside] == out_mask[inside]).mean() > 0.9
        # ...while a misaligned mask disagrees far more (the check is sensitive).
        shifted = np.roll(out_mask, 8, axis=1)
        assert (decoded[inside] == shifted[inside]).mean() < 0.8


def test_mask_never_gains_new_class_ids() -> None:
    rgb, mask = coded_pair()
    for seed in range(20):
        _, out_mask, _ = aug.augment(rgb, mask, 96, CONFIG, np.random.default_rng(seed))
        assert set(np.unique(out_mask)) <= set(np.unique(mask)) | {IGNORE_INDEX}


def test_short_side_is_padded_with_ignore() -> None:
    rgb, mask = coded_pair(h=30, w=96)

    out_rgb, out_mask, valid = aug.random_crop(rgb, mask, 64, np.random.default_rng(0))

    assert out_mask.shape == (64, 64)
    assert np.all(out_mask[~valid] == IGNORE_INDEX)
    assert np.all(out_rgb[~valid] == 0)
    assert valid[:30].all() and not valid[30:].any()


def test_photometric_changes_only_the_image() -> None:
    rgb = np.full((32, 32, 3), 128, dtype=np.uint8)
    config = aug.AugmentConfig(blur_prob=1.0, jpeg_prob=1.0)

    out = aug.photometric(rgb, config, np.random.default_rng(3))

    assert out.shape == rgb.shape
    assert out.dtype == np.uint8


def test_flip_is_applied_to_both() -> None:
    rgb, mask = coded_pair(h=64, w=64)
    flip_always = aug.AugmentConfig(crop_size=64, scale_range=(1.0, 1.0), hflip_prob=1.0)

    out_rgb, out_mask, _ = aug.augment(
        rgb, mask, 64, flip_always, np.random.default_rng(0), photometric_ops=False
    )

    np.testing.assert_array_equal(out_mask, mask[:, ::-1])
    np.testing.assert_array_equal(out_rgb, rgb[:, ::-1])
