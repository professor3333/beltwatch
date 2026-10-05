import io
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from beltwatch.inference import preprocessing as pp
from beltwatch.labels import IGNORE_INDEX

CONFIG = pp.PreprocessConfig()


def png_bytes(img: Image.Image, **kwargs: object) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", **kwargs)
    return buf.getvalue()


def test_load_rgb_keeps_channel_order() -> None:
    red = Image.new("RGB", (4, 2), (255, 0, 0))

    rgb = pp.load_rgb(png_bytes(red))

    assert rgb.shape == (2, 4, 3)
    assert rgb.dtype == np.uint8
    assert tuple(rgb[0, 0]) == (255, 0, 0)


@pytest.mark.parametrize("mode", ["L", "RGBA", "P"])
def test_load_rgb_converts_other_modes(mode: str) -> None:
    assert pp.load_rgb(png_bytes(Image.new(mode, (4, 2)))).shape == (2, 4, 3)


def test_load_rgb_applies_exif_orientation(tmp_path: Path) -> None:
    img = Image.new("RGB", (4, 2), (0, 0, 255))
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90° clockwise on display
    path = tmp_path / "rotated.jpg"
    img.save(path, exif=exif)

    assert pp.load_rgb(path).shape == (4, 2, 3)


def test_geometry_for_zerowaste_frames() -> None:
    geometry = pp.compute_geometry(1920, 1080, CONFIG)

    assert geometry.original_size == (1920, 1080)
    assert geometry.resized_size == (768, 432)
    assert geometry.padded_size == (768, 448)


def test_geometry_portrait_and_tiny_images() -> None:
    assert pp.compute_geometry(1080, 1920, CONFIG).resized_size == (432, 768)
    assert pp.compute_geometry(3, 3000, CONFIG).resized_size == (1, 768)


def test_normalize_uses_mean_and_std() -> None:
    mean_pixel = np.array(CONFIG.mean, dtype=np.float32) * 255
    rgb = np.broadcast_to(mean_pixel.round().astype(np.uint8), (2, 2, 3)).copy()

    chw = pp.normalize(rgb, CONFIG)

    assert chw.shape == (3, 2, 2)
    assert chw.dtype == np.float32
    assert np.abs(chw).max() < 0.02


def test_prepare_image_pads_with_normalized_zeros() -> None:
    rgb = np.full((1080, 1920, 3), 200, dtype=np.uint8)

    image, _ = pp.prepare_image(rgb, CONFIG)

    assert image.shape == (3, 448, 768)
    assert np.all(image[:, 432:, :] == 0.0)
    assert np.all(image[:, :432, :] > 0.0)


def test_prepare_mask_pads_with_ignore_and_preserves_class_ids() -> None:
    rng = np.random.default_rng(0)
    mask = rng.choice(np.array([0, 1, 3], dtype=np.uint8), size=(1080, 1920))
    geometry = pp.compute_geometry(1920, 1080, CONFIG)

    out = pp.prepare_mask(mask, geometry)

    assert out.shape == (448, 768)
    assert set(np.unique(out[:432])) <= {0, 1, 3}
    assert np.all(out[432:] == IGNORE_INDEX)


def test_prepare_mask_rejects_mismatched_size() -> None:
    geometry = pp.compute_geometry(1920, 1080, CONFIG)
    with pytest.raises(ValueError, match="mask size"):
        pp.prepare_mask(np.zeros((10, 10), dtype=np.uint8), geometry)


def test_restore_probabilities_round_trips_a_label_map() -> None:
    # Blocky label map at original resolution, one class per vertical band.
    labels = np.repeat(np.arange(5, dtype=np.int64), 384)[None, :].repeat(1080, axis=0)
    geometry = pp.compute_geometry(1920, 1080, CONFIG)
    small = pp.prepare_mask(labels.astype(np.uint8), geometry).astype(np.int64)
    small[small == IGNORE_INDEX] = 0
    logits = torch.nn.functional.one_hot(torch.from_numpy(small), 5).permute(2, 0, 1) * 10.0

    probs = pp.restore_probabilities(logits.float(), geometry)

    assert probs.shape == (5, 1080, 1920)
    assert torch.allclose(probs.sum(0), torch.ones(1080, 1920), atol=1e-5)
    agreement = (probs.argmax(0).numpy() == labels).mean()
    assert agreement > 0.995  # only band edges may differ after bilinear resizing


def test_restore_probabilities_temperature_softens() -> None:
    geometry = pp.compute_geometry(64, 32, pp.PreprocessConfig(long_side=64))
    logits = torch.zeros(5, 32, 64)
    logits[1] = 4.0

    sharp = pp.restore_probabilities(logits, geometry)[1].mean()
    soft = pp.restore_probabilities(logits, geometry, temperature=2.0)[1].mean()

    assert soft < sharp


def test_restore_probabilities_validates_shape() -> None:
    geometry = pp.compute_geometry(1920, 1080, CONFIG)
    with pytest.raises(ValueError, match="padded"):
        pp.restore_probabilities(torch.zeros(5, 432, 768), geometry)
