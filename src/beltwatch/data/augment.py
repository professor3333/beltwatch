"""Training-time augmentation with identical geometry for image and mask.

Geometric operations (scale, crop, flip) are applied to the image and the mask
together, using nearest-neighbour resizing for the mask. Photometric operations
(brightness/contrast, blur, JPEG compression) touch only the image. Evaluation
never uses this module; it uses :func:`beltwatch.inference.preprocessing.prepare_image`.
"""

import io
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFilter

from beltwatch.inference.preprocessing import resize_image, resize_mask, resized_size
from beltwatch.labels import IGNORE_INDEX


@dataclass(frozen=True)
class AugmentConfig:
    crop_size: int = 512
    scale_range: tuple[float, float] = (0.75, 1.25)
    """Multiplier on the working long side before cropping."""
    hflip_prob: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    blur_prob: float = 0.2
    blur_radius: tuple[float, float] = (0.5, 1.5)
    jpeg_prob: float = 0.2
    jpeg_quality: tuple[int, int] = (40, 90)


def random_scale(
    rgb: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.uint8],
    long_side: int,
    scale_range: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    target = max(1, round(long_side * rng.uniform(*scale_range)))
    size = resized_size(rgb.shape[1], rgb.shape[0], target)
    return resize_image(rgb, size), resize_mask(mask, size)


def random_crop(
    rgb: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.uint8],
    crop_size: int,
    rng: np.random.Generator,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
    """Crop ``crop_size``² from image and mask; pad first if a side is too short.

    Returns the cropped image, mask (padding = ignore), and a validity map that is
    False on padding so image padding can be neutralized after normalization.
    """
    h, w = mask.shape
    ph, pw = max(crop_size, h), max(crop_size, w)
    if (ph, pw) != (h, w):
        padded_rgb = np.zeros((ph, pw, 3), dtype=np.uint8)
        padded_rgb[:h, :w] = rgb
        padded_mask = np.full((ph, pw), IGNORE_INDEX, dtype=np.uint8)
        padded_mask[:h, :w] = mask
        rgb, mask = padded_rgb, padded_mask
    valid = np.zeros((ph, pw), dtype=bool)
    valid[:h, :w] = True
    top = int(rng.integers(0, ph - crop_size + 1))
    left = int(rng.integers(0, pw - crop_size + 1))
    window = (slice(top, top + crop_size), slice(left, left + crop_size))
    return rgb[window], mask[window], valid[window]


def photometric(
    rgb: npt.NDArray[np.uint8], config: AugmentConfig, rng: np.random.Generator
) -> npt.NDArray[np.uint8]:
    img = rgb.astype(np.float32)
    contrast = rng.uniform(1 - config.contrast, 1 + config.contrast)
    brightness = rng.uniform(-config.brightness, config.brightness) * 255.0
    img = (img - img.mean()) * contrast + img.mean() + brightness
    out = np.clip(img, 0, 255).astype(np.uint8)

    if rng.random() < config.blur_prob:
        radius = float(rng.uniform(*config.blur_radius))
        out = np.asarray(Image.fromarray(out).filter(ImageFilter.GaussianBlur(radius)))
    if rng.random() < config.jpeg_prob:
        buf = io.BytesIO()
        quality = int(rng.integers(config.jpeg_quality[0], config.jpeg_quality[1] + 1))
        Image.fromarray(out).save(buf, format="JPEG", quality=quality)
        out = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
    return out


def augment(
    rgb: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.uint8],
    long_side: int,
    config: AugmentConfig,
    rng: np.random.Generator,
    *,
    photometric_ops: bool = True,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
    """Scale → crop → flip (image and mask together), then photometric (image only)."""
    rgb, mask = random_scale(rgb, mask, long_side, config.scale_range, rng)
    rgb, mask, valid = random_crop(rgb, mask, config.crop_size, rng)
    if rng.random() < config.hflip_prob:
        rgb, mask, valid = rgb[:, ::-1], mask[:, ::-1], valid[:, ::-1]
    rgb = np.ascontiguousarray(rgb)
    mask = np.ascontiguousarray(mask)
    valid = np.ascontiguousarray(valid)
    if photometric_ops:
        rgb = photometric(rgb, config, rng)
    return rgb, mask, valid
