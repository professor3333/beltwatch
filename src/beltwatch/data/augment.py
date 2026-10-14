"""Training-time augmentation with identical geometry for image and mask.

Geometric operations (scale, crop, flip) are applied to the image and the mask
together, using nearest-neighbour resizing for the mask. Photometric operations
(brightness/contrast, blur, JPEG compression) touch only the image. Evaluation
never uses this module; it uses :func:`beltwatch.inference.preprocessing.prepare_image`.

Material-aware crop sampling (``material_crop_prob`` > 0): with that probability
a crop is placed to contain a pixel of one target material present in the
image, chosen uniformly among the materials present, so a rare material gets
the same chance as a common one. The pixel lands anywhere in the crop, not
only at its centre. The remaining crops are placed uniformly, which keeps
ordinary and background-heavy scenes in the mix. At 0 the random draws are the
same as before the option existed.
"""

import io
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFilter

from beltwatch.inference.preprocessing import resize_image, resize_mask, resized_size
from beltwatch.labels import IGNORE_INDEX, TARGET_CLASS_IDS


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
    material_crop_prob: float = 0.0
    """Share of crops placed on a target material; 0 is ordinary uniform cropping."""

    def __post_init__(self) -> None:
        if not 0.0 <= self.material_crop_prob <= 1.0:
            raise ValueError("material_crop_prob must be between 0 and 1")


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


def material_crop_origin(
    mask: npt.NDArray[np.uint8], crop_size: int, rng: np.random.Generator
) -> tuple[int, int] | None:
    """Top-left corner of a crop containing a random pixel of a present material.

    ``None`` if the mask holds no target material.
    """
    present = [c for c in TARGET_CLASS_IDS if (mask == c).any()]
    if not present:
        return None
    ys, xs = np.nonzero(mask == present[int(rng.integers(len(present)))])
    i = int(rng.integers(len(ys)))
    h, w = mask.shape
    origin = []
    for pos, size in ((int(ys[i]), h), (int(xs[i]), w)):
        low, high = max(0, pos - crop_size + 1), min(pos, size - crop_size)
        origin.append(int(rng.integers(low, high + 1)))
    return origin[0], origin[1]


def random_crop(
    rgb: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.uint8],
    crop_size: int,
    rng: np.random.Generator,
    material_crop_prob: float = 0.0,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], npt.NDArray[np.bool_]]:
    """Crop ``crop_size``² from image and mask; pad first if a side is too short.

    Returns the cropped image, mask (padding = ignore), and a validity map that is
    False on padding so image padding can be neutralized after normalization.
    With probability ``material_crop_prob`` the crop contains a target material
    (see :func:`material_crop_origin`).
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
    origin = None
    if material_crop_prob > 0 and rng.random() < material_crop_prob:
        origin = material_crop_origin(mask, crop_size, rng)
    if origin is None:
        origin = (
            int(rng.integers(0, ph - crop_size + 1)),
            int(rng.integers(0, pw - crop_size + 1)),
        )
    top, left = origin
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
    rgb, mask, valid = random_crop(rgb, mask, config.crop_size, rng, config.material_crop_prob)
    if rng.random() < config.hflip_prob:
        rgb, mask, valid = rgb[:, ::-1], mask[:, ::-1], valid[:, ::-1]
    rgb = np.ascontiguousarray(rgb)
    mask = np.ascontiguousarray(mask)
    valid = np.ascontiguousarray(valid)
    if photometric_ops:
        rgb = photometric(rgb, config, rng)
    return rgb, mask, valid
