"""Image preprocessing shared by training, evaluation, and serving.

This module is the **only** implementation of BeltWatch's input pipeline.
Training, evaluation, and the inference worker all import it, so a model sees
identically prepared pixels everywhere. Any change to its behaviour must bump
:data:`PREPROCESSING_VERSION`, which is recorded in model release bundles and
the inference cache key.

Pipeline: decode → explicit RGB (EXIF orientation applied) → resize so the long
side equals ``long_side`` (bilinear for images, nearest for masks) → normalize
with ImageNet statistics → pad bottom/right to a multiple of ``pad_multiple``
(normalized zeros for images, :data:`~beltwatch.labels.IGNORE_INDEX` for masks).
Predictions are mapped back with :func:`restore_probabilities`, which crops the
padding and resizes logits bilinearly to the original image size.
"""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import numpy.typing as npt
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from beltwatch.labels import IGNORE_INDEX

PREPROCESSING_VERSION: Final = "pp-1"

IMAGENET_MEAN: Final = (0.485, 0.456, 0.406)
IMAGENET_STD: Final = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PreprocessConfig:
    long_side: int = 768
    pad_multiple: int = 32
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD


@dataclass(frozen=True)
class Geometry:
    """Sizes as ``(width, height)`` at each step, for mapping predictions back."""

    original_size: tuple[int, int]
    resized_size: tuple[int, int]
    padded_size: tuple[int, int]


def load_rgb(source: Path | bytes) -> npt.NDArray[np.uint8]:
    """Decode an image as ``HxWx3`` RGB uint8, applying EXIF orientation."""
    fh = io.BytesIO(source) if isinstance(source, bytes) else source.open("rb")
    with fh, Image.open(fh) as img:
        upright = ImageOps.exif_transpose(img)
        return np.array(upright.convert("RGB"), dtype=np.uint8)


def resized_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    """Size with the long side scaled to ``long_side``, preserving aspect ratio."""
    scale = long_side / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def padded_size(width: int, height: int, multiple: int) -> tuple[int, int]:
    return -(-width // multiple) * multiple, -(-height // multiple) * multiple


def resize_image(rgb: npt.NDArray[np.uint8], size: tuple[int, int]) -> npt.NDArray[np.uint8]:
    if (rgb.shape[1], rgb.shape[0]) == size:
        return rgb
    return np.asarray(Image.fromarray(rgb).resize(size, Image.Resampling.BILINEAR))


def resize_mask(mask: npt.NDArray[np.uint8], size: tuple[int, int]) -> npt.NDArray[np.uint8]:
    """Nearest-neighbour resize: never invents class IDs that were not present."""
    if (mask.shape[1], mask.shape[0]) == size:
        return mask
    return np.asarray(Image.fromarray(mask).resize(size, Image.Resampling.NEAREST))


def normalize(rgb: npt.NDArray[np.uint8], config: PreprocessConfig) -> npt.NDArray[np.float32]:
    """``HxWx3`` uint8 → ``3xHxW`` float32 standardized with the configured mean/std."""
    mean = np.asarray(config.mean, dtype=np.float32) * 255.0
    std = np.asarray(config.std, dtype=np.float32) * 255.0
    chw: npt.NDArray[np.float32] = ((rgb.astype(np.float32) - mean) / std).transpose(2, 0, 1)
    return np.ascontiguousarray(chw)


def pad_chw(
    array: npt.NDArray[np.float32], size: tuple[int, int], fill: float = 0.0
) -> npt.NDArray[np.float32]:
    width, height = size
    _, h, w = array.shape
    if (w, h) == size:
        return array
    out = np.full((array.shape[0], height, width), fill, dtype=array.dtype)
    out[:, :h, :w] = array
    return out


def pad_mask(
    mask: npt.NDArray[np.uint8], size: tuple[int, int], fill: int = IGNORE_INDEX
) -> npt.NDArray[np.uint8]:
    width, height = size
    h, w = mask.shape
    if (w, h) == size:
        return mask
    out = np.full((height, width), fill, dtype=np.uint8)
    out[:h, :w] = mask
    return out


def compute_geometry(width: int, height: int, config: PreprocessConfig) -> Geometry:
    resized = resized_size(width, height, config.long_side)
    return Geometry((width, height), resized, padded_size(*resized, config.pad_multiple))


def prepare_image(
    rgb: npt.NDArray[np.uint8], config: PreprocessConfig
) -> tuple[npt.NDArray[np.float32], Geometry]:
    """Full-frame model input (``3xHxW`` float32) and the geometry to undo it."""
    geometry = compute_geometry(rgb.shape[1], rgb.shape[0], config)
    tensor = normalize(resize_image(rgb, geometry.resized_size), config)
    return pad_chw(tensor, geometry.padded_size), geometry


def prepare_mask(mask: npt.NDArray[np.uint8], geometry: Geometry) -> npt.NDArray[np.uint8]:
    """Resize (nearest) and pad a BeltWatch-ID mask to match :func:`prepare_image`."""
    if (mask.shape[1], mask.shape[0]) != geometry.original_size:
        raise ValueError(f"mask size {mask.shape[::-1]} != image size {geometry.original_size}")
    return pad_mask(resize_mask(mask, geometry.resized_size), geometry.padded_size)


def restore_probabilities(
    logits: torch.Tensor, geometry: Geometry, temperature: float = 1.0
) -> torch.Tensor:
    """Map ``CxHxW`` logits on the padded grid to ``CxHxW`` probabilities at original size.

    Padding is cropped, logits are resized bilinearly, divided by ``temperature``
    (from calibration), then softmaxed over classes.
    """
    if logits.ndim != 3:
        raise ValueError(f"expected CxHxW logits, got shape {tuple(logits.shape)}")
    if (logits.shape[2], logits.shape[1]) != geometry.padded_size:
        raise ValueError(
            f"logit size {(logits.shape[2], logits.shape[1])} != padded {geometry.padded_size}"
        )
    rw, rh = geometry.resized_size
    ow, oh = geometry.original_size
    cropped = logits[:, :rh, :rw].unsqueeze(0).float()
    restored = F.interpolate(cropped, size=(oh, ow), mode="bilinear", align_corners=False)
    return torch.softmax(restored[0] / temperature, dim=0)
