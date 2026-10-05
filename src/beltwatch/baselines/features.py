"""Handcrafted per-pixel features for the classical baseline.

Per RGB channel and at several Gaussian scales: smoothed intensity, gradient
magnitude (edges), and Hessian eigenvalues (texture), from scikit-image's
``multiscale_basic_features``, plus CIE Lab colour. Everything is local; there
is no learned representation or wide spatial context, which is exactly what the
baseline is meant to measure.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from skimage.color import rgb2lab
from skimage.feature import multiscale_basic_features


@dataclass(frozen=True)
class FeatureConfig:
    sigma_min: float = 1.0
    sigma_max: float = 16.0
    num_sigma: int = 5


def pixel_features(rgb: npt.NDArray[np.uint8], config: FeatureConfig) -> npt.NDArray[np.float32]:
    """``H x W x 3`` uint8 → ``H x W x F`` float32 features."""
    image = rgb.astype(np.float32) / 255.0
    multiscale = multiscale_basic_features(  # type: ignore[no-untyped-call]
        image,
        intensity=True,
        edges=True,
        texture=True,
        sigma_min=config.sigma_min,
        sigma_max=config.sigma_max,
        num_sigma=config.num_sigma,
        channel_axis=-1,
    )
    lab = rgb2lab(image) / np.array([100.0, 128.0, 128.0])
    return np.concatenate([multiscale, lab], axis=-1).astype(np.float32)
