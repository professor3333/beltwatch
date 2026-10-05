"""Load official ZeroWaste masks as BeltWatch class-ID masks."""

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

from beltwatch.labels import remap_mask


def load_source_remap(validation_summary: Path) -> dict[int, int]:
    """Read the source→BeltWatch mapping recorded by the validate stage."""
    summary = json.loads(validation_summary.read_text(encoding="utf-8"))
    return {int(k): int(v) for k, v in summary["source_to_beltwatch"].items()}


def load_mask(path: Path, source_to_internal: dict[int, int]) -> npt.NDArray[np.uint8]:
    """Decode a single-channel source mask and convert it to BeltWatch IDs."""
    with Image.open(path) as img:
        mask = np.asarray(img)
    if mask.ndim != 2:
        raise ValueError(f"{path}: expected a single-channel mask, got shape {mask.shape}")
    return remap_mask(mask, source_to_internal)
