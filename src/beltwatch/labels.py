"""Single source of truth for BeltWatch class labels.

Every other module (data conversion, training, evaluation, serving) must import
class IDs and names from here rather than redefining them.

Source datasets may number or order their categories differently, so source
category IDs are never assumed to match ours. They are remapped explicitly by
name with :func:`build_source_remap`.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import numpy as np
import numpy.typing as npt

BACKGROUND: Final = 0
CARDBOARD: Final = 1
SOFT_PLASTIC: Final = 2
RIGID_PLASTIC: Final = 3
METAL: Final = 4

IGNORE_INDEX: Final = 255
"""Label for padding and genuinely undefined pixels; excluded from losses and metrics."""

CLASS_NAMES: Final[Mapping[int, str]] = MappingProxyType(
    {
        BACKGROUND: "background",
        CARDBOARD: "cardboard",
        SOFT_PLASTIC: "soft_plastic",
        RIGID_PLASTIC: "rigid_plastic",
        METAL: "metal",
    }
)
NUM_CLASSES: Final = len(CLASS_NAMES)

TARGET_CLASS_IDS: Final[tuple[int, ...]] = (CARDBOARD, SOFT_PLASTIC, RIGID_PLASTIC, METAL)
"""The four material classes averaged by foreground macro IoU. Background is excluded."""

_NAME_TO_ID: Final[Mapping[str, int]] = MappingProxyType(
    {name: class_id for class_id, name in CLASS_NAMES.items()}
)


def normalize_name(name: str) -> str:
    """Normalize a category name, e.g. ``"Soft Plastic"`` -> ``"soft_plastic"``."""
    return "_".join(name.strip().lower().replace("-", " ").split())


def build_source_remap(source_categories: Mapping[int, str]) -> dict[int, int]:
    """Map a source dataset's category IDs to BeltWatch class IDs by category name.

    ``source_categories`` maps source category ID -> source category name, e.g. the
    ``categories`` list of a COCO annotation file. Every one of the four target
    materials must appear exactly once. Unknown names raise rather than being
    silently dropped. A source "background" category, if present, maps to
    :data:`BACKGROUND`.
    """
    remap: dict[int, int] = {}
    seen: dict[int, int] = {}
    for source_id, source_name in source_categories.items():
        normalized = normalize_name(source_name)
        if normalized not in _NAME_TO_ID:
            raise ValueError(f"unknown source category {source_id}: {source_name!r}")
        class_id = _NAME_TO_ID[normalized]
        if class_id in seen:
            raise ValueError(
                f"source categories {seen[class_id]} and {source_id} both map to "
                f"{CLASS_NAMES[class_id]!r}"
            )
        seen[class_id] = source_id
        remap[source_id] = class_id

    missing = [CLASS_NAMES[c] for c in TARGET_CLASS_IDS if c not in seen]
    if missing:
        raise ValueError(f"source categories missing target classes: {missing}")
    return remap


def remap_mask(
    mask: npt.NDArray[np.integer], source_to_internal: Mapping[int, int]
) -> npt.NDArray[np.uint8]:
    """Convert a source-ID mask to BeltWatch IDs.

    ``source_to_internal`` covers the source *mask values* (category IDs plus
    whatever value the source uses for background). :data:`IGNORE_INDEX` passes
    through unchanged. Any other unmapped value raises, so a wrong mapping can't
    silently corrupt labels.
    """
    if mask.size and (mask.min() < 0 or mask.max() > 255):
        raise ValueError("mask values must be within 0..255")

    unmapped = np.uint16(256)
    lut = np.full(256, unmapped, dtype=np.uint16)
    for source_value, class_id in source_to_internal.items():
        if class_id not in CLASS_NAMES:
            raise ValueError(f"invalid BeltWatch class id {class_id}")
        lut[source_value] = class_id
    if IGNORE_INDEX not in source_to_internal:
        lut[IGNORE_INDEX] = IGNORE_INDEX

    out: npt.NDArray[np.uint16] = lut[mask]
    if np.any(out == unmapped):
        bad = sorted({int(v) for v in np.unique(mask[out == unmapped])})
        raise ValueError(f"mask contains unmapped source values: {bad}")
    return out.astype(np.uint8)
