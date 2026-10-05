"""Validated data configuration (``configs/data.yaml``)."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, PositiveInt


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SourceConfig(_Frozen):
    """A pinned, checksummed source release."""

    name: str
    zenodo_record: PositiveInt
    version: str
    doi: str
    filename: str = Field(pattern=r"^[^/\\]+$")
    url: str = Field(pattern=r"^https://")
    size_bytes: PositiveInt
    md5: str = Field(pattern=r"^[0-9a-f]{32}$")


class LayoutConfig(_Frozen):
    """Directory layout inside the extracted archive."""

    root_subdir: str
    splits: tuple[str, ...] = Field(min_length=1)
    images_subdir: str
    masks_subdir: str
    annotations_file: str


class DuplicatesConfig(_Frozen):
    """Perceptual-hash settings for the duplicate audit."""

    hash_size: int = Field(ge=4, le=8)
    phash_max_hamming: int = Field(ge=0, le=64)


class SplitsConfig(_Frozen):
    """How the released splits are turned into BeltWatch's evaluation splits."""

    policy: Literal["official-repaired"]
    calibration_from_val_sequences: tuple[str, ...]
    train_buffer_frames: int = Field(ge=0)
    exclude_train_duplicates_of_eval: bool


class PathsConfig(_Frozen):
    downloads_dir: Path
    raw_dir: Path
    manifests_dir: Path
    splits_dir: Path


class DataConfig(_Frozen):
    source: SourceConfig
    layout: LayoutConfig
    duplicates: DuplicatesConfig
    splits: SplitsConfig
    paths: PathsConfig


def load_data_config(path: Path) -> DataConfig:
    with path.open(encoding="utf-8") as fh:
        return DataConfig.model_validate(yaml.safe_load(fh))
