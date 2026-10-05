"""Validated data configuration (``configs/data.yaml``)."""

from pathlib import Path

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


class PathsConfig(_Frozen):
    downloads_dir: Path
    raw_dir: Path
    manifests_dir: Path


class DataConfig(_Frozen):
    source: SourceConfig
    paths: PathsConfig


def load_data_config(path: Path) -> DataConfig:
    with path.open(encoding="utf-8") as fh:
        return DataConfig.model_validate(yaml.safe_load(fh))
