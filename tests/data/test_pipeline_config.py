from pathlib import Path
from typing import Any

import yaml

from beltwatch.data.config import load_data_config

ROOT = Path(__file__).resolve().parents[2]


def _out_paths(stage: dict[str, Any]) -> set[str]:
    return {next(iter(out)) if isinstance(out, dict) else out for out in stage["outs"]}


def test_download_stage_outputs_match_data_config() -> None:
    stages = yaml.safe_load((ROOT / "dvc.yaml").read_text())["stages"]
    paths = load_data_config(ROOT / "configs/data.yaml").paths
    source = load_data_config(ROOT / "configs/data.yaml").source

    assert _out_paths(stages["download"]) == {
        paths.raw_dir.as_posix(),
        (paths.manifests_dir / f"{source.name}-download.json").as_posix(),
    }


def test_download_stage_depends_on_its_code() -> None:
    stage = yaml.safe_load((ROOT / "dvc.yaml").read_text())["stages"]["download"]

    for dep in stage["deps"]:
        assert (ROOT / dep).is_file(), dep
