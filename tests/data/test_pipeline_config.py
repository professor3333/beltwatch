from pathlib import Path
from typing import Any

import yaml

from beltwatch.data.config import load_data_config

ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_data_config(ROOT / "configs/data.yaml")
STAGES: dict[str, Any] = yaml.safe_load((ROOT / "dvc.yaml").read_text())["stages"]


def _out_paths(stage: dict[str, Any]) -> set[str]:
    return {next(iter(out)) if isinstance(out, dict) else out for out in stage["outs"]}


def _manifest(suffix: str) -> str:
    return (CONFIG.paths.manifests_dir / f"{CONFIG.source.name}-{suffix}").as_posix()


def test_download_stage_outputs_match_data_config() -> None:
    assert _out_paths(STAGES["download"]) == {
        CONFIG.paths.raw_dir.as_posix(),
        _manifest("download.json"),
    }


def test_validate_stage_reads_download_output_and_matches_config() -> None:
    stage = STAGES["validate"]

    assert CONFIG.paths.raw_dir.as_posix() in stage["deps"]
    assert _out_paths(stage) == {
        _manifest("images.csv"),
        _manifest("quarantine.csv"),
        _manifest("validation.json"),
    }


def test_stage_code_dependencies_exist() -> None:
    for name, stage in STAGES.items():
        for dep in stage["deps"]:
            if dep.startswith("src/"):
                assert (ROOT / dep).is_file(), f"{name}: {dep}"


def test_duplicates_stage_reads_manifest_and_matches_config() -> None:
    stage = STAGES["duplicates"]

    assert _manifest("images.csv") in stage["deps"]
    assert _out_paths(stage) == {
        _manifest("hashes.csv"),
        _manifest("duplicates.csv"),
        _manifest("duplicates.json"),
    }


def test_splits_stage_reads_audit_outputs_and_matches_config() -> None:
    stage = STAGES["splits"]
    splits_dir = CONFIG.paths.splits_dir

    assert {_manifest("images.csv"), _manifest("duplicates.csv")} <= set(stage["deps"])
    assert _out_paths(stage) == {
        (splits_dir / f"{CONFIG.source.name}-splits.csv").as_posix(),
        (splits_dir / f"{CONFIG.source.name}-splits.json").as_posix(),
    }
