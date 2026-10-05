import json
from pathlib import Path

import pytest

from beltwatch.release.bundle import (
    ReleaseError,
    activate,
    active_version,
    build_release,
    load_active_release,
    load_release,
)
from tests.integration.conftest import POLICY


def test_build_and_load_verified_release(tmp_path: Path) -> None:
    model = tmp_path / "best.pt"
    model.write_bytes(b"weights")

    release = build_release(
        tmp_path / "r", version="v1", kind="unet", model_path=model, review_policy=POLICY
    )

    assert release.manifest.model.sha256 is not None
    assert release.manifest.review_policy.version == POLICY.version
    assert load_release(tmp_path / "r" / "v1").version == "v1"


def test_releases_are_immutable(tmp_path: Path) -> None:
    build_release(
        tmp_path, version="v1", kind="all-background", model_path=None, review_policy=POLICY
    )
    with pytest.raises(ReleaseError, match="immutable"):
        build_release(
            tmp_path, version="v1", kind="all-background", model_path=None, review_policy=POLICY
        )


def test_checksum_mismatch_is_detected(tmp_path: Path) -> None:
    model = tmp_path / "best.pt"
    model.write_bytes(b"weights")
    build_release(tmp_path / "r", version="v1", kind="unet", model_path=model, review_policy=POLICY)
    (tmp_path / "r" / "v1" / "model.pt").write_bytes(b"tampered")

    with pytest.raises(ReleaseError, match="checksum"):
        load_release(tmp_path / "r" / "v1")


def test_preprocessing_version_mismatch_is_refused(tmp_path: Path) -> None:
    build_release(
        tmp_path, version="v1", kind="all-background", model_path=None, review_policy=POLICY
    )
    manifest = tmp_path / "v1" / "manifest.json"
    data = json.loads(manifest.read_text())
    data["preprocessing_version"] = "pp-0"
    manifest.write_text(json.dumps(data))

    with pytest.raises(ReleaseError, match="preprocessing"):
        load_release(tmp_path / "v1")


def test_activate_and_roll_back_keep_history(tmp_path: Path) -> None:
    for v in ("v1", "v2"):
        build_release(
            tmp_path, version=v, kind="all-background", model_path=None, review_policy=POLICY
        )
    activate(tmp_path, "v1", "first")
    activate(tmp_path, "v2", "upgrade")
    activate(tmp_path, "v1", "rollback: regression")

    pointer = json.loads((tmp_path / "active.json").read_text())
    assert active_version(tmp_path) == "v1"
    assert pointer["replaces"] == "v2"
    assert [h["version"] for h in pointer["history"]] == ["v1", "v2"]
    assert load_active_release(tmp_path).version == "v1"


def test_missing_release_and_pointer(tmp_path: Path) -> None:
    with pytest.raises(ReleaseError, match="manifest"):
        load_release(tmp_path / "nope")
    with pytest.raises(ReleaseError, match=r"active\.json"):
        active_version(tmp_path)


def _calibration_report(tmp_path: Path, split: str = "calibration", fitted: bool = True) -> Path:
    bins = {"ece": 0.1}
    report = {
        "split": split,
        "fitted": fitted,
        "temperature": 1.7,
        "n_pixels": 1000,
        "n_images": 10,
        "before": {"overall": bins, "foreground": {"ece": 0.2}},
        "after": {"overall": {"ece": 0.02}, "foreground": {"ece": 0.05}},
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(report))
    return path


def test_release_takes_temperature_from_calibration_report(tmp_path: Path) -> None:
    release = build_release(
        tmp_path / "r",
        version="v1",
        kind="all-background",
        model_path=None,
        review_policy=POLICY,
        calibration_report=_calibration_report(tmp_path),
    )
    assert release.manifest.temperature == 1.7
    assert "0.1000 -> 0.0200" in release.manifest.calibration_note
    assert (tmp_path / "r" / "v1" / "calibration.json").is_file()


def test_release_refuses_calibration_not_fitted_on_calibration_split(tmp_path: Path) -> None:
    with pytest.raises(ReleaseError, match="calibration split"):
        build_release(
            tmp_path / "r",
            version="v1",
            kind="all-background",
            model_path=None,
            review_policy=POLICY,
            calibration_report=_calibration_report(tmp_path, split="val"),
        )


def test_release_refuses_calibration_that_hurts_foreground(tmp_path: Path) -> None:
    report = _calibration_report(tmp_path)
    data = json.loads(report.read_text())
    data["after"]["foreground"]["ece"] = 0.3  # worse than 0.2 before
    report.write_text(json.dumps(data))

    with pytest.raises(ReleaseError, match="worsens foreground"):
        build_release(
            tmp_path / "r",
            version="v1",
            kind="all-background",
            model_path=None,
            review_policy=POLICY,
            calibration_report=report,
        )
    release = build_release(
        tmp_path / "r",
        version="v2",
        kind="all-background",
        model_path=None,
        review_policy=POLICY,
        calibration_report=report,
        allow_foreground_calibration_regression=True,
    )
    assert "0.2000 -> 0.3000" in release.manifest.calibration_note
