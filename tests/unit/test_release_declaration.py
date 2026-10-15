import json
from pathlib import Path

import pytest

from beltwatch.release import declaration as decl

DECLARATION = Path("configs/releases/beltwatch-0.1.0.yaml")


def test_repo_declaration_matches_the_code() -> None:
    d = decl.load_declaration(DECLARATION)
    assert d.version == DECLARATION.stem
    assert decl.check(d) == []  # review policy and preprocessing versions


def test_wrong_weights_are_refused(tmp_path: Path) -> None:
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not the declared weights")
    problems = decl.check(decl.load_declaration(DECLARATION), model=weights)
    assert len(problems) == 1 and "not the declared weights" in problems[0]


def _calibration(tmp_path: Path, temperature: float, sha256: str) -> Path:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {
                "fitted": True,
                "split": "calibration",
                "fit_scope": "all",
                "temperature": temperature,
                "provenance": {"model_sha256": sha256},
            }
        )
    )
    return path


def test_calibration_must_reproduce_the_temperature(tmp_path: Path) -> None:
    d = decl.load_declaration(DECLARATION)
    assert decl.check(d, calibration=_calibration(tmp_path, 1.04, d.model.sha256)) == []
    problems = decl.check(d, calibration=_calibration(tmp_path, 1.05, d.model.sha256))
    assert any("does not reproduce" in p for p in problems)


@pytest.mark.parametrize(("score", "ok"), [(0.3185608, True), (0.3195, False)])
def test_val_score_must_reproduce(tmp_path: Path, score: float, ok: bool) -> None:
    d = decl.load_declaration(DECLARATION)
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "split": "val",
                "limited_to": None,
                "temperature": 1.04,
                "provenance": {"model_sha256": d.model.sha256},
                "segmentation": {"foreground_macro_iou": score},
            }
        )
    )
    assert (decl.check(d, val_report=report) == []) is ok


def test_a_completed_test_run_is_refused(tmp_path: Path) -> None:
    d = decl.load_declaration(DECLARATION)
    test_dir = tmp_path / "test-reports"
    (test_dir / "release").mkdir(parents=True)  # a stopped session may finish
    assert decl.check(d, test_dir=test_dir) == []
    (test_dir / decl.TEST_COMPLETE).write_text("{}")
    assert decl.check(d, test_dir=test_dir) != []


def test_unknown_fields_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.yaml"
    path.write_text(DECLARATION.read_text() + "\nsurprise: 1\n")
    with pytest.raises(ValueError, match="surprise"):
        decl.load_declaration(path)
