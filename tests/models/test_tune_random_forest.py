import json
from pathlib import Path

import pytest
import yaml

from beltwatch.baselines import tune_random_forest as tuner
from beltwatch.baselines.random_forest import RandomForestSegmenter
from beltwatch.evaluation import calibration, compare, runner
from beltwatch.provenance import file_digest

REPO = Path(__file__).resolve().parents[2]


def tiny_grid(tmp_path: Path, **overrides: object) -> Path:
    grid = {
        "version": "rf-tuning-test",
        "forests": [
            {"long_side": 48, "pixels_per_class": 40, "n_estimators": 5, "max_depth": 6},
            {"long_side": 48, "pixels_per_class": 80, "n_estimators": 5, "max_depth": 6},
        ],
        "prior_powers": [0.0, 1.0],
        **overrides,
    }
    path = tmp_path / "grid.yaml"
    path.write_text(yaml.safe_dump(grid))
    return path


def test_tuning_scores_every_candidate_on_val_and_keeps_the_best(
    tiny_data_config: Path, tmp_path: Path
) -> None:
    work, out, reports = tmp_path / "work", tmp_path / "rf", tmp_path / "reports"

    record = tuner.tune(tiny_grid(tmp_path), tiny_data_config, work, out, reports)

    names = [c["name"] for c in record["candidates"]]
    assert names == ["forest0-prior0", "forest0-prior1", "forest1-prior0", "forest1-prior1"]
    best = max(record["candidates"], key=lambda c: c["val_foreground_macro_iou"])
    assert record["selected"] == best["name"] and record["split"] == "val"
    assert json.loads((out / "tuning.json").read_text())["selected"] == best["name"]
    selected = RandomForestSegmenter.load(out)
    assert selected.config.prior_power == best["prior_power"]
    assert json.loads((out / "metadata.json").read_text())["tuning"]["candidate"] == best["name"]
    final = json.loads((reports / "random-forest-val/report.json").read_text())
    assert final["segmentation"]["foreground_macro_iou"] == best["val_foreground_macro_iou"]
    assert (reports / "random-forest-val/confusions.npz").exists()
    assert not list(work.glob("forest*/model.joblib"))  # only the selected forest is kept


def test_selected_forest_reproduces_its_report_through_the_cli(
    tiny_data_config: Path, tmp_path: Path
) -> None:
    out, reports = tmp_path / "rf", tmp_path / "reports"
    tuner.tune(tiny_grid(tmp_path), tiny_data_config, tmp_path / "work", out, reports)

    args = ["--model", "random-forest", "--model-dir", str(out), "--split", "val"]
    assert runner.main([*args, "--config", str(tiny_data_config), "--out-dir", str(tmp_path)]) == 0

    result = compare.compare_reports(tmp_path / "random-forest-val", reports / "random-forest-val")
    diff = result["difference_a_minus_b"]["foreground_macro_iou"]["groups_by_sequence"]
    assert diff["estimate"] == 0.0

    # Both reports trace back to the same weights: the tuned candidate and its copy in --out.
    digest = file_digest(out / "model.joblib")
    assert result["a"]["provenance"]["model_sha256"] == digest
    assert result["b"]["provenance"]["model_sha256"] == digest
    assert result["a"]["provenance"]["model_file"] == str(out / "model.joblib")

    check = ["--model", "random-forest", "--model-dir", str(out), "--split", "val"]
    cal_out = tmp_path / "calibration"
    extra = ["--temperature", "1.0", "--config", str(tiny_data_config), "--out", str(cal_out)]
    assert calibration.main([*check, *extra]) == 0
    saved = json.loads((cal_out / "calibration.json").read_text())
    assert saved["provenance"]["model_sha256"] == digest


def test_rerun_reuses_fitted_forests_and_reports(
    tiny_data_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grid, work = tiny_grid(tmp_path), tmp_path / "work"
    first = tuner.tune(grid, tiny_data_config, work, tmp_path / "rf", tmp_path / "r", keep_all=True)

    def no_refit(*args: object, **kwargs: object) -> None:
        raise AssertionError("refitted a forest that was already saved")

    monkeypatch.setattr(tuner, "train_and_save", no_refit)
    monkeypatch.setattr(tuner, "evaluate_images", no_refit)
    second = tuner.tune(grid, tiny_data_config, work, tmp_path / "rf", tmp_path / "r")

    assert second["candidates"] == first["candidates"]


def test_grid_is_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        tuner.load_tuning_config(tiny_grid(tmp_path, prior_powers=[-1.0]))
    with pytest.raises(ValueError, match="extra"):
        tuner.load_tuning_config(tiny_grid(tmp_path, forests=[{"class_weight": "balanced"}]))


def test_repository_grid_loads() -> None:
    config = tuner.load_tuning_config(REPO / "configs/random_forest_tuning.yaml")

    assert config.forests[0].pixels_per_class == 200  # the untuned baseline is a candidate
    assert 0.0 in config.prior_powers
