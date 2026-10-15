import json
from pathlib import Path

import numpy as np
import pytest

from beltwatch.baselines.trivial import AllBackground
from beltwatch.data.dataset import Sample
from beltwatch.evaluation import compare, runner
from beltwatch.inference.predictor import Predictor
from tests.models.helpers import REMAP, Oracle

Images = tuple[list[Sample], dict[int, np.ndarray]]


def write_report(predictor: Predictor, samples: list[Sample], out: Path) -> Path:
    """What ``runner.main`` writes, without needing a data config."""
    results = runner.evaluate_images(predictor, samples, REMAP)
    report = runner.build_report(results, model=predictor.name, split="val", n_resamples=50)
    report |= {"split_manifest_id": "f" * 64, "limited_to": None, "temperature": 1.0}
    out.mkdir(parents=True)
    (out / "report.json").write_text(json.dumps(report))
    runner.write_confusions(results, out / "confusions.npz")
    return out


def test_saved_confusions_round_trip(images: Images, tmp_path: Path) -> None:
    samples, masks = images
    results = runner.evaluate_images(Oracle(masks), samples, REMAP)
    runner.write_confusions(results, tmp_path / "confusions.npz")

    saved = runner.load_confusions(tmp_path / "confusions.npz")

    assert saved.image_ids == [s.image_id for s in samples]
    assert saved.sequence_ids == [r.sequence_id for r in results]
    assert saved.frame_indices == [r.frame_index for r in results]
    np.testing.assert_array_equal(saved.confusions, np.stack([r.confusion for r in results]))
    expected = runner.bootstrap_groups(
        [r.sequence_id for r in results], [r.frame_index for r in results]
    )
    assert saved.groups() == expected


def test_paired_difference_between_two_reports(images: Images, tmp_path: Path) -> None:
    samples, masks = images
    a = write_report(Oracle(masks), samples, tmp_path / "oracle-val")
    b = write_report(AllBackground(), samples, tmp_path / "all-background-val")

    result = compare.compare_reports(a, b, n_resamples=50)

    assert (result["a"]["model"], result["b"]["model"]) == ("oracle", "all-background")
    diff = result["difference_a_minus_b"]
    by_sequence = diff["foreground_macro_iou"]["groups_by_sequence"]
    assert by_sequence["estimate"] == pytest.approx(1.0)
    assert by_sequence["n_groups"] == 3 and by_sequence["too_few_groups"] is True
    assert diff["iou_cardboard"]["groups_by_sequence"]["estimate"] == pytest.approx(1.0)
    assert diff["iou_metal"]["groups_by_sequence"]["estimate"] is None  # absent: undefined
    json.dumps(result, allow_nan=False)


def test_seed_averaged_difference(images: Images, tmp_path: Path) -> None:
    samples, masks = images
    oracle = write_report(Oracle(masks), samples, tmp_path / "oracle-val")
    background = write_report(AllBackground(), samples, tmp_path / "all-background-val")

    result = compare.compare_reports([oracle, background], [background], n_resamples=50)

    assert result["a"]["mean_over_reports"] == 2
    assert result["a"]["foreground_macro_iou"] == pytest.approx(0.5)
    assert len(result["a"]["reports"]) == 2 and result["b"]["model"] == "all-background"
    assert result["a"]["iou"]["metal"] is None  # absent: undefined, not averaged
    by_sequence = result["difference_a_minus_b"]["foreground_macro_iou"]["groups_by_sequence"]
    assert by_sequence["estimate"] == pytest.approx(0.5)
    json.dumps(result, allow_nan=False)


def test_reports_must_match(images: Images, tmp_path: Path) -> None:
    samples, masks = images
    a = write_report(Oracle(masks), samples, tmp_path / "a")
    fewer = write_report(Oracle(masks), samples[:5], tmp_path / "fewer")
    other = write_report(Oracle(masks), samples, tmp_path / "other")
    report = json.loads((other / "report.json").read_text())
    (other / "report.json").write_text(json.dumps(report | {"split_manifest_id": "e" * 64}))

    with pytest.raises(ValueError, match="n_images"):
        compare.compare_reports(a, fewer)
    with pytest.raises(ValueError, match="split_manifest_id"):
        compare.compare_reports(a, other)


def test_confusions_must_match_their_report(images: Images, tmp_path: Path) -> None:
    samples, masks = images
    a = write_report(Oracle(masks), samples, tmp_path / "a")
    b = write_report(AllBackground(), samples, tmp_path / "b")
    (b / "confusions.npz").replace(tmp_path / "moved.npz")
    with pytest.raises(FileNotFoundError, match="re-run"):
        compare.compare_reports(a, b)

    (tmp_path / "moved.npz").replace(a / "confusions.npz")  # all-background's, under oracle
    with pytest.raises(ValueError, match="does not match"):
        compare.compare_reports(a, b)


def test_evaluate_cli_saves_confusions_and_compare_cli_reads_them(
    tiny_data_config: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = ["--model", "all-background", "--split", "val", "--config", str(tiny_data_config)]
    for name in ("first", "second"):
        assert runner.main([*args, "--out-dir", str(tmp_path / name)]) == 0
    first, second = tmp_path / "first/all-background-val", tmp_path / "second/all-background-val"
    assert (first / "confusions.npz").exists()

    assert compare.main([str(first), str(second), "--n-resamples", "20"]) == 0

    out = tmp_path / "first/compare-all-background-vs-all-background-val.json"
    diff = json.loads(out.read_text())["difference_a_minus_b"]["foreground_macro_iou"]
    assert diff["groups_by_sequence"]["estimate"] == 0.0
    assert "foreground_macro_iou groups_by_sequence: +0.0000" in capsys.readouterr().out
