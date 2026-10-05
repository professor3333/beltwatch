import csv
import json
from pathlib import Path

import pytest

from beltwatch.data import splits as sp
from beltwatch.data.config import SplitsConfig, load_data_config

POLICY = SplitsConfig(
    policy="official-repaired",
    calibration_from_val_sequences=(),
    train_buffer_frames=100,
    exclude_train_duplicates_of_eval=True,
)
CAL_POLICY = POLICY.model_copy(update={"calibration_from_val_sequences": ("02",)})


def img(split: str, seq: str, frame: int, suffix: str = "") -> dict[str, str]:
    return {
        "image_id": f"{split}/{seq}_frame_{frame:06d}{suffix}",
        "split": split,
        "sequence_id": seq,
        "frame_index": str(frame),
    }


def pair(a: dict[str, str], b: dict[str, str]) -> dict[str, str]:
    return {"image_id_a": a["image_id"], "image_id_b": b["image_id"]}


def by_id(rows: list[sp.Assignment]) -> dict[str, sp.Assignment]:
    return {r.image_id: r for r in rows}


def test_calibration_takes_whole_val_sequences() -> None:
    images = [img("val", "01", 10), img("val", "02", 10), img("val", "02", 20)]

    rows = by_id(sp.assign_splits(images, [], CAL_POLICY))

    assert rows["val/01_frame_000010"].split == sp.VAL
    assert rows["val/02_frame_000010"].split == sp.CALIBRATION
    assert rows["val/02_frame_000020"].split == sp.CALIBRATION


def test_train_frames_near_eval_frames_are_excluded() -> None:
    images = [
        img("test", "09", 3000),
        img("train", "09", 2950),  # 50 frames from test: excluded
        img("train", "09", 3100),  # exactly at the buffer: excluded
        img("train", "09", 3101),  # just outside: kept
        img("train", "05", 3000),  # other sequence: kept
        img("val", "01", 10),
    ]

    rows = by_id(sp.assign_splits(images, [], POLICY))

    assert rows["train/09_frame_002950"].exclusion_reason == "within_buffer_of_eval(gap=50)"
    assert rows["train/09_frame_003100"].split == sp.EXCLUDED
    assert rows["train/09_frame_003101"].split == sp.TRAIN
    assert rows["train/05_frame_003000"].split == sp.TRAIN


def test_buffer_also_protects_calibration_frames() -> None:
    images = [img("val", "02", 991), img("train", "02", 1000), img("train", "02", 1200)]

    rows = by_id(sp.assign_splits(images, [], CAL_POLICY))

    assert rows["val/02_frame_000991"].split == sp.CALIBRATION

    assert rows["train/02_frame_001000"].split == sp.EXCLUDED
    assert rows["train/02_frame_001200"].split == sp.TRAIN


def test_train_duplicates_of_eval_are_excluded() -> None:
    train, test = img("train", "06", 10), img("test", "08", 99)
    rows = by_id(sp.assign_splits([train, test], [pair(test, train)], POLICY))

    assert rows[train["image_id"]].exclusion_reason == f"duplicate_of_eval({test['image_id']})"


def test_train_duplicate_exclusion_can_be_disabled() -> None:
    train, test = img("train", "06", 10), img("test", "08", 99)
    policy = POLICY.model_copy(update={"exclude_train_duplicates_of_eval": False})

    rows = by_id(sp.assign_splits([train, test], [pair(train, test)], policy))

    assert rows[train["image_id"]].split == sp.TRAIN


def test_val_duplicates_of_test_are_excluded_and_test_is_kept() -> None:
    val, test = img("val", "01", 10), img("test", "08", 5)

    rows = by_id(sp.assign_splits([val, test], [pair(test, val)], POLICY))

    assert rows[val["image_id"]].split == sp.EXCLUDED
    assert rows[val["image_id"]].exclusion_reason.startswith("duplicate_of_test")
    assert rows[test["image_id"]].split == sp.TEST


def test_within_train_duplicates_are_kept() -> None:
    a, b = img("train", "09", 2000), img("train", "09", 2000, "-2")

    rows = by_id(sp.assign_splits([a, b], [pair(a, b)], POLICY))

    assert {r.split for r in rows.values()} == {sp.TRAIN}


def test_unseen_sequence_slice() -> None:
    images = [img("train", "01", 10), img("test", "01", 5000), img("test", "08", 10)]

    rows = by_id(sp.assign_splits(images, [], POLICY))

    assert rows["test/08_frame_000010"].unseen_sequence is True
    assert rows["test/01_frame_005000"].unseen_sequence is False


def test_sequence_becomes_unseen_when_buffer_removes_all_its_train_frames() -> None:
    images = [img("train", "07", 10), img("test", "07", 50)]

    rows = by_id(sp.assign_splits(images, [], POLICY))

    assert rows["train/07_frame_000010"].split == sp.EXCLUDED
    assert rows["test/07_frame_000050"].unseen_sequence is True


def test_test_membership_never_changes() -> None:
    images = [img("test", "01", f) for f in range(0, 1000, 10)] + [img("train", "01", 5)]

    rows = sp.assign_splits(images, [], POLICY)

    assert sum(r.split == sp.TEST for r in rows) == 100


def test_unknown_calibration_sequence_raises() -> None:
    policy = POLICY.model_copy(update={"calibration_from_val_sequences": ("99",)})

    with pytest.raises(sp.SplitPolicyError, match="not present in val"):
        sp.assign_splits([img("val", "01", 1)], [], policy)


def test_unknown_official_split_raises() -> None:
    with pytest.raises(sp.SplitPolicyError, match="unknown official split"):
        sp.assign_splits([img("holdout", "01", 1)], [], POLICY)


def test_run_writes_csv_and_summary(tmp_path: Path) -> None:
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    images = [img("train", "01", 10), img("train", "01", 5000), img("val", "02", 10)]
    images += [img("test", "01", 50), img("test", "08", 1)]
    with (manifests / "fixture-images.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(images[0]))
        writer.writeheader()
        writer.writerows(images)
    (manifests / "fixture-duplicates.csv").write_text("image_id_a,image_id_b\n")
    repo = load_data_config(Path("configs/data.yaml"))
    config = repo.model_copy(
        update={
            "source": repo.source.model_copy(update={"name": "fixture"}),
            "splits": CAL_POLICY,
            "paths": repo.paths.model_copy(
                update={"manifests_dir": manifests, "splits_dir": tmp_path / "splits"}
            ),
        }
    )

    paths = sp.run(config)

    with paths["splits"].open() as fh:
        rows = {r["image_id"]: r for r in csv.DictReader(fh)}
    assert rows["train/01_frame_000010"]["split"] == "excluded"
    assert rows["train/01_frame_005000"]["split"] == "train"
    assert rows["val/02_frame_000010"]["split"] == "calibration"
    assert rows["test/08_frame_000001"]["unseen_sequence"] == "True"
    summary = json.loads(paths["summary"].read_text())
    assert summary["counts"] == {"train": 1, "val": 0, "calibration": 1, "test": 2, "excluded": 1}
    assert summary["excluded_by_reason"] == {"within_buffer_of_eval": 1}
    assert summary["unseen_sequences"] == ["08"]
    assert len(summary["split_manifest_id"]) == 64
    assert summary["policy"]["train_buffer_frames"] == 100


def test_repo_policy_is_valid() -> None:
    config = load_data_config(Path("configs/data.yaml"))
    assert config.splits.policy == "official-repaired"
    assert config.splits.calibration_from_val_sequences == ("02", "10")
