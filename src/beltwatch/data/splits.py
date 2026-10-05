"""Assign BeltWatch evaluation splits with the "official-repaired" policy.

Usage::

    uv run python -m beltwatch.data.splits [--config configs/data.yaml]

Starting from the release's own train/val/test (so results stay comparable
with published work), this stage:

1. Moves whole val sequences listed in ``splits.calibration_from_val_sequences``
   to a grouped **calibration** holdout used only for temperature scaling.
2. Treats val, calibration, and test as protected evaluation groups. A train
   image is **excluded** if it lies within ``splits.train_buffer_frames`` frames
   of any evaluation frame in the same sequence, or (optionally) if it is an
   exact or near-duplicate of any evaluation image.
3. Excludes a val or calibration image that duplicates a test image, so model
   selection never sees near-copies of test frames.
4. Never removes or reassigns a test image: the locked test set is exactly the
   release's valid test images.
5. Marks test images whose sequence has no remaining train images as the
   ``unseen_sequence`` slice: the closest this dataset offers to an unseen
   recording.

What this supports claiming: performance on *held-out time windows of the same
recordings*, plus a small unseen-recording slice. It does not support claims
about other facilities or cameras.

Outputs (git-tracked): ``<splits_dir>/<name>-splits.csv`` with one row per
valid image, and ``<name>-splits.json`` with counts, parameters, and the split
manifest ID (SHA-256 of the CSV) used for lineage.
"""

import argparse
import bisect
import csv
import hashlib
import json
import logging
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from beltwatch import __version__
from beltwatch.data.config import DataConfig, SplitsConfig, load_data_config

log = logging.getLogger(__name__)

TRAIN: Final = "train"
VAL: Final = "val"
CALIBRATION: Final = "calibration"
TEST: Final = "test"
EXCLUDED: Final = "excluded"
EVAL_SPLITS: Final = frozenset({VAL, CALIBRATION, TEST})

SPLIT_FIELDS = (
    "image_id",
    "official_split",
    "split",
    "exclusion_reason",
    "sequence_id",
    "frame_index",
    "unseen_sequence",
)


class SplitPolicyError(RuntimeError):
    """The inputs or configuration cannot produce a valid split."""


@dataclass
class Assignment:
    image_id: str
    official_split: str
    sequence_id: str | None
    frame_index: int | None
    split: str
    exclusion_reason: str = ""
    unseen_sequence: bool = False


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def assign_splits(
    images: Sequence[Mapping[str, str]],
    duplicate_pairs: Sequence[Mapping[str, str]],
    policy: SplitsConfig,
) -> list[Assignment]:
    """Apply the official-repaired policy to validated manifest rows."""
    official = {TRAIN, VAL, TEST}
    rows: list[Assignment] = []
    for row in images:
        if row["split"] not in official:
            raise SplitPolicyError(f"{row['image_id']}: unknown official split {row['split']!r}")
        rows.append(
            Assignment(
                image_id=row["image_id"],
                official_split=row["split"],
                sequence_id=row["sequence_id"] or None,
                frame_index=int(row["frame_index"]) if row["frame_index"] else None,
                split=row["split"],
            )
        )

    val_sequences = {r.sequence_id for r in rows if r.official_split == VAL}
    missing = sorted(set(policy.calibration_from_val_sequences) - val_sequences)
    if missing:
        raise SplitPolicyError(f"calibration sequences not present in val: {missing}")
    calibration_sequences = set(policy.calibration_from_val_sequences)
    for r in rows:
        if r.official_split == VAL and r.sequence_id in calibration_sequences:
            r.split = CALIBRATION

    by_id = {r.image_id: r for r in rows}

    # Duplicates of test inside val/calibration, then train duplicates of any eval image.
    for pair in duplicate_pairs:
        a, b = by_id.get(pair["image_id_a"]), by_id.get(pair["image_id_b"])
        if a is None or b is None:
            continue
        for x, y in ((a, b), (b, a)):
            if x.split in {VAL, CALIBRATION} and y.split == TEST:
                x.split, x.exclusion_reason = EXCLUDED, f"duplicate_of_test({y.image_id})"
    if policy.exclude_train_duplicates_of_eval:
        for pair in duplicate_pairs:
            a, b = by_id.get(pair["image_id_a"]), by_id.get(pair["image_id_b"])
            if a is None or b is None:
                continue
            for x, y in ((a, b), (b, a)):
                if x.split == TRAIN and y.split in EVAL_SPLITS:
                    x.split, x.exclusion_reason = EXCLUDED, f"duplicate_of_eval({y.image_id})"

    # Temporal buffer around evaluation frames within each sequence.
    eval_frames: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        if r.split in EVAL_SPLITS and r.sequence_id is not None and r.frame_index is not None:
            eval_frames[r.sequence_id].append(r.frame_index)
    for sequence_frames in eval_frames.values():
        sequence_frames.sort()
    for r in rows:
        if r.split != TRAIN or r.sequence_id is None or r.frame_index is None:
            continue
        frames = eval_frames.get(r.sequence_id, [])
        if not frames:
            continue
        i = bisect.bisect_left(frames, r.frame_index)
        gap = min(abs(r.frame_index - frames[j]) for j in (i - 1, i) if 0 <= j < len(frames))
        if gap <= policy.train_buffer_frames:
            r.split, r.exclusion_reason = EXCLUDED, f"within_buffer_of_eval(gap={gap})"

    train_sequences = {r.sequence_id for r in rows if r.split == TRAIN}
    for r in rows:
        if r.split == TEST:
            r.unseen_sequence = r.sequence_id is not None and r.sequence_id not in train_sequences

    if any(r.official_split == TEST and r.split != TEST for r in rows):
        raise SplitPolicyError("policy changed test membership; the test set must stay locked")
    rows.sort(key=lambda r: r.image_id)
    return rows


def summarize(rows: Sequence[Assignment], policy: SplitsConfig) -> dict[str, Any]:
    counts = Counter(r.split for r in rows)
    reasons = Counter(r.exclusion_reason.split("(", 1)[0] for r in rows if r.exclusion_reason)
    sequences: dict[str, Counter[str]] = defaultdict(Counter)
    for r in rows:
        sequences[r.split][str(r.sequence_id)] += 1
    return {
        "policy": policy.model_dump(mode="json"),
        "counts": {s: counts.get(s, 0) for s in (TRAIN, VAL, CALIBRATION, TEST, EXCLUDED)},
        "excluded_by_reason": dict(sorted(reasons.items())),
        "excluded_by_official_split": dict(
            sorted(Counter(r.official_split for r in rows if r.split == EXCLUDED).items())
        ),
        "sequences_by_split": {
            split: dict(sorted(sequences[split].items()))
            for split in (TRAIN, VAL, CALIBRATION, TEST, EXCLUDED)
            if split in sequences
        },
        "unseen_sequence_test_images": sum(r.unseen_sequence for r in rows),
        "unseen_sequences": sorted({str(r.sequence_id) for r in rows if r.unseen_sequence}),
        "beltwatch_version": __version__,
    }


def _rows_for_csv(rows: Iterable[Assignment]) -> Iterable[dict[str, Any]]:
    for r in rows:
        yield {
            "image_id": r.image_id,
            "official_split": r.official_split,
            "split": r.split,
            "exclusion_reason": r.exclusion_reason,
            "sequence_id": r.sequence_id,
            "frame_index": r.frame_index,
            "unseen_sequence": r.unseen_sequence,
        }


def run(config: DataConfig) -> dict[str, Path]:
    manifests = config.paths.manifests_dir
    name = config.source.name
    images = _read_csv(manifests / f"{name}-images.csv")
    pairs = _read_csv(manifests / f"{name}-duplicates.csv")
    rows = assign_splits(images, pairs, config.splits)

    out_dir = config.paths.splits_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{name}-splits.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(SPLIT_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(_rows_for_csv(rows))

    summary = {
        "split_manifest_id": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        **summarize(rows, config.splits),
    }
    json_path = out_dir / f"{name}-splits.json"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info("splits %s; excluded %s", summary["counts"], summary["excluded_by_reason"])
    return {"splits": csv_path, "summary": json_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assign BeltWatch evaluation splits.")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_data_config(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
