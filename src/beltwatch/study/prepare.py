"""Select the usability-study image sets from the val split.

Usage::

    uv run python -m beltwatch.study.prepare --out data/study/<name> [--raw-dir DIR] [--seed 0]

Writes ``study.json`` (facilitator-only: ground truth and selection record)
and copies the selected frames to ``images/<study_id><ext>``, after checking
each against the SHA-256 recorded in the image manifest. The protocol,
including every rule applied here, is in docs/usability_study.md.

Ground truth comes from the per-class pixel counts that the validate stage
recorded in the image manifest, with the full frame as the inspection region.
Only the ``val`` split is used; the locked test split is never read.
"""

import argparse
import csv
import hashlib
import json
import logging
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from beltwatch.data.config import load_data_config
from beltwatch.evaluation.coverage import AUDIT_POSITIVE_THRESHOLD
from beltwatch.labels import CLASS_NAMES, TARGET_CLASS_IDS
from beltwatch.provenance import git_commit

log = logging.getLogger(__name__)

STUDY_VERSION = "usability-study-1"
STUDY_FILE = "study.json"
FULL_FRAME = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
MATERIAL_PRESENT_THRESHOLD = 0.01
"""A material counts as present when its coverage is at least 1% (truth and suggestion)."""
SETS = ("A", "B")


class StudyError(RuntimeError):
    """The study cannot be prepared, submitted or analysed as declared."""


@dataclass(frozen=True)
class Candidate:
    image_id: str
    sequence_id: str
    frame_index: int
    image_path: str
    image_sha256: str
    coverage: dict[str, float]

    @property
    def total(self) -> float:
        return sum(self.coverage.values())

    @property
    def audit_positive(self) -> bool:
        return self.total > AUDIT_POSITIVE_THRESHOLD


def load_val_candidates(splits_csv: Path, images_csv: Path) -> list[Candidate]:
    with splits_csv.open(newline="", encoding="utf-8") as fh:
        val = {r["image_id"] for r in csv.DictReader(fh) if r["split"] == "val"}
    out = []
    with images_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["image_id"] not in val:
                continue
            px = {name: int(row[f"px_{name}"]) for name in CLASS_NAMES.values()}
            valid = sum(px.values())
            out.append(
                Candidate(
                    image_id=row["image_id"],
                    sequence_id=row["sequence_id"],
                    frame_index=int(row["frame_index"]),
                    image_path=row["image_path"],
                    image_sha256=row["image_sha256"],
                    coverage={CLASS_NAMES[c]: px[CLASS_NAMES[c]] / valid for c in TARGET_CLASS_IDS},
                )
            )
    if len(out) != len(val):
        raise StudyError(f"{len(val) - len(out)} val images are missing from {images_csv}")
    return sorted(out, key=lambda c: c.image_id)


def _far_enough(c: Candidate, chosen: list[Candidate], min_gap: int) -> bool:
    return all(
        o.sequence_id != c.sequence_id or abs(o.frame_index - c.frame_index) >= min_gap
        for o in chosen
    )


def _draw(
    pool: list[Candidate],
    n: int,
    chosen: list[Candidate],
    min_gap: int,
    rng: np.random.Generator,
    label: str,
) -> list[Candidate]:
    picked: list[Candidate] = []
    for i in rng.permutation(len(pool)):
        if len(picked) == n:
            break
        if pool[i] not in chosen and _far_enough(pool[i], chosen + picked, min_gap):
            picked.append(pool[i])
    if len(picked) < n:
        raise StudyError(f"only {len(picked)} of {n} {label} images satisfy the frame gap")
    return picked


def select_study(
    candidates: list[Candidate],
    *,
    set_size: int = 20,
    positives_per_set: int = 14,
    min_frame_gap: int = 100,
    seed: int = 0,
) -> dict[str, list[Candidate]]:
    """Two coverage-matched sets plus one positive and one negative practice image."""
    rng = np.random.default_rng(seed)
    negatives = [c for c in candidates if not c.audit_positive]
    positives = [c for c in candidates if c.audit_positive]
    n_pos, n_neg = 2 * positives_per_set, 2 * (set_size - positives_per_set)
    edges = np.quantile([c.total for c in positives], [1 / 3, 2 / 3])
    tertiles = [
        [c for c in positives if c.total <= edges[0]],
        [c for c in positives if edges[0] < c.total <= edges[1]],
        [c for c in positives if c.total > edges[1]],
    ]
    quotas = [n_pos // 3 + (1 if i < n_pos % 3 else 0) for i in range(3)]
    chosen: list[Candidate] = []
    neg = _draw(negatives, n_neg, chosen, min_frame_gap, rng, "negative")
    chosen += neg
    pos: list[Candidate] = []
    for i, (tertile, quota) in enumerate(zip(tertiles, quotas, strict=True)):
        drawn = _draw(tertile, quota, chosen + pos, min_frame_gap, rng, f"tertile-{i + 1}")
        pos += drawn
    chosen += pos
    practice = _draw(tertiles[1], 1, chosen, min_frame_gap, rng, "practice positive")
    practice += _draw(negatives, 1, chosen + practice, min_frame_gap, rng, "practice negative")

    sets: dict[str, list[Candidate]] = {"A": [], "B": [], "practice": practice}
    for group in (sorted(pos, key=lambda c: c.total), sorted(neg, key=lambda c: c.total)):
        # Consecutive pairs by coverage, one image of each pair to each set.
        for j in range(0, len(group), 2):
            pair = group[j : j + 2]
            order = rng.permutation(len(pair))
            for set_name, k in zip(SETS, order, strict=False):
                sets[set_name].append(pair[k])
    return sets


def _set_summary(images: list[Candidate]) -> dict[str, Any]:
    return {
        "n_images": len(images),
        "n_audit_positive": sum(c.audit_positive for c in images),
        "prevalence": sum(c.audit_positive for c in images) / len(images),
        "mean_total_coverage": float(np.mean([c.total for c in images])),
        "by_sequence": {
            s: sum(c.sequence_id == s for c in images)
            for s in sorted({c.sequence_id for c in images})
        },
    }


def prepare_study(
    out: Path,
    *,
    splits_csv: Path,
    images_csv: Path,
    raw_dir: Path,
    seed: int = 0,
    set_size: int = 20,
    positives_per_set: int = 14,
    min_frame_gap: int = 100,
) -> dict[str, Any]:
    if out.exists() and any(out.iterdir()):
        raise StudyError(f"{out} is not empty; prepare a study into a new directory")
    candidates = load_val_candidates(splits_csv, images_csv)
    sets = select_study(
        candidates,
        set_size=set_size,
        positives_per_set=positives_per_set,
        min_frame_gap=min_frame_gap,
        seed=seed,
    )
    selected = [(name, c) for name, group in sets.items() for c in group]
    missing = [c.image_path for _, c in selected if not (raw_dir / c.image_path).is_file()]
    if missing:
        raise StudyError(f"{len(missing)} selected frames are not under {raw_dir}: {missing[:3]}")
    altered = [
        c.image_path
        for _, c in selected
        if hashlib.sha256((raw_dir / c.image_path).read_bytes()).hexdigest() != c.image_sha256
    ]
    if altered:
        raise StudyError(f"{len(altered)} frames do not match the manifest SHA-256: {altered[:3]}")
    # Study IDs are assigned in random order so they reveal neither set nor stratum.
    order = np.random.default_rng(seed + 1).permutation(len(selected))
    (out / "images").mkdir(parents=True)
    images = []
    for rank, i in enumerate(order, start=1):
        set_name, c = selected[i]
        study_id = f"s{rank:02d}"
        source = raw_dir / c.image_path
        file = f"images/{study_id}{source.suffix.lower()}"
        shutil.copyfile(source, out / file)
        images.append(
            {
                "study_id": study_id,
                "set": set_name,
                "file": file,
                "source_image_id": c.image_id,
                "sequence_id": c.sequence_id,
                "frame_index": c.frame_index,
                "coverage": c.coverage,
                "total_coverage": c.total,
                "audit_positive": c.audit_positive,
                "materials_present": sorted(
                    m for m, v in c.coverage.items() if v >= MATERIAL_PRESENT_THRESHOLD
                ),
            }
        )
    images.sort(key=lambda r: r["study_id"])
    record = {
        "study_version": STUDY_VERSION,
        "name": out.name,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "split": "val",
        "inspection_region": FULL_FRAME,
        "rules": {
            "seed": seed,
            "set_size": set_size,
            "positives_per_set": positives_per_set,
            "min_frame_gap": min_frame_gap,
            "audit_positive_threshold": AUDIT_POSITIVE_THRESHOLD,
            "material_present_threshold": MATERIAL_PRESENT_THRESHOLD,
        },
        "val_pool": {
            "n_images": len(candidates),
            "prevalence": sum(c.audit_positive for c in candidates) / len(candidates),
        },
        "sets": {name: _set_summary(group) for name, group in sets.items()},
        "images": images,
    }
    (out / STUDY_FILE).write_text(json.dumps(record, indent=2) + "\n")
    return record


def load_study(study_dir: Path) -> dict[str, Any]:
    path = study_dir / STUDY_FILE
    if not path.is_file():
        raise StudyError(f"{study_dir} has no {STUDY_FILE}; run beltwatch.study.prepare first")
    study: dict[str, Any] = json.loads(path.read_text())
    if study.get("study_version") != STUDY_VERSION:
        raise StudyError(f"unsupported study version {study.get('study_version')!r}")
    return study


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, help="extracted dataset (default: from config)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    data = load_data_config(args.config)
    name = data.source.name
    try:
        record = prepare_study(
            args.out,
            splits_csv=data.paths.splits_dir / f"{name}-splits.csv",
            images_csv=data.paths.manifests_dir / f"{name}-images.csv",
            raw_dir=args.raw_dir or data.paths.raw_dir,
            seed=args.seed,
        )
    except StudyError as exc:
        log.error("%s", exc)
        return 1
    for set_name, summary in record["sets"].items():
        log.info(
            "set %s: %d images, %d audit-positive, sequences %s",
            set_name,
            summary["n_images"],
            summary["n_audit_positive"],
            summary["by_sequence"],
        )
    log.info("wrote %s", args.out / STUDY_FILE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
