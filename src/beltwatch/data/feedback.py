"""Turn consented reviewer feedback into annotation-review candidates.

Usage::

    uv run python -m beltwatch.data.feedback export --data-dir var --out data/feedback/<batch>
    uv run python -m beltwatch.data.feedback review data/feedback/<batch> <candidate_id> \\
        --decision accept --annotator <name> --reason "<why>"
    uv run python -m beltwatch.data.feedback eligible data/feedback/<batch>

Reviewer feedback is not training data. It becomes a *candidate* here, and a
candidate is usable for a dataset version only after an annotator accepts it.

``export`` reads the app database read-only and takes only audits created with
``training_consent``. Uploads and corrected masks are copied out of app
storage (which expires after the retention period) into an immutable batch
directory, and each copied image is checked against its recorded SHA-256.
Each candidate keeps the original prediction and the full, ordered feedback
history, so the model output and every human decision stay distinguishable.

The latest decision sets the candidate's ``label_kind``:

* ``pixel_mask``: the reviewer submitted a corrected mask (internal class IDs).
* ``image_level``: the reviewer confirmed or rejected materials. This is an
  image-level label; it does not validate the predicted pixels.
* ``retake``: the reviewer marked the image unusable.

``review`` appends an annotator decision to ``reviews.jsonl`` (never rewritten).
``eligible`` lists candidates whose latest annotator decision is ``accept`` and
whose label is a pixel mask: the only ones a segmentation dataset version may
use. Accepted image-level candidates are reported separately.

Deleting an audit in the app does not reach batches already exported. Apply a
deletion request to the batch by hand and record it with a ``reject`` review.
"""

import argparse
import hashlib
import json
import logging
import shutil
import sqlite3
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image

from beltwatch.labels import CLASS_NAMES
from beltwatch.provenance import git_commit

log = logging.getLogger(__name__)

FEEDBACK_EXPORT_VERSION = "feedback-export-1"
CANDIDATES_FILE = "candidates.jsonl"
REVIEWS_FILE = "reviews.jsonl"
EXPORT_FILE = "export.json"

LabelKind = Literal["pixel_mask", "image_level", "retake"]
ReviewDecision = Literal["accept", "reject"]


class FeedbackExportError(RuntimeError):
    """The export or review request cannot be carried out."""


@dataclass(frozen=True)
class ExportSummary:
    out: Path
    candidates: int
    by_label_kind: dict[str, int]
    skipped: list[dict[str, str]]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _label_kind(latest: dict[str, Any]) -> LabelKind:
    if latest["decision"] == "needs_retake":
        return "retake"
    if latest["decision"] == "corrected" and latest["correction_path"]:
        return "pixel_mask"
    return "image_level"


def _check_mask(path: Path, width: int, height: int) -> str | None:
    with Image.open(path) as img:
        mask = np.asarray(img)
    if mask.ndim != 2 or mask.shape != (height, width):
        return "corrected mask is not single-channel at the image size"
    if not set(np.unique(mask).tolist()) <= set(CLASS_NAMES):
        return "corrected mask contains unknown class IDs"
    return None


def export_candidates(database: Path, data_dir: Path, out: Path) -> ExportSummary:
    """Copy consented, reviewed images into a new batch of review candidates."""
    if out.exists() and any(out.iterdir()):
        raise FeedbackExportError(f"{out} is not empty; batches are immutable")
    if not database.is_file():
        raise FeedbackExportError(f"no database at {database}")
    db = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    try:
        images = db.execute(
            """SELECT i.*, a.camera_id, a.inspection_region, a.model_version,
                      a.preprocessing_version, a.review_policy_version,
                      a.created_at AS audit_created_at
               FROM images i JOIN audits a ON a.id = i.audit_id
               WHERE a.training_consent = 1 AND i.status = 'complete'
                 AND EXISTS (SELECT 1 FROM feedback f WHERE f.image_id = i.id)
               ORDER BY a.created_at, i.audit_id, i.position"""
        ).fetchall()
        feedback_rows = db.execute(
            """SELECT f.*, ar.path AS correction_path FROM feedback f
               LEFT JOIN artifacts ar ON ar.id = f.correction_artifact_id
               ORDER BY f.id"""
        ).fetchall()
    finally:
        db.close()
    history: dict[str, list[dict[str, Any]]] = {}
    for row in feedback_rows:
        item = dict(row)
        item["materials_present"] = json.loads(item["materials_present"])
        history.setdefault(item["image_id"], []).append(item)

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for image in images:
        image_id = image["id"]
        source = data_dir / image["upload_path"]
        if not source.is_file():
            skipped.append({"image_id": image_id, "reason": "upload no longer in app storage"})
            continue
        if _sha256(source) != image["sha256"]:
            skipped.append({"image_id": image_id, "reason": "upload does not match its SHA-256"})
            continue
        feedback = history[image_id]
        latest = feedback[-1]
        kind = _label_kind(latest)
        mask_file = None
        if kind == "pixel_mask":
            mask_source = data_dir / latest["correction_path"]
            problem = (
                "corrected mask no longer in app storage"
                if not mask_source.is_file()
                else _check_mask(mask_source, image["width"], image["height"])
            )
            if problem:
                skipped.append({"image_id": image_id, "reason": problem})
                continue
            mask_file = f"masks/{image_id}.png"
            shutil.copyfile(mask_source, out / mask_file)
        image_file = f"images/{image['sha256']}{source.suffix.lower()}"
        shutil.copyfile(source, out / image_file)
        candidates.append(
            {
                "candidate_id": image_id,
                "audit_id": image["audit_id"],
                "audit_created_at": image["audit_created_at"],
                "camera_id": image["camera_id"],
                "inspection_region": json.loads(image["inspection_region"]),
                "original_filename": image["original_filename"],
                "image_file": image_file,
                "image_sha256": image["sha256"],
                "width": image["width"],
                "height": image["height"],
                "model_version": image["model_version"],
                "preprocessing_version": image["preprocessing_version"],
                "review_policy_version": image["review_policy_version"],
                "prediction": json.loads(image["result"]) if image["result"] else None,
                "feedback": [
                    {k: f[k] for k in f if k not in {"correction_path", "audit_id", "image_id"}}
                    for f in feedback
                ],
                "label_kind": kind,
                "mask_file": mask_file,
                "status": "pending_annotation_review",
            }
        )
    with (out / CANDIDATES_FILE).open("w") as fh:
        for c in candidates:
            fh.write(json.dumps(c, sort_keys=True) + "\n")
    (out / REVIEWS_FILE).touch()
    by_kind = dict(sorted(Counter(c["label_kind"] for c in candidates).items()))
    record = {
        "export_version": FEEDBACK_EXPORT_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "label_map": {str(k): v for k, v in CLASS_NAMES.items()},
        "mask_values": "internal class IDs (beltwatch.labels)",
        "candidates": len(candidates),
        "by_label_kind": by_kind,
        "skipped": skipped,
        "rule": "only accepted pixel_mask candidates may enter a dataset version",
    }
    (out / EXPORT_FILE).write_text(json.dumps(record, indent=2) + "\n")
    return ExportSummary(out, len(candidates), by_kind, skipped)


def load_candidates(batch: Path) -> dict[str, dict[str, Any]]:
    path = batch / CANDIDATES_FILE
    if not path.is_file():
        raise FeedbackExportError(f"{batch} is not a feedback batch (no {CANDIDATES_FILE})")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["candidate_id"]: r for r in rows}


def load_reviews(batch: Path) -> list[dict[str, Any]]:
    path = batch / REVIEWS_FILE
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def record_review(
    batch: Path, candidate_id: str, decision: ReviewDecision, annotator: str, reason: str
) -> dict[str, Any]:
    """Append one annotator decision. Earlier decisions are kept, never edited."""
    if candidate_id not in load_candidates(batch):
        raise FeedbackExportError(f"no candidate {candidate_id} in {batch}")
    if decision not in ("accept", "reject"):
        raise FeedbackExportError(f"decision must be accept or reject, not {decision!r}")
    if not annotator.strip() or not reason.strip():
        raise FeedbackExportError("an annotator name and a reason are required")
    entry = {
        "candidate_id": candidate_id,
        "decision": decision,
        "annotator": annotator.strip(),
        "reason": reason.strip(),
        "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with (batch / REVIEWS_FILE).open("a") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
    return entry


def review_status(batch: Path) -> dict[str, dict[str, Any]]:
    """Each candidate with its latest annotator decision (or none yet)."""
    candidates = load_candidates(batch)
    latest: dict[str, dict[str, Any]] = {}
    for entry in load_reviews(batch):
        latest[entry["candidate_id"]] = entry
    return {
        cid: {**c, "annotation_review": latest.get(cid), "status": _status(latest.get(cid))}
        for cid, c in candidates.items()
    }


def _status(review: dict[str, Any] | None) -> str:
    if review is None:
        return "pending_annotation_review"
    return "accepted" if review["decision"] == "accept" else "rejected"


def eligible_candidates(batch: Path) -> list[dict[str, Any]]:
    """Accepted pixel-mask candidates: the only ones a dataset version may use."""
    return [
        c
        for c in review_status(batch).values()
        if c["status"] == "accepted" and c["label_kind"] == "pixel_mask"
    ]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("export", help="copy consented feedback into a new candidate batch")
    ex.add_argument("--data-dir", type=Path, default=Path("var"), help="app data directory")
    ex.add_argument("--database", type=Path, help="default: <data-dir>/beltwatch.sqlite3")
    ex.add_argument("--out", type=Path, required=True)
    rv = sub.add_parser("review", help="append an annotator decision for one candidate")
    rv.add_argument("batch", type=Path)
    rv.add_argument("candidate_id")
    rv.add_argument("--decision", choices=["accept", "reject"], required=True)
    rv.add_argument("--annotator", required=True)
    rv.add_argument("--reason", required=True)
    el = sub.add_parser("eligible", help="list candidates usable for a dataset version")
    el.add_argument("batch", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        if args.command == "export":
            database = args.database or args.data_dir / "beltwatch.sqlite3"
            summary = export_candidates(database, args.data_dir, args.out)
            log.info(
                "exported %d candidates to %s %s",
                summary.candidates,
                summary.out,
                summary.by_label_kind,
            )
            for s in summary.skipped:
                log.warning("skipped %s: %s", s["image_id"], s["reason"])
        elif args.command == "review":
            entry = record_review(
                args.batch, args.candidate_id, args.decision, args.annotator, args.reason
            )
            log.info("recorded %s for %s", entry["decision"], entry["candidate_id"])
        else:
            status = review_status(args.batch)
            counts = Counter(c["status"] for c in status.values())
            eligible = eligible_candidates(args.batch)
            image_level = [
                c
                for c in status.values()
                if c["status"] == "accepted" and c["label_kind"] == "image_level"
            ]
            print(
                json.dumps(
                    {
                        "status": dict(sorted(counts.items())),
                        "eligible_pixel_masks": [c["candidate_id"] for c in eligible],
                        "accepted_image_level_labels": [c["candidate_id"] for c in image_level],
                    },
                    indent=2,
                )
            )
    except FeedbackExportError as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
