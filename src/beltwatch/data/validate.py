"""Validate the extracted ZeroWaste-f dataset and write image manifests.

Usage::

    uv run python -m beltwatch.data.validate [--config configs/data.yaml] [--workers N]

Every image/mask pair in every released split is checked. Problems fall into
two classes:

* **errors** quarantine the image: it is excluded from the usable manifest and
  listed in the quarantine file with explicit reasons. Raw files are never moved
  or modified.
* **warnings** keep the image but record the issue on its manifest row, e.g. a
  degenerate polygon or a category that is annotated but fully occluded.

The official ``sem_seg`` masks are the authoritative pixel labels (they encode
the release's own resolution of overlapping polygons). COCO polygons are
validated for consistency only. Mask values are *source* category IDs; they are
mapped to BeltWatch IDs by name with :func:`beltwatch.labels.build_source_remap`.

Outputs are deterministic (sorted, no timestamps) so DVC can detect real changes.
"""

import argparse
import csv
import hashlib
import io
import json
import logging
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from beltwatch import __version__, labels
from beltwatch.data.config import DataConfig, LayoutConfig, load_data_config

log = logging.getLogger(__name__)

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})
POLYGON_TOLERANCE_PX = 1.0
"""How far a polygon vertex may lie outside the image before it is flagged."""

_FRAME_NAME = re.compile(
    r"^(?P<seq>\d+)_frame_(?P<frame>\d+)(?:-(?P<variant>\d+))?$", re.IGNORECASE
)

MANIFEST_FIELDS = (
    "image_id",
    "split",
    "file_name",
    "sequence_id",
    "frame_index",
    "variant",
    "image_path",
    "mask_path",
    "width",
    "height",
    "image_sha256",
    "mask_sha256",
    "n_annotations",
    *(f"px_{name}" for name in labels.CLASS_NAMES.values()),
    "warnings",
)
QUARANTINE_FIELDS = ("image_id", "split", "file_name", "errors")


class DatasetValidationError(RuntimeError):
    """The dataset is structurally broken, so per-image validation is meaningless."""


@dataclass(frozen=True)
class FrameName:
    sequence_id: str | None
    frame_index: int | None
    variant: int | None


def parse_frame_name(file_name: str) -> FrameName:
    """Parse ``"09_frame_002000-2.PNG"`` into sequence ``"09"``, frame 2000, variant 2."""
    match = _FRAME_NAME.match(Path(file_name).stem)
    if match is None:
        return FrameName(None, None, None)
    variant = match.group("variant")
    return FrameName(
        match.group("seq"), int(match.group("frame")), int(variant) if variant else None
    )


@dataclass
class SplitAnnotations:
    categories: dict[int, str]
    images: dict[str, dict[str, Any]]
    """Image entries keyed by file name."""
    annotations: dict[int, list[dict[str, Any]]] = field(default_factory=dict)
    """Annotations keyed by image ID."""


def load_split_annotations(path: Path) -> SplitAnnotations:
    """Load a COCO annotation file, raising on structural corruption."""
    try:
        coco = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetValidationError(f"{path}: cannot read annotations: {exc}") from exc

    categories = {int(c["id"]): str(c["name"]) for c in coco["categories"]}
    images: dict[str, dict[str, Any]] = {}
    ids: set[int] = set()
    for entry in coco["images"]:
        if entry["id"] in ids:
            raise DatasetValidationError(f"{path}: duplicate image id {entry['id']}")
        if entry["file_name"] in images:
            raise DatasetValidationError(f"{path}: duplicate file name {entry['file_name']}")
        ids.add(entry["id"])
        images[entry["file_name"]] = entry

    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in coco["annotations"]:
        if ann["image_id"] not in ids:
            raise DatasetValidationError(
                f"{path}: annotation {ann['id']} references missing image {ann['image_id']}"
            )
        by_image[ann["image_id"]].append(ann)
    return SplitAnnotations(categories, images, dict(by_image))


def check_annotation(
    ann: Mapping[str, Any], categories: Mapping[int, str], width: int, height: int
) -> tuple[list[str], list[str]]:
    """Return ``(errors, warnings)`` for one COCO annotation."""
    errors: list[str] = []
    warnings: list[str] = []
    ann_id = ann.get("id")
    if ann.get("category_id") not in categories:
        errors.append(f"unknown_category(ann={ann_id}, category={ann.get('category_id')})")

    segmentation = ann.get("segmentation")
    if isinstance(segmentation, dict):
        warnings.append(f"rle_segmentation(ann={ann_id})")
        return errors, warnings
    if not segmentation:
        warnings.append(f"empty_segmentation(ann={ann_id})")
        return errors, warnings

    for polygon in segmentation:
        if len(polygon) < 6 or len(polygon) % 2:
            warnings.append(f"degenerate_polygon(ann={ann_id})")
            continue
        xs = np.asarray(polygon[0::2], dtype=np.float64)
        ys = np.asarray(polygon[1::2], dtype=np.float64)
        tol = POLYGON_TOLERANCE_PX
        if xs.min() < -tol or ys.min() < -tol or xs.max() > width + tol or ys.max() > height + tol:
            warnings.append(f"polygon_out_of_bounds(ann={ann_id})")
        area = 0.5 * abs(float(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))))
        if area == 0.0:
            warnings.append(f"zero_area_polygon(ann={ann_id})")
    return errors, warnings


@dataclass(frozen=True)
class PairResult:
    """Facts about one image/mask pair. ``errors`` non-empty means quarantine."""

    width: int | None = None
    height: int | None = None
    image_sha256: str | None = None
    mask_sha256: str | None = None
    mask_values: dict[int, int] = field(default_factory=dict)
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


def check_pair(
    image_path: Path,
    mask_path: Path,
    expected_size: tuple[int, int],
    allowed_mask_values: frozenset[int],
) -> PairResult:
    """Decode and check one image and its mask. Never raises for bad files."""
    errors: list[str] = []
    warnings: list[str] = []

    try:
        image_bytes = image_path.read_bytes()
    except OSError:
        return PairResult(errors=("missing_image",))
    image_sha = hashlib.sha256(image_bytes).hexdigest()
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.load()
            size, mode = img.size, img.mode
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        return PairResult(image_sha256=image_sha, errors=(f"image_decode_failed({exc})",))

    width, height = size
    if mode != "RGB":
        warnings.append(f"image_mode({mode})")
    if size != expected_size:
        errors.append(f"image_size_mismatch(file={size}, annotations={expected_size})")

    try:
        mask_bytes = mask_path.read_bytes()
    except OSError:
        errors.append("missing_mask")
        return PairResult(width, height, image_sha, errors=tuple(errors), warnings=tuple(warnings))
    mask_sha = hashlib.sha256(mask_bytes).hexdigest()
    try:
        with Image.open(io.BytesIO(mask_bytes)) as mask_img:
            mask = np.asarray(mask_img)
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        errors.append(f"mask_decode_failed({exc})")
        return PairResult(
            width, height, image_sha, mask_sha, errors=tuple(errors), warnings=tuple(warnings)
        )

    mask_values: dict[int, int] = {}
    if mask.ndim != 2:
        errors.append(f"mask_not_single_channel(shape={mask.shape})")
    elif (mask.shape[1], mask.shape[0]) != size:
        errors.append(f"mask_size_mismatch(mask={(mask.shape[1], mask.shape[0])}, image={size})")
    else:
        values, counts = np.unique(mask, return_counts=True)
        mask_values = {int(v): int(c) for v, c in zip(values, counts, strict=True)}
        unexpected = sorted(set(mask_values) - allowed_mask_values)
        if unexpected:
            errors.append(f"unexpected_mask_values({unexpected})")

    return PairResult(
        width,
        height,
        image_sha,
        mask_sha,
        mask_values,
        tuple(errors),
        tuple(warnings),
    )


def _check_pair_star(args: tuple[Path, Path, tuple[int, int], frozenset[int]]) -> PairResult:
    return check_pair(*args)


@dataclass
class ValidationResult:
    manifest: list[dict[str, Any]]
    quarantine: list[dict[str, Any]]
    summary: dict[str, Any]


def _code(issue: str) -> str:
    return issue.split("(", 1)[0]


def _list_images(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {p.name for p in directory.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}


def validate_dataset(raw_dir: Path, layout: LayoutConfig, *, workers: int = 1) -> ValidationResult:
    """Validate every split under ``raw_dir`` and build manifest rows."""
    root = raw_dir / layout.root_subdir
    if not root.is_dir():
        raise DatasetValidationError(f"{root} does not exist; run the download stage first")

    split_annotations = {
        split: load_split_annotations(root / split / layout.annotations_file)
        for split in layout.splits
    }
    category_sets = {split: a.categories for split, a in split_annotations.items()}
    first_split = layout.splits[0]
    categories = category_sets[first_split]
    for split, cats in category_sets.items():
        if cats != categories:
            raise DatasetValidationError(
                f"categories differ between splits {first_split!r} and {split!r}: "
                f"{categories} vs {cats}"
            )
    try:
        remap = {0: labels.BACKGROUND, **labels.build_source_remap(categories)}
    except ValueError as exc:
        raise DatasetValidationError(f"cannot map source categories: {exc}") from exc
    allowed_values = frozenset(remap)

    manifest: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    error_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    split_stats: dict[str, dict[str, int]] = {}
    orphan_masks: list[str] = []

    for split in layout.splits:
        ann = split_annotations[split]
        image_dir = root / split / layout.images_subdir
        mask_dir = root / split / layout.masks_subdir
        image_files = _list_images(image_dir)
        mask_files = _list_images(mask_dir)
        orphan_masks += [f"{split}/{name}" for name in sorted(mask_files - image_files)]

        names = sorted(set(ann.images) | image_files)
        jobs = []
        for name in names:
            entry = ann.images.get(name)
            if entry is None:
                continue
            size = (int(entry["width"]), int(entry["height"]))
            jobs.append((image_dir / name, mask_dir / name, size, allowed_values))

        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_check_pair_star, jobs, chunksize=16))
        else:
            results = [_check_pair_star(job) for job in jobs]
        pair_results = {job[0].name: result for job, result in zip(jobs, results, strict=True)}

        valid = 0
        for name in names:
            image_id = f"{split}/{Path(name).stem}"
            entry = ann.images.get(name)
            errors: list[str] = []
            warnings: list[str] = []
            if entry is None:
                errors.append("not_in_annotations")
                pair = PairResult()
                anns: list[dict[str, Any]] = []
            else:
                pair = pair_results[name]
                errors += pair.errors
                warnings += pair.warnings
                anns = ann.annotations.get(entry["id"], [])
                for a in anns:
                    a_err, a_warn = check_annotation(
                        a, categories, int(entry["width"]), int(entry["height"])
                    )
                    errors += a_err
                    warnings += a_warn
                if not anns:
                    warnings.append("no_annotations")
                if pair.mask_values:
                    in_mask = {v for v in pair.mask_values if v != 0}
                    in_json = {int(a["category_id"]) for a in anns} & set(categories)
                    if in_json - in_mask:
                        missing = sorted(categories[c] for c in in_json - in_mask)
                        warnings.append(f"category_missing_from_mask({missing})")
                    if (in_mask - in_json) & set(categories):
                        extra = sorted(categories[c] for c in (in_mask - in_json) & set(categories))
                        warnings.append(f"category_missing_from_annotations({extra})")

            error_counts.update(_code(e) for e in errors)
            warning_counts.update(_code(w) for w in warnings)
            if errors:
                quarantine.append(
                    {
                        "image_id": image_id,
                        "split": split,
                        "file_name": name,
                        "errors": "; ".join(errors),
                    }
                )
                continue

            valid += 1
            frame = parse_frame_name(name)
            if frame.sequence_id is None:
                warnings.append("unparsed_frame_name")
                warning_counts["unparsed_frame_name"] += 1
            pixels = Counter[int]()
            for source_value, count in pair.mask_values.items():
                pixels[remap[source_value]] += count
            manifest.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "file_name": name,
                    "sequence_id": frame.sequence_id,
                    "frame_index": frame.frame_index,
                    "variant": frame.variant,
                    "image_path": (image_dir / name).relative_to(raw_dir).as_posix(),
                    "mask_path": (mask_dir / name).relative_to(raw_dir).as_posix(),
                    "width": pair.width,
                    "height": pair.height,
                    "image_sha256": pair.image_sha256,
                    "mask_sha256": pair.mask_sha256,
                    "n_annotations": len(anns),
                    **{
                        f"px_{class_name}": pixels.get(class_id, 0)
                        for class_id, class_name in labels.CLASS_NAMES.items()
                    },
                    "warnings": "; ".join(warnings),
                }
            )

        split_stats[split] = {
            "images_in_annotations": len(ann.images),
            "image_files": len(image_files),
            "annotations": sum(len(v) for v in ann.annotations.values()),
            "valid": valid,
            "quarantined": len(names) - valid,
        }
        log.info("%s: %d valid, %d quarantined", split, valid, len(names) - valid)

    manifest.sort(key=lambda row: row["image_id"])
    quarantine.sort(key=lambda row: row["image_id"])
    summary = _summarize(
        manifest, quarantine, categories, remap, split_stats, error_counts, warning_counts
    )
    summary["orphan_masks"] = orphan_masks
    return ValidationResult(manifest, quarantine, summary)


def _summarize(
    manifest: Sequence[Mapping[str, Any]],
    quarantine: Sequence[Mapping[str, Any]],
    categories: Mapping[int, str],
    remap: Mapping[int, int],
    split_stats: Mapping[str, Mapping[str, int]],
    error_counts: Counter[str],
    warning_counts: Counter[str],
) -> dict[str, Any]:
    total_px = {name: 0 for name in labels.CLASS_NAMES.values()}
    images_with = {name: 0 for name in labels.CLASS_NAMES.values()}
    sequences: dict[str, Counter[str]] = defaultdict(Counter)
    for row in manifest:
        for name in total_px:
            px = int(row[f"px_{name}"])
            total_px[name] += px
            images_with[name] += px > 0
        sequences[str(row["sequence_id"])][str(row["split"])] += 1
    all_px = sum(total_px.values()) or 1
    return {
        "source_categories": {str(k): v for k, v in sorted(categories.items())},
        "source_to_beltwatch": {str(k): v for k, v in sorted(remap.items())},
        "splits": dict(split_stats),
        "total_valid": len(manifest),
        "total_quarantined": len(quarantine),
        "error_counts": dict(sorted(error_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "class_pixel_fraction": {k: round(v / all_px, 6) for k, v in total_px.items()},
        "images_with_class": images_with,
        "sequences_by_split": {k: dict(sorted(v.items())) for k, v in sorted(sequences.items())},
        "beltwatch_version": __version__,
    }


def _write_csv(path: Path, fields: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(result: ValidationResult, config: DataConfig) -> dict[str, Path]:
    out_dir = config.paths.manifests_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name = config.source.name
    paths = {
        "manifest": out_dir / f"{name}-images.csv",
        "quarantine": out_dir / f"{name}-quarantine.csv",
        "summary": out_dir / f"{name}-validation.json",
    }
    _write_csv(paths["manifest"], MANIFEST_FIELDS, result.manifest)
    _write_csv(paths["quarantine"], QUARANTINE_FIELDS, result.quarantine)
    summary = {
        "source": {
            "name": config.source.name,
            "version": config.source.version,
            "md5": config.source.md5,
        },
        **result.summary,
    }
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the extracted dataset.")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_data_config(args.config)
    result = validate_dataset(config.paths.raw_dir, config.layout, workers=args.workers)
    paths = write_outputs(result, config)
    log.info(
        "%d valid, %d quarantined; wrote %s",
        result.summary["total_valid"],
        result.summary["total_quarantined"],
        ", ".join(str(p) for p in paths.values()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
