"""Audit exact and near-duplicate images, and temporal proximity across splits.

Usage::

    uv run python -m beltwatch.data.duplicates [--config configs/data.yaml] [--workers N]

Reads the validated image manifest and writes, to the manifests directory:

* ``<name>-hashes.csv``: perceptual hashes (pHash and dHash) per image.
* ``<name>-duplicates.csv``: every pair of images that are byte-identical or
  whose pHash Hamming distance is at or below the configured threshold.
* ``<name>-duplicates.json``: a summary covering exact-duplicate groups,
  near-duplicate pair counts within and across splits, and, for each recording
  sequence present in several splits, the smallest frame gap between splits.

This stage only reports. Excluding leaking examples is the split stage's job.
A clean report does not prove independence: perceptual hashing misses
similar-but-not-identical scenes, which is why sequence grouping matters too.
"""

import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import cache
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image

from beltwatch import __version__
from beltwatch.data.config import DataConfig, load_data_config

log = logging.getLogger(__name__)

HASH_FIELDS = ("image_id", "split", "phash", "dhash")
PAIR_FIELDS = (
    "image_id_a",
    "image_id_b",
    "split_a",
    "split_b",
    "sequence_a",
    "sequence_b",
    "frame_gap",
    "phash_distance",
    "dhash_distance",
    "exact",
    "cross_split",
)
_PHASH_HIGHFREQ_FACTOR = 4


@cache
def _dct_matrix(n: int) -> npt.NDArray[np.float64]:
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    m: npt.NDArray[np.float64] = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    m[0] /= np.sqrt(2.0)
    return m


def _bits_to_int(bits: npt.NDArray[np.bool_]) -> int:
    return int("".join("1" if b else "0" for b in bits.flatten()), 2)


def phash(image: Image.Image, hash_size: int = 8) -> int:
    """DCT perceptual hash: robust to re-encoding, mild noise, and small brightness changes."""
    n = hash_size * _PHASH_HIGHFREQ_FACTOR
    gray = np.asarray(image.convert("L").resize((n, n), Image.Resampling.LANCZOS), dtype=np.float64)
    dct = _dct_matrix(n)
    low = (dct @ gray @ dct.T)[:hash_size, :hash_size]
    median = float(np.median(low.flatten()[1:]))  # exclude the DC term
    return _bits_to_int(low > median)


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Difference hash: compares horizontally adjacent pixels of a tiny grayscale image."""
    gray = np.asarray(
        image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS),
        dtype=np.int16,
    )
    return _bits_to_int(gray[:, 1:] > gray[:, :-1])


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def _hash_file(args: tuple[Path, int]) -> tuple[int, int]:
    path, hash_size = args
    with Image.open(path) as img:
        img.load()
        return phash(img, hash_size), dhash(img, hash_size)


@dataclass(frozen=True)
class ImageEntry:
    image_id: str
    split: str
    sequence_id: str | None
    frame_index: int | None
    image_sha256: str
    phash: int
    dhash: int


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def hash_images(
    rows: Sequence[Mapping[str, str]], raw_dir: Path, hash_size: int, *, workers: int = 1
) -> list[ImageEntry]:
    jobs = [(raw_dir / row["image_path"], hash_size) for row in rows]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            hashes = list(pool.map(_hash_file, jobs, chunksize=16))
    else:
        hashes = [_hash_file(job) for job in jobs]
    return [
        ImageEntry(
            image_id=row["image_id"],
            split=row["split"],
            sequence_id=row["sequence_id"] or None,
            frame_index=int(row["frame_index"]) if row["frame_index"] else None,
            image_sha256=row["image_sha256"],
            phash=p,
            dhash=d,
        )
        for row, (p, d) in zip(rows, hashes, strict=True)
    ]


def find_pairs(entries: Sequence[ImageEntry], max_hamming: int) -> list[dict[str, Any]]:
    """All pairs that are byte-identical or within ``max_hamming`` pHash bits."""
    n = len(entries)
    phashes = np.array([e.phash for e in entries], dtype=np.uint64)
    candidates: set[tuple[int, int]] = set()

    chunk = 512
    for start in range(0, n, chunk):
        block = phashes[start : start + chunk, None] ^ phashes[None, :]
        rows, cols = np.nonzero(np.bitwise_count(block) <= max_hamming)
        for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
            i = start + r
            if c > i:
                candidates.add((i, c))

    by_sha: dict[str, list[int]] = defaultdict(list)
    for idx, entry in enumerate(entries):
        by_sha[entry.image_sha256].append(idx)
    for group in by_sha.values():
        candidates.update(combinations(sorted(group), 2))

    pairs = []
    for i, j in sorted(candidates):
        a, b = entries[i], entries[j]
        same_sequence = a.sequence_id is not None and a.sequence_id == b.sequence_id
        gap = (
            abs(a.frame_index - b.frame_index)
            if same_sequence and a.frame_index is not None and b.frame_index is not None
            else None
        )
        pairs.append(
            {
                "image_id_a": a.image_id,
                "image_id_b": b.image_id,
                "split_a": a.split,
                "split_b": b.split,
                "sequence_a": a.sequence_id,
                "sequence_b": b.sequence_id,
                "frame_gap": gap,
                "phash_distance": hamming(a.phash, b.phash),
                "dhash_distance": hamming(a.dhash, b.dhash),
                "exact": a.image_sha256 == b.image_sha256,
                "cross_split": a.split != b.split,
            }
        )
    return pairs


def cross_split_frame_gaps(entries: Iterable[ImageEntry]) -> dict[str, dict[str, int]]:
    """For each sequence in several splits: the smallest frame gap per split pair."""
    frames: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for e in entries:
        if e.sequence_id is not None and e.frame_index is not None:
            frames[e.sequence_id][e.split].append(e.frame_index)

    result: dict[str, dict[str, int]] = {}
    for seq in sorted(frames):
        splits = frames[seq]
        if len(splits) < 2:
            continue
        gaps: dict[str, int] = {}
        for sa, sb in combinations(sorted(splits), 2):
            fa = np.sort(np.array(splits[sa]))
            fb = np.array(splits[sb])
            idx = np.clip(np.searchsorted(fa, fb), 1, len(fa)) - 1
            nearest = np.minimum(
                np.abs(fb - fa[idx]), np.abs(fb - fa[np.minimum(idx + 1, len(fa) - 1)])
            )
            gaps[f"{sa}|{sb}"] = int(nearest.min())
        result[seq] = gaps
    return result


def summarize(
    entries: Sequence[ImageEntry], pairs: Sequence[Mapping[str, Any]], max_hamming: int
) -> dict[str, Any]:
    by_sha: dict[str, list[str]] = defaultdict(list)
    for e in entries:
        by_sha[e.image_sha256].append(e.image_id)
    exact_groups = sorted(sorted(ids) for ids in by_sha.values() if len(ids) > 1)

    near = [p for p in pairs if not p["exact"]]
    by_split_pair: dict[str, int] = defaultdict(int)
    for p in near:
        key = "|".join(sorted((p["split_a"], p["split_b"])))
        by_split_pair[key] += 1
    return {
        "images": len(entries),
        "phash_max_hamming": max_hamming,
        "exact_duplicate_groups": exact_groups,
        "near_duplicate_pairs": len(near),
        "near_duplicate_pairs_by_split_pair": dict(sorted(by_split_pair.items())),
        "cross_split_pairs": sum(1 for p in pairs if p["cross_split"]),
        "cross_split_min_frame_gap": cross_split_frame_gaps(entries),
        "beltwatch_version": __version__,
    }


def _write_csv(path: Path, fields: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(config: DataConfig, *, workers: int = 1) -> dict[str, Path]:
    out_dir = config.paths.manifests_dir
    name = config.source.name
    rows = read_manifest(out_dir / f"{name}-images.csv")
    settings = config.duplicates
    log.info("hashing %d images", len(rows))
    entries = hash_images(rows, config.paths.raw_dir, settings.hash_size, workers=workers)
    pairs = find_pairs(entries, settings.phash_max_hamming)
    summary = summarize(entries, pairs, settings.phash_max_hamming)

    width = settings.hash_size**2 // 4
    paths = {
        "hashes": out_dir / f"{name}-hashes.csv",
        "pairs": out_dir / f"{name}-duplicates.csv",
        "summary": out_dir / f"{name}-duplicates.json",
    }
    _write_csv(
        paths["hashes"],
        HASH_FIELDS,
        (
            {
                "image_id": e.image_id,
                "split": e.split,
                "phash": f"{e.phash:0{width}x}",
                "dhash": f"{e.dhash:0{width}x}",
            }
            for e in entries
        ),
    )
    _write_csv(paths["pairs"], PAIR_FIELDS, pairs)
    paths["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info(
        "%d exact-duplicate groups, %d near-duplicate pairs (%d cross-split)",
        len(summary["exact_duplicate_groups"]),
        summary["near_duplicate_pairs"],
        summary["cross_split_pairs"],
    )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit duplicate and near-duplicate images.")
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(load_data_config(args.config), workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
