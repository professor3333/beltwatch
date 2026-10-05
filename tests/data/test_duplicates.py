import csv
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from beltwatch.data import duplicates as d
from beltwatch.data.config import load_data_config


def scene(seed: int, size: tuple[int, int] = (96, 64)) -> Image.Image:
    """A smooth random 'scene': low-resolution noise upsampled, like large objects."""
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 256, (6, 8, 3), dtype=np.uint8)
    return Image.fromarray(small).resize(size, Image.Resampling.BICUBIC)


def jpeg_roundtrip(img: Image.Image, quality: int = 70) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def entry(
    image_id: str,
    phash: int,
    *,
    split: str = "train",
    seq: str | None = "01",
    frame: int | None = 0,
    sha: str | None = None,
) -> d.ImageEntry:
    return d.ImageEntry(image_id, split, seq, frame, sha or image_id, phash, 0)


def test_identical_images_hash_identically() -> None:
    img = scene(1)
    assert d.phash(img) == d.phash(img.copy())
    assert d.dhash(img) == d.dhash(img.copy())


def test_reencoded_image_is_near_duplicate() -> None:
    img = scene(1)
    assert d.hamming(d.phash(img), d.phash(jpeg_roundtrip(img))) <= 6


def test_different_scenes_are_far_apart() -> None:
    distances = [d.hamming(d.phash(scene(1)), d.phash(scene(s))) for s in range(2, 12)]
    assert min(distances) > 6


def test_hash_size_controls_bit_width() -> None:
    assert d.phash(scene(1), hash_size=4) < 2**16
    assert d.dhash(scene(1), hash_size=4) < 2**16


def test_find_pairs_flags_near_and_exact_duplicates() -> None:
    entries = [
        entry("train/a", 0b0000, frame=10),
        entry("train/b", 0b0011, frame=20),  # 2 bits from a
        entry("test/c", 0xFFFF, split="test", frame=15),  # far from both
        entry("test/a-copy", 0xF0F0, split="test", seq="02", sha="train/a"),  # same bytes as a
    ]

    pairs = d.find_pairs(entries, max_hamming=2)

    by_ids = {(p["image_id_a"], p["image_id_b"]): p for p in pairs}
    assert set(by_ids) == {("train/a", "train/b"), ("train/a", "test/a-copy")}
    near = by_ids[("train/a", "train/b")]
    assert (near["phash_distance"], near["frame_gap"], near["exact"]) == (2, 10, False)
    assert near["cross_split"] is False
    exact = by_ids[("train/a", "test/a-copy")]
    assert exact["exact"] is True
    assert exact["cross_split"] is True
    assert exact["frame_gap"] is None  # different sequences


def test_find_pairs_across_chunk_boundary() -> None:
    entries = [entry(f"train/{i:04d}", (i * 0x9E3779B97F4A7C15) % 2**64) for i in range(600)]
    entries.append(entry("train/twin", entries[0].phash))

    pairs = d.find_pairs(entries, max_hamming=0)

    assert [(p["image_id_a"], p["image_id_b"]) for p in pairs] == [("train/0000", "train/twin")]


def test_cross_split_frame_gaps() -> None:
    entries = [
        entry("train/1", 0, frame=1000),
        entry("train/2", 0, frame=2000),
        entry("val/1", 0, split="val", frame=991),
        entry("test/1", 0, split="test", frame=3000),
        entry("test/2", 0, split="test", frame=1500),
        entry("train/x", 0, seq="06", frame=5),  # single-split sequence is omitted
    ]

    gaps = d.cross_split_frame_gaps(entries)

    assert gaps == {"01": {"test|train": 500, "test|val": 509, "train|val": 9}}


def test_summary_counts() -> None:
    entries = [
        entry("train/a", 0),
        entry("test/b", 1, split="test"),
        entry("train/c", 0xFF, sha="train/a"),
    ]
    pairs = d.find_pairs(entries, max_hamming=1)

    summary = d.summarize(entries, pairs, max_hamming=1)

    assert summary["exact_duplicate_groups"] == [["train/a", "train/c"]]
    assert summary["near_duplicate_pairs"] == 1
    assert summary["near_duplicate_pairs_by_split_pair"] == {"test|train": 1}
    assert summary["cross_split_pairs"] == 1


def test_run_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    (raw / "imgs").mkdir(parents=True)
    images = {
        "train/01_frame_000010": scene(1),
        "train/01_frame_000020": jpeg_roundtrip(scene(1)),
        "val/02_frame_000100": scene(2),
    }
    rows = []
    for image_id, img in images.items():
        rel = f"imgs/{image_id.replace('/', '_')}.png"
        img.save(raw / rel)
        split, stem = image_id.split("/")
        rows.append(
            {
                "image_id": image_id,
                "split": split,
                "sequence_id": stem[:2],
                "frame_index": str(int(stem[-6:])),
                "image_path": rel,
                "image_sha256": image_id,
            }
        )
    manifests = tmp_path / "manifests"
    manifests.mkdir()
    with (manifests / "fixture-images.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    repo = load_data_config(Path("configs/data.yaml"))
    config = repo.model_copy(
        update={
            "source": repo.source.model_copy(update={"name": "fixture"}),
            "paths": repo.paths.model_copy(update={"raw_dir": raw, "manifests_dir": manifests}),
        }
    )

    paths = d.run(config)

    with paths["pairs"].open() as fh:
        pairs = list(csv.DictReader(fh))
    assert [(p["image_id_a"], p["image_id_b"], p["frame_gap"]) for p in pairs] == [
        ("train/01_frame_000010", "train/01_frame_000020", "10")
    ]
    with paths["hashes"].open() as fh:
        hashes = list(csv.DictReader(fh))
    assert len(hashes) == 3
    assert all(len(h["phash"]) == 16 for h in hashes)
    summary = json.loads(paths["summary"].read_text())
    assert summary["near_duplicate_pairs"] == 1
    assert summary["cross_split_pairs"] == 0


@pytest.mark.parametrize("workers", [1, 2])
def test_hash_images_parallel_matches_serial(tmp_path: Path, workers: int) -> None:
    scene(3).save(tmp_path / "x.png")
    rows = [
        {
            "image_id": "train/x",
            "split": "train",
            "sequence_id": "",
            "frame_index": "",
            "image_path": "x.png",
            "image_sha256": "h",
        }
    ]

    (result,) = d.hash_images(rows, tmp_path, 8, workers=workers)

    assert result.phash == d.phash(scene(3))
    assert result.sequence_id is None
    assert result.frame_index is None
