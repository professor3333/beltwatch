"""Fetch frames of one split from the pinned archive without downloading it.

The archive is read through HTTP Range requests: the zip's central directory
first, then only the requested members, whose CRC-32 the zip reader checks.
It is for machines that need real data from one split (a benchmark host, CI)
rather than the 7.5 GB download.

Two modes:

- ``--count N``: the first N image files of the split's image directory in
  name order (the same frames a benchmark over the extracted directory
  reads), written flat into ``--out`` with a ``frames.json`` index.
- ``--whole``: every image and mask of the split, at their archive paths
  under ``--out``, so ``--out`` can serve as the raw directory of a data
  config. Each file is checked against the SHA-256 in the committed image
  manifest, and files already present and intact are kept.

Usage::

    uv run python -m beltwatch.data.sample --split val --count 20 --out frames/
    uv run python -m beltwatch.data.sample --split val --whole --out data/raw/zerowaste-f

The locked test split is refused. The frames are dataset images: keep them
out of git and out of published artifacts.
"""

import argparse
import csv
import hashlib
import http.client
import io
import json
import logging
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from beltwatch.data.config import load_data_config
from beltwatch.data.download import IMAGE_SUFFIXES

log = logging.getLogger(__name__)

ALLOWED_SPLITS = ("train", "val")  # never test: it is locked
BLOCK_SIZE = 8 << 20
"""Bytes per Range request. Members are read in archive order, so one block
serves several frames; this keeps the request count low enough for Zenodo's
rate limit (about 1.1 GB of val in under 150 requests)."""
RATE_LIMITED = {429, 503}


class RangeError(RuntimeError):
    """The server did not return the requested byte range."""


class HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file of known size.

    Reads are served from aligned blocks of ``block_size`` bytes, each fetched
    with one Range request; the most recent blocks are kept.
    """

    def __init__(
        self,
        url: str,
        size: int,
        *,
        block_size: int = BLOCK_SIZE,
        keep_blocks: int = 2,
        timeout: float = 120,
        attempts: int = 8,
        backoff: float = 2,
    ) -> None:
        self.url, self.size = url, size
        self.block_size, self.keep_blocks = block_size, keep_blocks
        self.timeout, self.attempts, self.backoff = timeout, attempts, backoff
        self.position = 0
        self.requests = 0
        self._blocks: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, base + offset)
        return self.position

    def readinto(self, buffer: memoryview) -> int:  # type: ignore[override]
        if self.position >= self.size or not len(buffer):
            return 0
        index, offset = divmod(self.position, self.block_size)
        block = self._block(index)
        data = block[offset : offset + len(buffer)]
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)

    def _block(self, index: int) -> bytes:
        if index not in self._blocks:
            first = index * self.block_size
            last = min(first + self.block_size, self.size) - 1
            if len(self._blocks) >= self.keep_blocks:
                del self._blocks[next(iter(self._blocks))]  # the oldest
            self._blocks[index] = self._fetch(first, last)
        return self._blocks[index]

    def _fetch(self, first: int, last: int) -> bytes:
        """Bytes ``first..last`` inclusive, retrying transient failures."""
        expected = last - first + 1
        for attempt in range(1, self.attempts + 1):
            self.requests += 1
            wait = self.backoff * attempt
            try:
                request = Request(self.url, headers={"Range": f"bytes={first}-{last}"})
                with urlopen(request, timeout=self.timeout) as resp:
                    if resp.status != 206:
                        raise RangeError(f"expected 206 Partial Content, got {resp.status}")
                    data: bytes = resp.read()
                if len(data) != expected:
                    raise OSError(f"short read: {len(data)} of {expected} bytes")
                return data
            except RangeError:
                raise
            except HTTPError as exc:
                if exc.code not in RATE_LIMITED or attempt == self.attempts:
                    raise
                retry_after = exc.headers.get("Retry-After", "")
                wait = float(retry_after) if retry_after.isdigit() else 30.0 * attempt
                log.warning("rate limited (%d); waiting %.0f s", exc.code, wait)
            except (OSError, http.client.HTTPException) as exc:  # incl. a dropped connection
                if attempt == self.attempts:
                    raise
                log.warning("range %d-%d failed (%s); retry %d", first, last, exc, attempt)
            time.sleep(wait)
        raise AssertionError("unreachable")


def split_members(names: list[str], prefix: str, count: int) -> list[str]:
    """The first ``count`` image files directly inside ``prefix``, in name order."""
    images = sorted(
        name
        for name in names
        if name.startswith(prefix)
        and "/" not in name[len(prefix) :]
        and Path(name).suffix.lower() in IMAGE_SUFFIXES
    )
    return images[:count]


def fetch_sample(
    url: str, size: int, prefix: str, count: int, out_dir: Path, *, block_size: int = BLOCK_SIZE
) -> list[dict[str, object]]:
    """Write the first ``count`` images under ``prefix`` to ``out_dir``; returns their records."""
    remote = HTTPRangeFile(url, size, block_size=block_size)
    with zipfile.ZipFile(io.BufferedReader(remote)) as zf:
        members = split_members(zf.namelist(), prefix, count)
        if len(members) < count:
            raise ValueError(f"{prefix}: only {len(members)} images, {count} requested")
        out_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, object]] = []
        for name in members:
            data = zf.read(name)  # checks the member's CRC-32
            (out_dir / Path(name).name).write_bytes(data)
            records.append(
                {
                    "member": name,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            log.info("fetched %s (%d bytes)", name, len(data))
    log.info("%d frames in %d range requests", len(records), remote.requests)
    return records


class ManifestMismatchError(RuntimeError):
    """A fetched file does not match the committed image manifest."""


def manifest_digests(images_csv: Path, split: str) -> dict[str, str]:
    """Archive-relative path -> SHA-256 for every image and mask of ``split``."""
    digests = {}
    with images_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] == split:
                digests[row["image_path"]] = row["image_sha256"]
                digests[row["mask_path"]] = row["mask_sha256"]
    return digests


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_whole_split(
    url: str,
    size: int,
    digests: dict[str, str],
    out_root: Path,
    *,
    workers: int = 4,
    block_size: int = BLOCK_SIZE,
) -> dict[str, int]:
    """Fetch every file in ``digests`` to ``out_root/<archive path>``, verifying each digest.

    Missing files are read in archive order, split into ``workers`` contiguous
    runs, each with its own reader, so every block is fetched once.
    """
    missing = []
    for name in sorted(digests):
        target = out_root / name
        if not (target.exists() and _sha256(target) == digests[name]):
            missing.append(name)
    kept = len(digests) - len(missing)
    if not missing:
        return {"files": len(digests), "fetched": 0, "kept": kept}

    def reader() -> zipfile.ZipFile:
        return zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(url, size, block_size=block_size)))

    with reader() as zf:
        offsets = {info.filename: info.header_offset for info in zf.infolist()}
    missing.sort(key=lambda name: offsets[name])
    per_run = -(-len(missing) // max(1, workers))
    runs = [missing[i : i + per_run] for i in range(0, len(missing), per_run)]
    done = threading.Lock()
    fetched = [0]

    def fetch_run(names: list[str]) -> None:
        with reader() as zf:
            for name in names:
                data = zf.read(name)  # checks the member's CRC-32
                if hashlib.sha256(data).hexdigest() != digests[name]:
                    raise ManifestMismatchError(f"{name}: SHA-256 differs from the image manifest")
                target = out_root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                part = target.with_name(target.name + ".part")
                part.write_bytes(data)
                part.replace(target)
                with done:
                    fetched[0] += 1
                    if fetched[0] % 100 == 0 or fetched[0] == len(missing):
                        log.info("%d / %d files fetched", fetched[0], len(missing))

    with ThreadPoolExecutor(max_workers=len(runs)) as pool:
        for future in [pool.submit(fetch_run, run) for run in runs]:
            future.result()
    return {"files": len(digests), "fetched": fetched[0], "kept": kept}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--split", choices=ALLOWED_SPLITS, default="val")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--count", type=int, default=20, help="the first N frames, flat")
    mode.add_argument("--whole", action="store_true", help="every image and mask of the split")
    parser.add_argument("--workers", type=int, default=4, help="parallel readers (--whole)")
    parser.add_argument("--out", type=Path, required=True, help="directory for the frames")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_data_config(args.config)
    if args.whole:
        images_csv = config.paths.manifests_dir / f"{config.source.name}-images.csv"
        counts = fetch_whole_split(
            config.source.url,
            config.source.size_bytes,
            manifest_digests(images_csv, args.split),
            args.out,
            workers=args.workers,
        )
        log.info("%s: %s", args.split, counts)
        return 0
    layout = config.layout
    prefix = f"{layout.root_subdir}/{args.split}/{layout.images_subdir}/"
    records = fetch_sample(
        config.source.url, config.source.size_bytes, prefix, args.count, args.out
    )
    index = {
        "source": {"url": config.source.url, "md5": config.source.md5},
        "split": args.split,
        "frames": records,
    }
    (args.out / "frames.json").write_text(json.dumps(index, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
