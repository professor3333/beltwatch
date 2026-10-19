"""Fetch a few frames of one split from the pinned archive without downloading it.

The archive is read through HTTP Range requests: the zip's central directory
first, then only the requested members, whose CRC-32 the zip reader checks.
It is for machines that need a small real sample (a benchmark host, CI)
rather than the 7.5 GB download. The frames are the first ``--count`` image
files of the split's image directory in name order, the same frames a
benchmark over the extracted directory reads.

Usage::

    uv run python -m beltwatch.data.sample --split val --count 20 --out frames/

The locked test split is refused. The frames are dataset images: keep them
out of git and out of published artifacts.
"""

import argparse
import hashlib
import io
import json
import logging
import time
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

from beltwatch.data.config import load_data_config
from beltwatch.data.download import IMAGE_SUFFIXES

log = logging.getLogger(__name__)

ALLOWED_SPLITS = ("train", "val")  # never test: it is locked
BLOCK_SIZE = 1 << 20


class RangeError(RuntimeError):
    """The server did not return the requested byte range."""


class HTTPRangeFile(io.RawIOBase):
    """A read-only, seekable view of a remote file of known size, one Range request per read."""

    def __init__(
        self, url: str, size: int, *, timeout: float = 60, attempts: int = 5, backoff: float = 2
    ) -> None:
        self.url, self.size = url, size
        self.timeout, self.attempts, self.backoff = timeout, attempts, backoff
        self.position = 0
        self.requests = 0

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
        end = min(self.position + len(buffer), self.size)
        if end <= self.position:
            return 0
        data = self._fetch(self.position, end - 1)
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)

    def _fetch(self, first: int, last: int) -> bytes:
        """Bytes ``first..last`` inclusive, retrying transient failures."""
        expected = last - first + 1
        for attempt in range(1, self.attempts + 1):
            self.requests += 1
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
            except OSError as exc:
                if attempt == self.attempts:
                    raise
                log.warning("range %d-%d failed (%s); retry %d", first, last, exc, attempt)
                time.sleep(self.backoff * attempt)
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
    remote = HTTPRangeFile(url, size)
    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=block_size)) as zf:
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--split", choices=ALLOWED_SPLITS, default="val")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True, help="directory for the frames")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_data_config(args.config)
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
