"""Download, verify, and extract the pinned ZeroWaste-f release.

Usage::

    uv run python -m beltwatch.data.download [--config configs/data.yaml] [--skip-extract]

The archive is streamed to a ``.part`` file (resumed with an HTTP Range request
after an interruption), verified against the pinned size and MD5, and only then
renamed into place. Extraction goes to a temporary directory that is renamed
into place when complete, so a half-extracted dataset never looks finished.
A download record with the *measured* image count is written to the manifests
directory.
"""

import argparse
import hashlib
import json
import logging
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from urllib.request import Request, urlopen

from beltwatch import __version__
from beltwatch.data.config import SourceConfig, load_data_config

log = logging.getLogger(__name__)

CHUNK_SIZE = 1 << 20
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg"})


class _Readable(Protocol):
    def read(self, size: int, /) -> bytes: ...


class ChecksumError(RuntimeError):
    """A file does not match its pinned size or checksum."""


class UnsafeArchiveError(RuntimeError):
    """An archive member would be written outside the extraction directory."""


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path: Path, source: SourceConfig) -> None:
    """Raise :class:`ChecksumError` unless ``path`` matches the pinned size and MD5."""
    size = path.stat().st_size
    if size != source.size_bytes:
        raise ChecksumError(f"{path}: expected {source.size_bytes} bytes, found {size}")
    actual = md5_file(path)
    if actual != source.md5:
        raise ChecksumError(f"{path}: expected md5 {source.md5}, found {actual}")


def download(source: SourceConfig, downloads_dir: Path, *, timeout: float = 60.0) -> Path:
    """Download ``source`` into ``downloads_dir`` and return the verified archive path.

    An existing archive is verified and reused. If it fails verification it is
    left untouched and :class:`ChecksumError` is raised, so a bad file is never
    silently replaced or deleted.
    """
    downloads_dir.mkdir(parents=True, exist_ok=True)
    dest = downloads_dir / source.filename
    if dest.exists():
        verify(dest, source)
        log.info("%s already downloaded and verified", dest)
        return dest

    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > source.size_bytes:
        log.warning("discarding oversized partial download %s", part)
        part.unlink()
        offset = 0

    if offset < source.size_bytes:
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with urlopen(Request(source.url, headers=headers), timeout=timeout) as resp:
            if offset and resp.status != 206:
                log.warning("server ignored the resume request; restarting download")
                offset = 0
            if offset:
                log.info("resuming %s at byte %d", source.filename, offset)
            _stream_to(resp, part, offset=offset, total=source.size_bytes)

    try:
        verify(part, source)
    except ChecksumError:
        part.unlink()
        raise
    part.replace(dest)
    log.info("downloaded and verified %s", dest)
    return dest


def _stream_to(resp: _Readable, part: Path, *, offset: int, total: int) -> None:
    written = offset
    next_report = 0
    with part.open("ab" if offset else "wb") as fh:
        while chunk := resp.read(CHUNK_SIZE):
            fh.write(chunk)
            written += len(chunk)
            percent = written * 100 // total
            if percent >= next_report:
                log.info("%s: %d%% (%d / %d bytes)", part.name, percent, written, total)
                next_report = percent - percent % 5 + 5


def extract(archive: Path, out_dir: Path) -> Path:
    """Extract ``archive`` to ``out_dir`` atomically; reuse ``out_dir`` if it already exists."""
    if out_dir.exists():
        log.info("%s already extracted", out_dir)
        return out_dir

    tmp = out_dir.with_name(out_dir.name + ".tmp")
    if tmp.exists():
        log.warning("removing incomplete extraction %s", tmp)
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    with zipfile.ZipFile(archive) as zf:
        root = tmp.resolve()
        for name in zf.namelist():
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                shutil.rmtree(tmp)
                raise UnsafeArchiveError(f"{archive}: member {name!r} escapes the archive root")
        zf.extractall(tmp)

    tmp.replace(out_dir)
    log.info("extracted %s to %s", archive.name, out_dir)
    return out_dir


def count_images(root: Path) -> dict[str, int]:
    """Count image files per directory, keyed by POSIX path relative to ``root``."""
    counts: dict[str, int] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            key = path.parent.relative_to(root).as_posix()
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def write_download_record(manifests_dir: Path, source: SourceConfig, extracted_dir: Path) -> Path:
    """Record the verified source and the measured image counts."""
    counts = count_images(extracted_dir)
    record = {
        "source": source.model_dump(),
        "archive_verified": {"size_bytes": source.size_bytes, "md5": source.md5},
        "extracted_dir": extracted_dir.as_posix(),
        "image_counts_by_directory": counts,
        "total_images": sum(counts.values()),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "beltwatch_version": __version__,
    }
    manifests_dir.mkdir(parents=True, exist_ok=True)
    out = manifests_dir / f"{source.name}-download.json"
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    log.info("found %d images; wrote %s", record["total_images"], out)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--skip-extract", action="store_true", help="download and verify only")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_data_config(args.config)
    archive = download(config.source, config.paths.downloads_dir)
    if not args.skip_extract:
        extracted = extract(archive, config.paths.raw_dir)
        write_download_record(config.paths.manifests_dir, config.source, extracted)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
