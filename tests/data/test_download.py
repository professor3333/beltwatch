import hashlib
import json
import threading
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from beltwatch.data import download as dl
from beltwatch.data.config import SourceConfig, load_data_config

PAYLOAD = bytes(range(256)) * 64  # 16 KiB


@dataclass
class FakeServer:
    url: str
    payload: bytes = PAYLOAD
    honor_range: bool = True
    # Close the connection after this many body bytes, for the next `truncations` responses.
    truncate_after: int = 4000
    truncations: int = 0
    range_headers: list[str | None] = field(default_factory=list)


@pytest.fixture
def server() -> Iterator[FakeServer]:
    state = FakeServer(url="")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            range_header = self.headers.get("Range")
            state.range_headers.append(range_header)
            body, status = state.payload, 200
            if range_header and state.honor_range:
                start = int(range_header.removeprefix("bytes=").rstrip("-"))
                body, status = state.payload[start:], 206
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if state.truncations:
                state.truncations -= 1
                body = body[: state.truncate_after]
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    state.url = f"http://127.0.0.1:{httpd.server_address[1]}/archive.zip"
    yield state
    httpd.shutdown()
    httpd.server_close()


def make_source(url: str, payload: bytes = PAYLOAD) -> SourceConfig:
    # The config model requires https; tests bypass validation for the local server.
    return SourceConfig.model_construct(
        name="fixture",
        zenodo_record=1,
        version="0",
        doi="10.0/fixture",
        filename="archive.zip",
        url=url,
        size_bytes=len(payload),
        md5=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
    )


def test_repo_config_is_valid_and_pinned() -> None:
    config = load_data_config(Path("configs/data.yaml"))
    assert config.source.filename == "zerowaste-f-final.zip"
    assert config.source.size_bytes == 7_518_242_799
    assert config.source.md5 == "e26e31a58080bca6782dca0e56074c5d"


def test_config_rejects_malformed_md5() -> None:
    with pytest.raises(ValueError, match="md5"):
        SourceConfig(
            name="x",
            zenodo_record=1,
            version="0",
            doi="d",
            filename="a.zip",
            url="https://example.org/a.zip",
            size_bytes=1,
            md5="not-a-checksum",
        )


def test_download_verifies_and_renames(server: FakeServer, tmp_path: Path) -> None:
    archive = dl.download(make_source(server.url), tmp_path)

    assert archive == tmp_path / "archive.zip"
    assert archive.read_bytes() == PAYLOAD
    assert not (tmp_path / "archive.zip.part").exists()


def test_download_resumes_partial_file(server: FakeServer, tmp_path: Path) -> None:
    (tmp_path / "archive.zip.part").write_bytes(PAYLOAD[:5000])

    archive = dl.download(make_source(server.url), tmp_path)

    assert server.range_headers == ["bytes=5000-"]
    assert archive.read_bytes() == PAYLOAD


def test_download_resumes_after_server_closes_early(server: FakeServer, tmp_path: Path) -> None:
    server.truncations = 2

    archive = dl.download(make_source(server.url), tmp_path, retry_wait=0)

    assert server.range_headers == [None, "bytes=4000-", "bytes=8000-"]
    assert archive.read_bytes() == PAYLOAD


def test_incomplete_download_keeps_partial_file(server: FakeServer, tmp_path: Path) -> None:
    server.truncations = 99

    with pytest.raises(dl.IncompleteDownloadError, match="12000 of 16384"):
        dl.download(make_source(server.url), tmp_path, max_attempts=3, retry_wait=0)

    assert (tmp_path / "archive.zip.part").read_bytes() == PAYLOAD[:12000]
    assert not (tmp_path / "archive.zip").exists()


def test_download_retries_network_errors(tmp_path: Path) -> None:
    # Nothing listens on this port, so every attempt fails with a connection error.
    source = make_source("http://127.0.0.1:9/archive.zip")
    (tmp_path / "archive.zip.part").write_bytes(PAYLOAD[:5000])

    with pytest.raises(dl.IncompleteDownloadError, match="5000 of 16384"):
        dl.download(source, tmp_path, max_attempts=2, retry_wait=0)

    assert (tmp_path / "archive.zip.part").read_bytes() == PAYLOAD[:5000]


def test_download_restarts_when_server_ignores_range(server: FakeServer, tmp_path: Path) -> None:
    server.honor_range = False
    (tmp_path / "archive.zip.part").write_bytes(PAYLOAD[:5000])

    archive = dl.download(make_source(server.url), tmp_path)

    assert archive.read_bytes() == PAYLOAD


def test_download_rejects_corrupt_payload(server: FakeServer, tmp_path: Path) -> None:
    source = make_source(server.url)
    server.payload = PAYLOAD[:-1] + b"\x00"

    with pytest.raises(dl.ChecksumError, match="md5"):
        dl.download(source, tmp_path)

    assert not (tmp_path / "archive.zip").exists()
    assert not (tmp_path / "archive.zip.part").exists()


def test_existing_valid_archive_skips_network(server: FakeServer, tmp_path: Path) -> None:
    (tmp_path / "archive.zip").write_bytes(PAYLOAD)

    dl.download(make_source(server.url), tmp_path)

    assert server.range_headers == []


def test_existing_corrupt_archive_is_left_untouched(server: FakeServer, tmp_path: Path) -> None:
    bad = tmp_path / "archive.zip"
    bad.write_bytes(b"truncated")

    with pytest.raises(dl.ChecksumError, match="bytes"):
        dl.download(make_source(server.url), tmp_path)

    assert bad.read_bytes() == b"truncated"
    assert server.range_headers == []


def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_extract_and_record_counts(tmp_path: Path) -> None:
    archive = make_zip(
        tmp_path / "a.zip",
        {
            "train/data/1.PNG": b"x",
            "train/data/2.png": b"x",
            "train/labels.json": b"{}",
            "test/data/3.jpg": b"x",
        },
    )
    source = make_source("https://example.org/a.zip")

    extracted = dl.extract(archive, tmp_path / "raw")
    record_path = dl.write_download_record(tmp_path / "manifests", source, extracted)

    record = json.loads(record_path.read_text())
    assert record["image_counts_by_directory"] == {"test/data": 1, "train/data": 2}
    assert record["total_images"] == 3
    assert record["source"]["md5"] == source.md5
    assert not (tmp_path / "raw.tmp").exists()


def test_extract_is_idempotent(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "a.zip", {"x.png": b"x"})
    out = dl.extract(archive, tmp_path / "raw")
    (out / "marker").write_text("kept")

    dl.extract(archive, tmp_path / "raw")

    assert (out / "marker").read_text() == "kept"


def test_extract_rejects_path_traversal(tmp_path: Path) -> None:
    archive = make_zip(tmp_path / "evil.zip", {"../escape.png": b"x"})

    with pytest.raises(dl.UnsafeArchiveError, match="escapes"):
        dl.extract(archive, tmp_path / "raw")

    assert not (tmp_path / "raw").exists()
    assert not (tmp_path / "raw.tmp").exists()
    assert not (tmp_path / "escape.png").exists()
