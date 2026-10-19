import hashlib
import io
import threading
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from beltwatch.data import sample

PREFIX = "splits_final_deblurred/val/data/"


def make_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for split in ("train", "val", "test"):
            root = f"splits_final_deblurred/{split}"
            for i in (3, 1, 2):  # written out of order
                zf.writestr(f"{root}/data/{split}_{i:02d}.png", f"{split}{i}")
                zf.writestr(f"{root}/sem_seg/{split}_{i:02d}.png", "mask")
        zf.writestr(f"{PREFIX}labels.json", "{}")
        zf.writestr(f"{PREFIX}nested/deeper.png", "not directly inside")
    return buffer.getvalue()


@dataclass
class RangeServer:
    url: str
    payload: bytes
    honor_range: bool = True
    ranges: list[str] = field(default_factory=list)


@pytest.fixture
def server() -> Iterator[RangeServer]:
    state = RangeServer(url="", payload=make_archive())

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            header = self.headers.get("Range", "")
            state.ranges.append(header)
            body, status = state.payload, 200
            if header and state.honor_range:
                first, last = (int(v) for v in header.removeprefix("bytes=").split("-"))
                body, status = state.payload[first : last + 1], 206
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
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


def test_fetches_the_first_frames_of_the_split_in_name_order(
    server: RangeServer, tmp_path: Path
) -> None:
    records = sample.fetch_sample(
        server.url, len(server.payload), PREFIX, 2, tmp_path, block_size=64
    )

    assert [r["member"] for r in records] == [f"{PREFIX}val_01.png", f"{PREFIX}val_02.png"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["val_01.png", "val_02.png"]
    assert (tmp_path / "val_01.png").read_bytes() == b"val1"
    assert records[0]["sha256"] == hashlib.sha256(b"val1").hexdigest()
    assert server.ranges and all(r.startswith("bytes=") for r in server.ranges)


def test_skips_nested_and_non_image_members() -> None:
    names = [f"{PREFIX}b.png", f"{PREFIX}labels.json", f"{PREFIX}x/a.png", f"{PREFIX}a.JPG"]

    assert sample.split_members(names, PREFIX, 5) == [f"{PREFIX}a.JPG", f"{PREFIX}b.png"]


def test_refuses_more_frames_than_the_split_has(server: RangeServer, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only 3 images"):
        sample.fetch_sample(server.url, len(server.payload), PREFIX, 4, tmp_path)


def test_a_server_ignoring_range_requests_is_an_error(server: RangeServer, tmp_path: Path) -> None:
    server.honor_range = False

    with pytest.raises(sample.RangeError):
        sample.fetch_sample(server.url, len(server.payload), PREFIX, 1, tmp_path)


def test_the_locked_test_split_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        sample.main(["--split", "test", "--out", str(tmp_path)])
