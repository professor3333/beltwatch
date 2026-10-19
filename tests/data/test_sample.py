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
    truncations: int = 0  # responses that close the connection early
    rate_limits: int = 0  # responses that answer 429 with Retry-After: 0
    ranges: list[str] = field(default_factory=list)


@pytest.fixture
def server() -> Iterator[RangeServer]:
    state = RangeServer(url="", payload=make_archive())

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            header = self.headers.get("Range", "")
            state.ranges.append(header)
            if state.rate_limits:
                state.rate_limits -= 1
                self.send_response(429)
                self.send_header("Retry-After", "0")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body, status = state.payload, 200
            if header and state.honor_range:
                first, last = (int(v) for v in header.removeprefix("bytes=").split("-"))
                body, status = state.payload[first : last + 1], 206
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if state.truncations:
                state.truncations -= 1
                body = body[: len(body) // 2]
                self.close_connection = True
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


def test_a_dropped_connection_is_retried(server: RangeServer, tmp_path: Path) -> None:
    server.truncations = 2
    remote = sample.HTTPRangeFile(server.url, len(server.payload), backoff=0)

    assert remote._fetch(0, 99) == server.payload[:100]
    assert remote.requests == 3


def test_a_rate_limit_waits_as_asked_and_retries(server: RangeServer, tmp_path: Path) -> None:
    server.rate_limits = 2
    remote = sample.HTTPRangeFile(server.url, len(server.payload), backoff=0)

    assert remote._fetch(0, 99) == server.payload[:100]
    assert remote.requests == 3


def test_reads_inside_one_block_need_one_request(server: RangeServer) -> None:
    remote = sample.HTTPRangeFile(server.url, len(server.payload), block_size=1 << 20)

    first, second = remote.read(10), remote.read(10)
    remote.seek(5)

    assert first + second == server.payload[:20] and remote.read(5) == server.payload[5:10]
    assert remote.requests == 1


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


def val_digests() -> dict[str, str]:
    root = "splits_final_deblurred/val"
    digests = {}
    for i in (1, 2, 3):
        digests[f"{root}/data/val_{i:02d}.png"] = hashlib.sha256(f"val{i}".encode()).hexdigest()
        digests[f"{root}/sem_seg/val_{i:02d}.png"] = hashlib.sha256(b"mask").hexdigest()
    return digests


def test_whole_split_fetches_every_listed_file_at_its_archive_path(
    server: RangeServer, tmp_path: Path
) -> None:
    digests = val_digests()

    counts = sample.fetch_whole_split(server.url, len(server.payload), digests, tmp_path, workers=3)

    assert counts == {"files": 6, "fetched": 6, "kept": 0}
    assert len(server.ranges) <= 4  # one block per reader: the index, then 3 runs
    assert (tmp_path / "splits_final_deblurred/val/data/val_02.png").read_bytes() == b"val2"
    assert not list(tmp_path.rglob("*test*")) and not list(tmp_path.rglob("*.part"))

    again = sample.fetch_whole_split(server.url, len(server.payload), digests, tmp_path)
    assert again == {"files": 6, "fetched": 0, "kept": 6}


def test_whole_split_refetches_a_damaged_file(server: RangeServer, tmp_path: Path) -> None:
    digests = val_digests()
    sample.fetch_whole_split(server.url, len(server.payload), digests, tmp_path)
    damaged = tmp_path / "splits_final_deblurred/val/data/val_01.png"
    damaged.write_bytes(b"corrupt")

    counts = sample.fetch_whole_split(server.url, len(server.payload), digests, tmp_path)

    assert counts["fetched"] == 1 and damaged.read_bytes() == b"val1"


def test_whole_split_refuses_a_file_that_differs_from_the_manifest(
    server: RangeServer, tmp_path: Path
) -> None:
    digests = {"splits_final_deblurred/val/data/val_01.png": "0" * 64}

    with pytest.raises(sample.ManifestMismatchError):
        sample.fetch_whole_split(server.url, len(server.payload), digests, tmp_path)
    assert not (tmp_path / "splits_final_deblurred/val/data/val_01.png").exists()


def test_manifest_digests_cover_images_and_masks_of_one_split(tmp_path: Path) -> None:
    images_csv = tmp_path / "images.csv"
    images_csv.write_text(
        "image_id,split,image_path,mask_path,image_sha256,mask_sha256\n"
        "val/a,val,v/data/a.PNG,v/sem_seg/a.PNG,ia,ma\n"
        "test/b,test,t/data/b.PNG,t/sem_seg/b.PNG,ib,mb\n"
    )

    assert sample.manifest_digests(images_csv, "val") == {
        "v/data/a.PNG": "ia",
        "v/sem_seg/a.PNG": "ma",
    }
