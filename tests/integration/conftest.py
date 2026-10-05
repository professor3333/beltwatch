import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from beltwatch.api.app import create_app
from beltwatch.api.settings import Settings
from beltwatch.jobs.store import JobStore
from beltwatch.jobs.worker import Worker
from beltwatch.release.bundle import activate, build_release
from beltwatch.review.policy import load_review_policy

REPO = Path(__file__).resolve().parents[2]
POLICY = load_review_policy(REPO / "configs/review_policy.yaml")


def png_bytes(seed: int = 0, size: tuple[int, int] = (96, 64), fmt: str = "PNG") -> bytes:
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(40, 216, (size[1], size[0], 3), dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    releases = tmp_path / "releases"
    build_release(
        releases, version="test-1", kind="all-background", model_path=None, review_policy=POLICY
    )
    activate(releases, "test-1", "test fixture")
    return Settings(
        data_dir=tmp_path / "var", releases_dir=releases, lease_seconds=30, max_attempts=3
    )


@pytest.fixture
def store(settings: Settings) -> JobStore:
    return JobStore(
        settings.database_path,
        lease_seconds=settings.lease_seconds,
        max_attempts=settings.max_attempts,
    )


@pytest.fixture
def worker(settings: Settings, store: JobStore) -> Worker:
    w = Worker(settings, store, worker_id="worker-test")
    store.heartbeat(w.worker_id, None)
    return w


@pytest.fixture
def client(settings: Settings, worker: Worker) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c


def upload(client: TestClient, *images: bytes, options: str = "{}", key: str | None = None):  # type: ignore[no-untyped-def]
    files = [("files", (f"frame_{i}.png", data, "image/png")) for i, data in enumerate(images)]
    headers = {"Idempotency-Key": key} if key else {}
    return client.post("/v1/audits", files=files, data={"options": options}, headers=headers)
