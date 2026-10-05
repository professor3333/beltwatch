import importlib.util
import socket
import threading
import time
from pathlib import Path

import pytest
import uvicorn

from beltwatch.api.app import create_app
from beltwatch.api.settings import Settings
from beltwatch.jobs.worker import Worker
from beltwatch.release.activate import main as activate_main
from beltwatch.release.bundle import build_release
from tests.integration.conftest import POLICY

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "deploy_smoke.py"
spec = importlib.util.spec_from_file_location("deploy_smoke", SCRIPT)
assert spec and spec.loader
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def live_service(settings: Settings, worker: Worker):  # type: ignore[no-untyped-def]
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(settings), port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    stop = threading.Event()

    def work() -> None:
        while not stop.is_set():
            if not worker.run_once():
                time.sleep(0.1)

    threading.Thread(target=work, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    stop.set()
    server.should_exit = True


def test_synthetic_png_is_valid() -> None:
    import io

    from PIL import Image

    with Image.open(io.BytesIO(smoke.synthetic_png())) as img:
        assert img.size == (160, 96) and img.mode == "RGB"


def test_smoke_passes_against_a_live_service(live_service: str) -> None:
    assert (
        smoke.main(
            ["--base-url", live_service, "--expect-version", "test-1", "--ready-timeout", "20"]
        )
        == 0
    )


def test_smoke_detects_wrong_version_and_follows_rollback(
    live_service: str, settings: Settings
) -> None:
    with pytest.raises(SystemExit, match="model version"):
        smoke.main(
            ["--base-url", live_service, "--expect-version", "other", "--ready-timeout", "20"]
        )

    build_release(
        settings.releases_dir,
        version="test-2",
        kind="all-background",
        model_path=None,
        review_policy=POLICY,
    )
    releases = ["--releases-dir", str(settings.releases_dir)]
    activate_main(["test-2", "--reason", "upgrade", *releases])
    assert smoke.main(["--base-url", live_service, "--expect-version", "test-2"]) == 0
    activate_main(["test-1", "--reason", "rollback", *releases])
    assert smoke.main(["--base-url", live_service, "--expect-version", "test-1"]) == 0
