import subprocess
import sys
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from beltwatch.api.app import create_app
from beltwatch.api.settings import Settings
from beltwatch.demo import demo_env, fetch_configured_release, supervise
from beltwatch.jobs.worker import Worker
from beltwatch.release.bundle import ReleaseError

from .conftest import png_bytes


def test_public_demo_refuses_training_consent(settings: Settings, worker: Worker) -> None:
    demo = settings.model_copy(update={"public_demo": True})
    with TestClient(create_app(demo)) as c:
        service = c.get("/v1/model").json()["service"]
        assert service == {
            "public_demo": True,
            "retention_hours": 24.0,
            "max_files_per_audit": 20,
            "accepts_training_consent": False,
        }
        files = [("files", ("a.png", png_bytes(), "image/png"))]
        refused = c.post("/v1/audits", files=files, data={"options": '{"training_consent": true}'})
        assert refused.status_code == 422
        assert refused.json()["error"]["code"] == "training_consent_unavailable"
        assert c.post("/v1/audits", files=files).status_code == 201


def test_self_hosted_service_accepts_consent(client: TestClient) -> None:
    assert client.get("/v1/model").json()["service"]["accepts_training_consent"] is True


def test_demo_env_hides_the_hub_token_and_locks_metrics() -> None:
    env = demo_env({"HF_TOKEN": "secret", "PATH": "/bin"})
    assert "HF_TOKEN" not in env and env["PATH"] == "/bin"
    assert env["BELTWATCH_PUBLIC_DEMO"] == "true"
    assert len(env["BELTWATCH_METRICS_TOKEN"]) >= 32
    assert demo_env({"BELTWATCH_METRICS_TOKEN": "given"})["BELTWATCH_METRICS_TOKEN"] == "given"


def test_demo_needs_the_whole_release_pin(tmp_path: Path) -> None:
    with pytest.raises(ReleaseError, match="BELTWATCH_DEMO_RELEASE_REVISION"):
        fetch_configured_release(
            {"BELTWATCH_DEMO_RELEASE_REPO": "o/r", "BELTWATCH_DEMO_RELEASE_VERSION": "v1"},
            tmp_path,
        )


def test_supervise_reports_a_stopped_child_and_a_requested_stop() -> None:
    sleeper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    quitter = subprocess.Popen([sys.executable, "-c", "pass"])
    try:
        assert supervise([sleeper, quitter], threading.Event(), poll=0.05) == 1
        stopped = threading.Event()
        threading.Timer(0.1, stopped.set).start()
        assert supervise([sleeper], stopped, poll=0.05) == 0
    finally:
        sleeper.kill()
        sleeper.wait()
