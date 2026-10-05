import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from beltwatch.api.settings import Settings
from beltwatch.inference.neural import save_inference_checkpoint
from beltwatch.inference.preprocessing import PreprocessConfig
from beltwatch.jobs.store import JobStore
from beltwatch.jobs.worker import Worker
from beltwatch.models.registry import ModelConfig, build_model
from beltwatch.release.bundle import activate, build_release
from tests.integration.conftest import POLICY, png_bytes, upload


def test_interrupted_worker_is_recovered_without_duplicates(
    client: TestClient, settings: Settings, store: JobStore
) -> None:
    audit_id = upload(client, png_bytes(1), png_bytes(2)).json()["audit_id"]
    crashed = store.claim_next("crashed-worker", now=0.0)  # claims at t=0, then dies
    assert crashed is not None

    rescuer = Worker(settings, store, worker_id="rescuer")
    assert rescuer.run_once()  # the 30 s lease expired long ago: reclaimed, attempt 2

    audit = store.get_audit(audit_id)
    assert audit is not None
    assert (audit["status"], audit["attempts"]) == ("complete", 2)
    image = store.list_images(audit_id)[0]
    assert not store.complete_image(crashed, image["id"], {"late": True}, "k", [])
    assert "late" not in store.list_images(audit_id)[0]["result"]


def test_missing_pinned_release_fails_explicitly(
    client: TestClient, settings: Settings, worker: Worker
) -> None:
    audit_id = upload(client, png_bytes()).json()["audit_id"]
    shutil.rmtree(settings.releases_dir / "test-1")

    worker.run_once()

    audit = client.get(f"/v1/audits/{audit_id}").json()
    assert audit["status"] == "failed"
    assert audit["error"].startswith("model_release_unavailable")


def test_transient_errors_retry_then_fail(
    client: TestClient, settings: Settings, store: JobStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_id = upload(client, png_bytes()).json()["audit_id"]
    worker = Worker(settings, store, worker_id="flaky")
    _, predictor = worker.resolve("test-1")

    def boom(rgb: np.ndarray) -> np.ndarray:
        raise RuntimeError("simulated out-of-memory")

    monkeypatch.setattr(predictor, "predict_proba", boom)
    for _ in range(settings.max_attempts):
        assert worker.run_once()
        assert store.get_audit(audit_id)["status"] in {"running", "queued"}  # type: ignore[index]
    worker.run_once()

    audit = store.get_audit(audit_id)
    assert audit is not None and audit["status"] == "failed"
    assert audit["error"] == "max_attempts_exceeded" and audit["attempts"] == settings.max_attempts


def test_identical_request_is_served_from_cache(
    client: TestClient, settings: Settings, worker: Worker
) -> None:
    first = upload(client, png_bytes(5)).json()["audit_id"]
    worker.run_once()
    second = upload(client, png_bytes(5)).json()["audit_id"]
    worker.run_once()

    detail = client.get(f"/v1/audits/{second}").json()["images"][0]
    full = client.get(detail["detail_url"]).json()
    assert full["cache_hit"] is True
    assert client.get(full["overlay_url"]).status_code == 200
    client.delete(f"/v1/audits/{first}")
    assert client.get(full["overlay_url"]).status_code == 200  # cached artifacts were copied


def test_corrupt_upload_fails_only_that_image(
    client: TestClient, settings: Settings, worker: Worker
) -> None:
    audit_id = upload(client, png_bytes(1), png_bytes(2)).json()["audit_id"]
    first = client.get(f"/v1/audits/{audit_id}").json()["images"][0]["image_id"]
    next((settings.data_dir / "uploads" / audit_id).glob(f"{first}.*")).write_bytes(b"corrupted")

    worker.run_once()

    audit = client.get(f"/v1/audits/{audit_id}").json()
    assert audit["status"] == "partial"
    failed = next(i for i in audit["images"] if i["image_id"] == first)
    assert failed["status"] == "failed" and failed["error"].startswith("invalid_image")


def test_neural_release_end_to_end(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model_config = ModelConfig(pretrained=False)
    ckpt = tmp_path / "best.pt"
    save_inference_checkpoint(
        ckpt, build_model(model_config), model_config, PreprocessConfig(long_side=96), {}
    )
    releases = tmp_path / "releases"
    build_release(releases, version="unet-test", kind="unet", model_path=ckpt, review_policy=POLICY)
    activate(releases, "unet-test", "test")
    settings = Settings(data_dir=tmp_path / "var", releases_dir=releases)
    store = JobStore(settings.database_path)
    worker = Worker(settings, store, worker_id="w")
    store.heartbeat("w", None)
    from beltwatch.api.app import create_app

    with TestClient(create_app(settings)) as client:
        audit_id = upload(client, png_bytes(7)).json()["audit_id"]
        worker.run_once()
        image = client.get(f"/v1/audits/{audit_id}").json()["images"][0]
        detail = client.get(image["detail_url"]).json()

    assert detail["status"] == "complete" and detail["model_version"] == "unet-test"
    assert 0.0 <= detail["uncertain_pixel_fraction"] <= 1.0
    assert 0.0 <= detail["total_target_coverage"] <= 1.0
