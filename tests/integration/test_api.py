import base64
import io
import json
import struct
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from beltwatch.api.app import create_app
from beltwatch.api.settings import Settings
from beltwatch.jobs.worker import Worker
from tests.integration.conftest import png_bytes, upload


def test_full_audit_workflow(client: TestClient, worker: Worker, settings: Settings) -> None:
    created = upload(client, png_bytes(1), png_bytes(2))
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["status"] == "queued" and body["model_version"] == "test-1"
    audit_id = body["audit_id"]

    assert client.get(f"/v1/audits/{audit_id}").json()["progress"]["pending"] == 2
    assert worker.run_once()

    audit = client.get(f"/v1/audits/{audit_id}").json()
    assert audit["status"] == "complete"
    assert audit["progress"] == {"total": 2, "complete": 2, "failed": 0, "pending": 0}
    image_id = audit["images"][0]["image_id"]

    detail = client.get(f"/v1/audits/{audit_id}/images/{image_id}").json()
    assert set(detail["visible_coverage"]) == {
        "cardboard",
        "soft_plastic",
        "rigid_plastic",
        "metal",
    }
    assert detail["total_target_coverage"] == 0.0  # the fixture release predicts background
    assert detail["route"] in {"none", "review"}
    for kind in ("original", "overlay", "mask", "uncertainty"):
        art = client.get(detail[f"{kind}_url"])
        assert art.status_code == 200 and art.headers["content-type"] == "image/png"

    fb = client.post(
        f"/v1/audits/{audit_id}/feedback",
        json={
            "image_id": image_id,
            "decision": "corrected",
            "materials_present": ["metal"],
            "reviewer": "qa-1",
            "note": "small can near the edge",
        },
    )
    assert fb.status_code == 201
    report = client.get(f"/v1/audits/{audit_id}/report").json()
    assert report["images"][0]["feedback"][0]["decision"] == "corrected"
    assert "not contamination by weight" in report["measurement"]
    csv_text = client.get(f"/v1/audits/{audit_id}/report", params={"format": "csv"}).text
    assert "corrected" in csv_text and "coverage_metal" in csv_text.splitlines()[0]

    assert client.delete(f"/v1/audits/{audit_id}").status_code == 204
    assert client.get(f"/v1/audits/{audit_id}").status_code == 404
    assert not (settings.data_dir / "uploads" / audit_id).exists()


def error_code(response) -> str:  # type: ignore[no-untyped-def]
    return str(response.json()["error"]["code"])


def test_upload_validation(client: TestClient, settings: Settings) -> None:
    bad_region = json.dumps({"inspection_region": [[0, 0], [2, 0], [1, 1]]})
    assert error_code(upload(client, png_bytes(), options=bad_region)) == "invalid_options"
    r = client.post("/v1/audits", files=[("files", ("x.png", b"not an image", "image/png"))])
    assert (r.status_code, error_code(r)) == (422, "invalid_image")
    gif = png_bytes(fmt="GIF")
    assert error_code(upload(client, gif)) == "unsupported_image_type"
    assert error_code(upload(client, png_bytes(size=(20, 20)))) == "image_too_small"
    sliver = json.dumps({"inspection_region": [[0.5, 0.5], [0.5001, 0.5], [0.5, 0.5001]]})
    assert error_code(upload(client, png_bytes(), options=sliver)) == "inspection_region_too_small"
    many = [png_bytes(i) for i in range(settings.max_files_per_audit + 1)]
    assert error_code(upload(client, *many)) == "invalid_file_count"


def test_oversized_upload_is_rejected(settings: Settings, worker: Worker) -> None:
    small = settings.model_copy(update={"max_file_bytes": 1000})
    with TestClient(create_app(small)) as c:
        r = upload(c, png_bytes())
    assert (r.status_code, error_code(r)) == (413, "file_too_large")


def header_only_png(width: int, height: int) -> bytes:
    """A PNG whose header claims ``width x height`` but carries almost no data."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        crc = struct.pack(">I", zlib.crc32(kind + body))
        return struct.pack(">I", len(body)) + kind + body + crc

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"\x00" * 64)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


@pytest.mark.parametrize("size", [(8000, 6000), (20000, 20000)])
def test_pixel_limit_rejects_before_decoding(client: TestClient, size: tuple[int, int]) -> None:
    # 48 MP is over our 40 MP limit; 400 MP is also over Pillow's own bomb limit.
    r = upload(client, header_only_png(*size))

    assert (r.status_code, error_code(r)) == (413, "image_too_large")


def test_idempotency_key(client: TestClient) -> None:
    first = upload(client, png_bytes(1), key="abc")
    replay = upload(client, png_bytes(1), key="abc")
    conflict = upload(client, png_bytes(2), key="abc")

    assert first.status_code == 201 and replay.status_code == 200
    assert replay.json()["audit_id"] == first.json()["audit_id"]
    assert (conflict.status_code, error_code(conflict)) == (409, "idempotency_conflict")


def test_queue_full(settings: Settings, worker: Worker) -> None:
    with TestClient(create_app(settings.model_copy(update={"max_pending_audits": 1}))) as c:
        assert upload(c, png_bytes()).status_code == 201
        r = upload(c, png_bytes(3))
    assert (r.status_code, error_code(r)) == (503, "queue_full")
    assert r.headers["Retry-After"] == "30"


def test_no_worker_and_no_model_are_explicit(settings: Settings) -> None:
    with TestClient(create_app(settings)) as c:  # no heartbeat registered
        r = upload(c, png_bytes())
        assert (r.status_code, error_code(r)) == (503, "worker_unavailable")
        assert c.get("/health/ready").status_code == 503
    (settings.releases_dir / "active.json").unlink()
    no_worker_needed = settings.model_copy(update={"require_worker": False})
    with TestClient(create_app(no_worker_needed)) as c:
        r = upload(c, png_bytes())
        assert (r.status_code, error_code(r)) == (503, "model_unavailable")
        assert c.get("/v1/model").status_code == 503


def test_model_info_and_health(client: TestClient) -> None:
    info = client.get("/v1/model").json()
    assert info["version"] == "test-1" and info["model_kind"] == "all-background"
    assert any("not contamination by weight" in s for s in info["limitations"])
    assert info["review_policy"]["version"] == "review-policy-1"
    assert client.get("/health/live").json()["status"] == "alive"
    ready = client.get("/health/ready")
    assert ready.status_code == 200 and ready.json()["checks"]["model_release"] == "test-1"


def test_metrics_and_token(settings: Settings, worker: Worker) -> None:
    with TestClient(create_app(settings)) as c:
        text = c.get("/metrics").text
        assert "beltwatch_queue_length" in text and "beltwatch_live_workers 1.0" in text
    with TestClient(create_app(settings.model_copy(update={"metrics_token": "s3cret"}))) as c:
        assert c.get("/metrics").status_code == 401
        assert c.get("/metrics", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_feedback_rules_and_corrected_masks(client: TestClient, worker: Worker) -> None:
    audit_id = upload(client, png_bytes()).json()["audit_id"]
    image = client.get(f"/v1/audits/{audit_id}").json()["images"][0]
    pending = client.post(
        f"/v1/audits/{audit_id}/feedback",
        json={"image_id": image["image_id"], "decision": "confirmed"},
    )
    assert (pending.status_code, error_code(pending)) == (409, "image_not_complete")
    worker.run_once()

    def mask_b64(shape: tuple[int, int], value: int = 1) -> str:
        buf = io.BytesIO()
        Image.fromarray(np.full(shape, value, dtype=np.uint8)).save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    ok = client.post(
        f"/v1/audits/{audit_id}/feedback",
        json={
            "image_id": image["image_id"],
            "decision": "corrected",
            "materials_present": ["cardboard"],
            "corrected_mask_png_base64": mask_b64((64, 96)),
        },
    )
    assert ok.status_code == 201 and ok.json()["correction_artifact_id"]
    wrong_size = client.post(
        f"/v1/audits/{audit_id}/feedback",
        json={
            "image_id": image["image_id"],
            "decision": "corrected",
            "corrected_mask_png_base64": mask_b64((10, 10)),
        },
    )
    assert error_code(wrong_size) == "invalid_mask"
    bad_class = client.post(
        f"/v1/audits/{audit_id}/feedback",
        json={
            "image_id": image["image_id"],
            "decision": "corrected",
            "corrected_mask_png_base64": mask_b64((64, 96), 9),
        },
    )
    assert error_code(bad_class) == "invalid_mask"
    unknown = client.post(
        f"/v1/audits/{audit_id}/feedback", json={"image_id": "img_nope", "decision": "confirmed"}
    )
    assert error_code(unknown) == "image_not_found"


def test_exif_rotated_upload_uses_the_upright_grid(client: TestClient, worker: Worker) -> None:
    img = Image.fromarray(np.random.default_rng(0).integers(40, 216, (64, 96, 3), dtype=np.uint8))
    exif = Image.Exif()
    exif[0x0112] = 6  # stored 96x64 landscape, displayed 64x96 portrait
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    audit_id = upload(client, buf.getvalue()).json()["audit_id"]
    worker.run_once()
    image_id = client.get(f"/v1/audits/{audit_id}").json()["images"][0]["image_id"]
    info = client.get(f"/v1/audits/{audit_id}/images/{image_id}").json()

    mask = Image.open(io.BytesIO(client.get(info["mask_url"]).content))
    assert (info["width"], info["height"]) == mask.size == (64, 96)
    correction = io.BytesIO()
    Image.fromarray(np.zeros((96, 64), dtype=np.uint8)).save(correction, format="PNG")
    ok = client.post(
        f"/v1/audits/{audit_id}/feedback",
        json={
            "image_id": image_id,
            "decision": "corrected",
            "corrected_mask_png_base64": base64.b64encode(correction.getvalue()).decode(),
        },
    )
    assert ok.status_code == 201


def test_unknown_resources_are_404(client: TestClient) -> None:
    assert error_code(client.get("/v1/audits/audit_missing")) == "audit_not_found"
    audit_id = upload(client, png_bytes()).json()["audit_id"]
    assert (
        error_code(client.get(f"/v1/audits/{audit_id}/artifacts/overlay_x")) == "artifact_not_found"
    )
    # Encoded traversal never reaches the handler: routing rejects it with a plain 404.
    assert client.get(f"/v1/audits/{audit_id}/artifacts/..%2F..%2Fsecret").status_code == 404
    assert error_code(client.get(f"/v1/audits/{audit_id}/images/img_nope")) == "image_not_found"
