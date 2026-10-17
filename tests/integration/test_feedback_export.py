import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from beltwatch.api.settings import Settings
from beltwatch.data.feedback import (
    FeedbackExportError,
    eligible_candidates,
    export_candidates,
    load_reviews,
    main,
    record_review,
    review_status,
)
from beltwatch.jobs.worker import Worker
from tests.integration.conftest import png_bytes, upload


def mask_b64(shape: tuple[int, int], value: int) -> str:
    buf = io.BytesIO()
    Image.fromarray(np.full(shape, value, dtype=np.uint8)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def feedback(client: TestClient, audit_id: str, image_id: str, **body: object) -> None:
    r = client.post(f"/v1/audits/{audit_id}/feedback", json={"image_id": image_id, **body})
    assert r.status_code == 201, r.text


@pytest.fixture
def reviewed(client: TestClient, worker: Worker) -> dict[str, list[str]]:
    """A consented audit (3 images) and an unconsented one, both reviewed."""
    consented = upload(
        client, png_bytes(1), png_bytes(2), png_bytes(3), options='{"training_consent": true}'
    ).json()
    other = upload(client, png_bytes(4)).json()
    worker.run_once()
    worker.run_once()
    a, b = consented["audit_id"], other["audit_id"]
    img = [i["image_id"] for i in consented["images"]]
    feedback(client, a, img[0], decision="confirmed", materials_present=["cardboard"])
    feedback(
        client,
        a,
        img[0],
        decision="corrected",
        materials_present=["metal"],
        corrected_mask_png_base64=mask_b64((64, 96), 4),
        reviewer="ana",
    )
    feedback(client, a, img[1], decision="confirmed", materials_present=["soft_plastic"])
    # img[2] has no feedback; the unconsented audit has a correction that must not leak.
    other_img = other["images"][0]["image_id"]
    feedback(
        client,
        b,
        other_img,
        decision="corrected",
        materials_present=["cardboard"],
        corrected_mask_png_base64=mask_b64((64, 96), 1),
    )
    return {"consented": img, "other": [other_img]}


def test_export_takes_only_consented_reviewed_images(
    settings: Settings, reviewed: dict[str, list[str]], tmp_path: Path
) -> None:
    out = tmp_path / "batch-1"
    summary = export_candidates(settings.database_path, settings.data_dir, out)
    assert summary.candidates == 2 and not summary.skipped
    assert summary.by_label_kind == {"image_level": 1, "pixel_mask": 1}

    status = review_status(out)
    img = reviewed["consented"]
    assert set(status) == {img[0], img[1]}
    corrected, confirmed = status[img[0]], status[img[1]]
    assert corrected["label_kind"] == "pixel_mask" and confirmed["label_kind"] == "image_level"
    assert confirmed["mask_file"] is None
    # The full history is kept, and the original prediction stays separate from it.
    assert [f["decision"] for f in corrected["feedback"]] == ["confirmed", "corrected"]
    assert corrected["prediction"]["model_version"] == "test-1"
    assert corrected["model_version"] == "test-1"
    assert all(c["status"] == "pending_annotation_review" for c in status.values())

    image_bytes = (out / corrected["image_file"]).read_bytes()
    assert image_bytes == png_bytes(1)
    mask = np.asarray(Image.open(out / corrected["mask_file"]))
    assert mask.shape == (64, 96) and set(np.unique(mask)) == {4}

    record = json.loads((out / "export.json").read_text())
    assert record["export_version"] == "feedback-export-1"
    assert record["label_map"]["0"] == "background"
    assert eligible_candidates(out) == []  # nothing is eligible before annotation review


def test_annotation_review_is_append_only_and_gates_eligibility(
    settings: Settings, reviewed: dict[str, list[str]], tmp_path: Path
) -> None:
    out = tmp_path / "batch-1"
    export_candidates(settings.database_path, settings.data_dir, out)
    corrected, confirmed = reviewed["consented"][:2]
    record_review(out, corrected, "reject", "lee", "boundary misses the can rim")
    assert eligible_candidates(out) == []
    record_review(out, corrected, "accept", "lee", "re-checked after zooming in")
    record_review(out, confirmed, "accept", "lee", "film is visible")

    assert [r["decision"] for r in load_reviews(out)] == ["reject", "accept", "accept"]
    # An accepted image-level label is not a pixel label.
    assert [c["candidate_id"] for c in eligible_candidates(out)] == [corrected]
    assert review_status(out)[confirmed]["status"] == "accepted"

    with pytest.raises(FeedbackExportError, match="no candidate"):
        record_review(out, reviewed["other"][0], "accept", "lee", "not consented")
    with pytest.raises(FeedbackExportError, match="reason"):
        record_review(out, corrected, "accept", "lee", "  ")


def test_export_refuses_a_used_batch_and_skips_tampered_uploads(
    settings: Settings, reviewed: dict[str, list[str]], tmp_path: Path
) -> None:
    out = tmp_path / "batch-1"
    export_candidates(settings.database_path, settings.data_dir, out)
    with pytest.raises(FeedbackExportError, match="immutable"):
        export_candidates(settings.database_path, settings.data_dir, out)

    uploads = sorted((settings.data_dir / "uploads").rglob("*.png"))
    for path in uploads:
        if path.read_bytes() == png_bytes(2):
            path.write_bytes(png_bytes(99))
    summary = export_candidates(settings.database_path, settings.data_dir, tmp_path / "batch-2")
    assert summary.candidates == 1
    assert summary.skipped == [
        {"image_id": reviewed["consented"][1], "reason": "upload does not match its SHA-256"}
    ]


def test_cli(
    settings: Settings,
    reviewed: dict[str, list[str]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "batch-cli"
    assert main(["export", "--data-dir", str(settings.data_dir), "--out", str(out)]) == 0
    corrected = reviewed["consented"][0]
    assert (
        main(
            [
                "review",
                str(out),
                corrected,
                "--decision",
                "accept",
                "--annotator",
                "lee",
                "--reason",
                "ok",
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert main(["eligible", str(out)]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["eligible_pixel_masks"] == [corrected]
    assert listing["status"] == {"accepted": 1, "pending_annotation_review": 1}
    assert (
        main(
            [
                "review",
                str(out),
                "missing",
                "--decision",
                "accept",
                "--annotator",
                "lee",
                "--reason",
                "ok",
            ]
        )
        == 1
    )
