import sqlite3
from pathlib import Path

import pytest

from beltwatch.jobs.store import IdempotencyConflict, JobStore, NewImage


def make_store(tmp_path: Path, **kwargs: float) -> JobStore:
    return JobStore(tmp_path / "db.sqlite3", **kwargs)  # type: ignore[arg-type]


def create(store: JobStore, n: int = 2, key: str | None = None, request_hash: str = "h") -> str:
    images = [
        NewImage(f"{i}.png", f"sha{i}", 96, 64, f"uploads/x/{i}.png", "image/png") for i in range(n)
    ]
    audit_id, _ = store.create_audit(
        camera_id="cam",
        profile="paper-stream-v1",
        inspection_region=[[0, 0], [1, 0], [1, 1]],
        model_version="m1",
        preprocessing_version="pp-1",
        review_policy_version="p1",
        images=images,
        request_hash=request_hash,
        idempotency_key=key,
    )
    return audit_id


def test_idempotent_creation(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    first = create(store, key="k1")
    assert create(store, key="k1") == first
    with pytest.raises(IdempotencyConflict):
        create(store, key="k1", request_hash="other")


def test_claim_order_and_lease_exclusivity(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    a, b = create(store), create(store)

    first = store.claim_next("w1", now=100.0)
    second = store.claim_next("w2", now=100.0)

    assert first is not None and first.audit_id == a and first.attempt == 1
    assert second is not None and second.audit_id == b
    assert store.claim_next("w3", now=100.0) is None


def test_expired_lease_is_reclaimed_and_counts_attempts(tmp_path: Path) -> None:
    store = make_store(tmp_path, lease_seconds=10, max_attempts=2)
    audit = create(store)

    assert store.claim_next("w1", now=0.0) is not None
    assert store.claim_next("w2", now=5.0) is None  # lease still held
    reclaimed = store.claim_next("w2", now=11.0)
    assert reclaimed is not None and reclaimed.attempt == 2

    assert store.claim_next("w3", now=30.0) is None  # attempts exhausted -> failed
    final = store.get_audit(audit)
    assert final is not None
    assert final["status"] == "failed" and final["error"] == "max_attempts_exceeded"


def test_results_are_recorded_once_and_stale_workers_are_blocked(tmp_path: Path) -> None:
    store = make_store(tmp_path, lease_seconds=10)
    audit = create(store, n=1)
    image = store.list_images(audit)[0]["id"]
    stale = store.claim_next("w1", now=0.0)
    fresh = store.claim_next("w2", now=20.0)
    assert stale is not None and fresh is not None

    assert not store.complete_image(stale, image, {"v": 1}, "key", [])  # lost lease
    assert store.complete_image(fresh, image, {"v": 2}, "key", [("overlay", "p.png", "image/png")])
    assert not store.complete_image(fresh, image, {"v": 3}, "key", [])  # no duplicate

    assert store.list_images(audit)[0]["result"] == {"v": 2}
    assert store.finish_audit(stale) is None
    assert store.finish_audit(fresh) == "complete"


def test_final_status_reflects_failures(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    audit = create(store, n=2)
    claim = store.claim_next("w")
    assert claim is not None
    one, two = (i["id"] for i in store.list_images(audit))
    store.complete_image(claim, one, {}, "k", [])
    assert store.finish_audit(claim) is None  # one image still pending
    store.fail_image(claim, two, "invalid_image")
    assert store.finish_audit(claim) == "partial"


def test_feedback_is_append_only(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    audit = create(store, n=1)
    image = store.list_images(audit)[0]["id"]
    fid = store.add_feedback(
        audit_id=audit,
        image_id=image,
        decision="confirmed",
        materials_present=["metal"],
        model_version="m1",
    )
    store.add_feedback(
        audit_id=audit,
        image_id=image,
        decision="corrected",
        materials_present=[],
        model_version="m1",
    )

    with (
        sqlite3.connect(store.path) as db,
        pytest.raises(sqlite3.DatabaseError, match="append-only"),
    ):
        db.execute("UPDATE feedback SET decision = 'rejected' WHERE id = ?", (fid,))
    assert [f["decision"] for f in store.list_feedback(audit)] == ["confirmed", "corrected"]


def test_delete_returns_files_and_cascades(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    audit = create(store, n=2)

    paths = store.delete_audit(audit)

    assert sorted(paths) == ["uploads/x/0.png", "uploads/x/1.png"]
    assert store.get_audit(audit) is None
    assert store.list_images(audit) == []
