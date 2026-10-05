"""SQLite-backed audit jobs, results, artifacts, and reviewer feedback.

Concurrency model (one API process, one or a few workers):

* SQLite runs in WAL mode so readers do not block the writer.
* A worker **claims** an audit inside ``BEGIN IMMEDIATE``, setting a lease and
  incrementing the attempt counter. If the worker dies, the lease expires and
  another worker can claim the audit; past ``max_attempts`` the audit fails
  with an explicit reason.
* Image results are written with ``... WHERE status = 'pending'`` *and* a check
  that the writer still holds the lease, so a retried or stale worker can never
  produce a duplicate completed result.
* Feedback is **append-only**: a trigger rejects updates, so the original
  prediction and every later human decision stay distinguishable.
"""

import json
import sqlite3
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);

CREATE TABLE IF NOT EXISTS audits (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'complete', 'partial', 'failed')),
    camera_id TEXT NOT NULL,
    profile TEXT NOT NULL,
    inspection_region TEXT NOT NULL,
    model_version TEXT NOT NULL,
    preprocessing_version TEXT NOT NULL,
    review_policy_version TEXT NOT NULL,
    training_consent INTEGER NOT NULL DEFAULT 0,
    idempotency_key TEXT UNIQUE,
    request_hash TEXT NOT NULL,
    lease_owner TEXT,
    lease_expires_at REAL,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    expires_at TEXT NOT NULL,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS audits_status ON audits (status, created_at);

CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    audit_id TEXT NOT NULL REFERENCES audits (id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    original_filename TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    upload_path TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('pending', 'complete', 'failed')),
    error TEXT,
    cache_key TEXT,
    result TEXT,
    completed_at TEXT
);
CREATE INDEX IF NOT EXISTS images_audit ON images (audit_id, position);
CREATE INDEX IF NOT EXISTS images_cache ON images (cache_key) WHERE status = 'complete';

CREATE TABLE IF NOT EXISTS artifacts (
    id TEXT PRIMARY KEY,
    audit_id TEXT NOT NULL REFERENCES audits (id) ON DELETE CASCADE,
    image_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    content_type TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audit_id TEXT NOT NULL REFERENCES audits (id) ON DELETE CASCADE,
    image_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    reviewer TEXT,
    decision TEXT NOT NULL
        CHECK (decision IN ('confirmed', 'rejected', 'corrected', 'needs_retake')),
    materials_present TEXT NOT NULL,
    correction_artifact_id TEXT,
    note TEXT,
    model_version TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS feedback_append_only BEFORE UPDATE ON feedback
BEGIN
    SELECT RAISE(ABORT, 'feedback is append-only');
END;

CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY,
    model_version TEXT,
    last_seen REAL NOT NULL
);
"""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


class IdempotencyConflict(RuntimeError):
    """The idempotency key was reused with a different request."""


@dataclass(frozen=True)
class NewImage:
    original_filename: str
    sha256: str
    width: int
    height: int
    upload_path: str
    content_type: str


@dataclass(frozen=True)
class Claim:
    audit_id: str
    worker_id: str
    attempt: int


class JobStore:
    def __init__(self, path: Path, *, lease_seconds: float = 120.0, max_attempts: int = 3):
        self.path = path
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.executescript(SCHEMA)
            if db.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0] == 0:
                db.execute("INSERT INTO schema_version VALUES (?)", (SCHEMA_VERSION,))

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=30000")
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
            except BaseException:
                db.execute("ROLLBACK")
                raise
            db.execute("COMMIT")

    # --- creation and lookup -------------------------------------------------

    def create_audit(
        self,
        *,
        camera_id: str,
        profile: str,
        inspection_region: Sequence[Sequence[float]],
        model_version: str,
        preprocessing_version: str,
        review_policy_version: str,
        images: Sequence[NewImage],
        request_hash: str,
        idempotency_key: str | None = None,
        training_consent: bool = False,
        retention_hours: float = 24.0,
        audit_id: str | None = None,
        image_ids: Sequence[str] | None = None,
    ) -> tuple[str, bool]:
        """Create an audit; returns ``(audit_id, created)``.

        Reusing an idempotency key with the same request returns the existing
        audit (``created`` False); with a different request it raises.
        """
        with self._transaction() as db:
            if idempotency_key is not None:
                row = db.execute(
                    "SELECT id, request_hash FROM audits WHERE idempotency_key = ?",
                    (idempotency_key,),
                ).fetchone()
                if row is not None:
                    if row["request_hash"] != request_hash:
                        raise IdempotencyConflict(idempotency_key)
                    return str(row["id"]), False
            audit_id = audit_id or new_id("audit")
            expires = datetime.now(UTC) + timedelta(hours=retention_hours)
            db.execute(
                """INSERT INTO audits (id, created_at, status, camera_id, profile,
                   inspection_region, model_version, preprocessing_version,
                   review_policy_version, training_consent, idempotency_key, request_hash,
                   expires_at) VALUES (?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    audit_id,
                    _now_iso(),
                    camera_id,
                    profile,
                    json.dumps([list(v) for v in inspection_region]),
                    model_version,
                    preprocessing_version,
                    review_policy_version,
                    int(training_consent),
                    idempotency_key,
                    request_hash,
                    expires.isoformat(timespec="seconds"),
                ),
            )
            ids = list(image_ids) if image_ids else [new_id("img") for _ in images]
            for position, (image_id, image) in enumerate(zip(ids, images, strict=True)):
                db.execute(
                    """INSERT INTO images (id, audit_id, position, original_filename, sha256,
                       width, height, upload_path, status)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
                    (
                        image_id,
                        audit_id,
                        position,
                        image.original_filename,
                        image.sha256,
                        image.width,
                        image.height,
                        image.upload_path,
                    ),
                )
                db.execute(
                    """INSERT INTO artifacts (id, audit_id, image_id, kind, path, content_type,
                       created_at) VALUES (?, ?, ?, 'original', ?, ?, ?)""",
                    (
                        f"original_{image_id}",
                        audit_id,
                        image_id,
                        image.upload_path,
                        image.content_type,
                        _now_iso(),
                    ),
                )
            return audit_id, True

    def get_audit(self, audit_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM audits WHERE id = ?", (audit_id,)).fetchone()
            if row is None:
                return None
            audit = dict(row)
            audit["inspection_region"] = json.loads(audit["inspection_region"])
            audit["training_consent"] = bool(audit["training_consent"])
            counts = db.execute(
                "SELECT status, COUNT(*) AS n FROM images WHERE audit_id = ? GROUP BY status",
                (audit_id,),
            ).fetchall()
            audit["image_counts"] = {r["status"]: r["n"] for r in counts}
            return audit

    def list_images(self, audit_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM images WHERE audit_id = ? ORDER BY position", (audit_id,)
            ).fetchall()
        return [self._image_dict(r) for r in rows]

    def get_image(self, audit_id: str, image_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM images WHERE audit_id = ? AND id = ?", (audit_id, image_id)
            ).fetchone()
        return self._image_dict(row) if row else None

    @staticmethod
    def _image_dict(row: sqlite3.Row) -> dict[str, Any]:
        image = dict(row)
        image["result"] = json.loads(image["result"]) if image["result"] else None
        return image

    def pending_count(self) -> int:
        with self._connect() as db:
            row = db.execute(
                "SELECT COUNT(*) FROM audits WHERE status IN ('queued', 'running')"
            ).fetchone()
        return int(row[0])

    def oldest_pending_age(self) -> float:
        with self._connect() as db:
            row = db.execute(
                "SELECT MIN(created_at) FROM audits WHERE status IN ('queued', 'running')"
            ).fetchone()
        if row[0] is None:
            return 0.0
        return (datetime.now(UTC) - datetime.fromisoformat(row[0])).total_seconds()

    # --- worker side ---------------------------------------------------------

    def heartbeat(self, worker_id: str, model_version: str | None) -> None:
        with self._connect() as db:
            db.execute(
                """INSERT INTO workers (id, model_version, last_seen) VALUES (?, ?, ?)
                   ON CONFLICT (id) DO UPDATE SET model_version = excluded.model_version,
                   last_seen = excluded.last_seen""",
                (worker_id, model_version, time.time()),
            )

    def live_workers(self, within_seconds: float) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM workers WHERE last_seen >= ?", (time.time() - within_seconds,)
            ).fetchall()
        return [dict(r) for r in rows]

    def claim_next(self, worker_id: str, now: float | None = None) -> Claim | None:
        """Claim the oldest claimable audit, or None. Exhausted audits are failed."""
        now = time.time() if now is None else now
        with self._transaction() as db:
            while True:
                row = db.execute(
                    """SELECT id, attempts FROM audits
                       WHERE status IN ('queued', 'running')
                         AND (lease_expires_at IS NULL OR lease_expires_at < ?)
                       ORDER BY created_at, rowid LIMIT 1""",
                    (now,),
                ).fetchone()
                if row is None:
                    return None
                if row["attempts"] >= self.max_attempts:
                    self._fail_audit(db, row["id"], "max_attempts_exceeded")
                    continue
                db.execute(
                    """UPDATE audits SET status = 'running', lease_owner = ?,
                       lease_expires_at = ?, attempts = attempts + 1 WHERE id = ?""",
                    (worker_id, now + self.lease_seconds, row["id"]),
                )
                return Claim(row["id"], worker_id, row["attempts"] + 1)

    def extend_lease(self, claim: Claim, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        with self._connect() as db:
            cur = db.execute(
                """UPDATE audits SET lease_expires_at = ? WHERE id = ? AND lease_owner = ?
                   AND status = 'running'""",
                (now + self.lease_seconds, claim.audit_id, claim.worker_id),
            )
            return cur.rowcount == 1

    def find_cached_result(self, cache_key: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                """SELECT * FROM images WHERE cache_key = ? AND status = 'complete'
                   ORDER BY completed_at LIMIT 1""",
                (cache_key,),
            ).fetchone()
        return self._image_dict(row) if row else None

    def complete_image(
        self,
        claim: Claim,
        image_id: str,
        result: dict[str, Any],
        cache_key: str,
        artifacts: Sequence[tuple[str, str, str]],
    ) -> bool:
        """Record a result once. False if already complete or the lease was lost."""
        with self._transaction() as db:
            cur = db.execute(
                """UPDATE images SET status = 'complete', result = ?, cache_key = ?,
                   completed_at = ?, error = NULL
                   WHERE id = ? AND audit_id = ? AND status = 'pending'
                     AND EXISTS (SELECT 1 FROM audits WHERE id = ? AND lease_owner = ?
                                 AND status = 'running')""",
                (
                    json.dumps(result),
                    cache_key,
                    _now_iso(),
                    image_id,
                    claim.audit_id,
                    claim.audit_id,
                    claim.worker_id,
                ),
            )
            if cur.rowcount != 1:
                return False
            for kind, path, content_type in artifacts:
                db.execute(
                    """INSERT OR REPLACE INTO artifacts (id, audit_id, image_id, kind, path,
                       content_type, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        f"{kind}_{image_id}",
                        claim.audit_id,
                        image_id,
                        kind,
                        path,
                        content_type,
                        _now_iso(),
                    ),
                )
            return True

    def release_lease(self, claim: Claim) -> None:
        """Give up a claim after a transient failure so the audit is retried promptly."""
        with self._connect() as db:
            db.execute(
                "UPDATE audits SET lease_expires_at = 0 WHERE id = ? AND lease_owner = ?",
                (claim.audit_id, claim.worker_id),
            )

    def fail_image(self, claim: Claim, image_id: str, error: str) -> bool:
        with self._connect() as db:
            cur = db.execute(
                """UPDATE images SET status = 'failed', error = ?, completed_at = ?
                   WHERE id = ? AND audit_id = ? AND status = 'pending'
                     AND EXISTS (SELECT 1 FROM audits WHERE id = ? AND lease_owner = ?)""",
                (error, _now_iso(), image_id, claim.audit_id, claim.audit_id, claim.worker_id),
            )
            return cur.rowcount == 1

    def finish_audit(self, claim: Claim) -> str | None:
        """Set the final status once no image is pending; returns it, or None."""
        with self._transaction() as db:
            owner = db.execute(
                "SELECT lease_owner, status FROM audits WHERE id = ?", (claim.audit_id,)
            ).fetchone()
            if owner is None or owner["lease_owner"] != claim.worker_id:
                return None
            counts = {
                r["status"]: r["n"]
                for r in db.execute(
                    "SELECT status, COUNT(*) AS n FROM images WHERE audit_id = ? GROUP BY status",
                    (claim.audit_id,),
                )
            }
            if counts.get("pending"):
                return None
            done, failed = counts.get("complete", 0), counts.get("failed", 0)
            status = "complete" if failed == 0 else ("partial" if done else "failed")
            db.execute(
                """UPDATE audits SET status = ?, completed_at = ?, lease_owner = NULL,
                   lease_expires_at = NULL WHERE id = ?""",
                (status, _now_iso(), claim.audit_id),
            )
            return status

    def fail_audit(self, audit_id: str, error: str) -> None:
        with self._transaction() as db:
            self._fail_audit(db, audit_id, error)

    @staticmethod
    def _fail_audit(db: sqlite3.Connection, audit_id: str, error: str) -> None:
        db.execute(
            """UPDATE images SET status = 'failed', error = ?, completed_at = ?
               WHERE audit_id = ? AND status = 'pending'""",
            (error, _now_iso(), audit_id),
        )
        db.execute(
            """UPDATE audits SET status = CASE WHEN EXISTS (SELECT 1 FROM images
               WHERE audit_id = ? AND status = 'complete') THEN 'partial' ELSE 'failed' END,
               error = ?, completed_at = ?, lease_owner = NULL, lease_expires_at = NULL
               WHERE id = ?""",
            (audit_id, error, _now_iso(), audit_id),
        )

    # --- artifacts, feedback, deletion ---------------------------------------

    def get_artifact(self, audit_id: str, artifact_id: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM artifacts WHERE audit_id = ? AND id = ?", (audit_id, artifact_id)
            ).fetchone()
        return dict(row) if row else None

    def add_artifact(
        self, audit_id: str, image_id: str, kind: str, path: str, content_type: str
    ) -> str:
        artifact_id = new_id(kind)
        with self._connect() as db:
            db.execute(
                """INSERT INTO artifacts (id, audit_id, image_id, kind, path, content_type,
                   created_at) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (artifact_id, audit_id, image_id, kind, path, content_type, _now_iso()),
            )
        return artifact_id

    def add_feedback(
        self,
        *,
        audit_id: str,
        image_id: str,
        decision: str,
        materials_present: Sequence[str],
        model_version: str,
        reviewer: str | None = None,
        note: str | None = None,
        correction_artifact_id: str | None = None,
    ) -> int:
        with self._connect() as db:
            cur = db.execute(
                """INSERT INTO feedback (audit_id, image_id, created_at, reviewer, decision,
                   materials_present, correction_artifact_id, note, model_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    audit_id,
                    image_id,
                    _now_iso(),
                    reviewer,
                    decision,
                    json.dumps(sorted(materials_present)),
                    correction_artifact_id,
                    note,
                    model_version,
                ),
            )
            return int(cur.lastrowid or 0)

    def list_feedback(self, audit_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM feedback WHERE audit_id = ? ORDER BY id", (audit_id,)
            ).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["materials_present"] = json.loads(item["materials_present"])
            out.append(item)
        return out

    def delete_audit(self, audit_id: str) -> list[str]:
        """Delete an audit and return the file paths that belonged to it."""
        with self._transaction() as db:
            paths = [
                r[0]
                for r in db.execute(
                    "SELECT upload_path FROM images WHERE audit_id = ? UNION "
                    "SELECT path FROM artifacts WHERE audit_id = ?",
                    (audit_id, audit_id),
                )
            ]
            db.execute("DELETE FROM audits WHERE id = ?", (audit_id,))
        return paths

    def expired_audits(self, now: datetime | None = None) -> list[str]:
        now = now or datetime.now(UTC)
        with self._connect() as db:
            rows = db.execute(
                """SELECT id FROM audits WHERE expires_at < ?
                   AND status NOT IN ('queued', 'running')""",
                (now.isoformat(timespec="seconds"),),
            ).fetchall()
        return [r[0] for r in rows]
