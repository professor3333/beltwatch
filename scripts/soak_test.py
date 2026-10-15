"""Reliability soak test for a running BeltWatch service (standard library only).

Submits many single-image audits, each with its own inspection region so that
none is served from the result cache, and keeps a bounded number in flight.
Optionally runs the inference worker itself and kills it with SIGKILL at a
fixed interval, restarting it at once, so recovery from lost leases is part of
the run. Some audits are re-sent with the same Idempotency-Key, which must
return the same audit.

When every audit has finished, it checks each one: final status, exactly one
image result, a retrievable overlay, and a report with one row per image.
With ``--database`` it also counts rows in the job store directly. The
targets are those of the design: at least 99% of audits complete and no
duplicate completions.

Usage::

    python scripts/soak_test.py --base-url http://localhost:8000 --audits 200 \\
        --images frame1.png frame2.png --worker-command "uv run python -m beltwatch.jobs.worker" \\
        --kill-every 60 --database var/beltwatch.sqlite3 --out soak.json

Exits non-zero if a target is missed.
"""

import argparse
import contextlib
import json
import os
import platform
import random
import shlex
import signal
import sqlite3
import subprocess
import sys
import time
import urllib.error
import uuid
from pathlib import Path
from typing import Any

from deploy_smoke import Smoke, synthetic_png

TERMINAL = {"complete", "partial", "failed"}
RETRYABLE = {"worker_unavailable", "queue_full", "model_unavailable"}


def region(rng: random.Random) -> list[list[float]]:
    """A random axis-aligned rectangle covering at least a quarter of the frame."""
    x0, y0 = round(rng.uniform(0, 0.25), 4), round(rng.uniform(0, 0.25), 4)
    x1, y1 = round(rng.uniform(0.75, 1), 4), round(rng.uniform(0.75, 1), 4)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def multipart(options: dict[str, Any], filename: str, image: bytes) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    disposition = "Content-Disposition: form-data; name="
    body = b"".join(
        [
            f'--{boundary}\r\n{disposition}"options"\r\n\r\n{json.dumps(options)}\r\n'.encode(),
            f'--{boundary}\r\n{disposition}"files"; filename="{filename}"\r\n'
            "Content-Type: image/png\r\n\r\n".encode()
            + image
            + b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    return body, f"multipart/form-data; boundary={boundary}"


class WorkerProcess:
    """Runs the worker command and kills it with SIGKILL every ``kill_every`` seconds.

    The command runs in its own process group, and the whole group is killed,
    so a wrapper such as ``uv run`` cannot leave the real worker running.
    """

    def __init__(self, command: str, kill_every: float | None) -> None:
        self.command = shlex.split(command)
        self.kill_every = kill_every
        self.kills = 0
        self.process: subprocess.Popen[bytes] | None = None
        self.started = 0.0

    def start(self) -> None:
        self.process = subprocess.Popen(self.command, start_new_session=True)
        self.started = time.monotonic()

    def _kill_group(self, sig: signal.Signals) -> None:
        assert self.process is not None
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.process.pid, sig)
        self.process.wait(timeout=30)

    def tick(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is not None:
            print(f"worker exited with {self.process.returncode}; restarting", flush=True)
            self._kill_group(signal.SIGKILL)  # anything it left behind
            self.start()
        elif self.kill_every and time.monotonic() - self.started >= self.kill_every:
            self._kill_group(signal.SIGKILL)
            self.kills += 1
            print(f"worker killed (#{self.kills}); restarting", flush=True)
            self.start()

    def stop(self, grace: float = 30) -> None:
        """SIGTERM the group, wait for all of it to exit, then SIGKILL what remains."""
        if self.process is None:
            return
        self._kill_group(signal.SIGTERM)
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            try:
                os.killpg(self.process.pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.2)
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal.SIGKILL)


class Soak:
    def __init__(self, smoke: Smoke, images: list[tuple[str, bytes]], seed: int) -> None:
        self.smoke = smoke
        self.images = images
        self.rng = random.Random(seed)
        self.retries = 0
        self.replays: list[bool] = []

    def create(self, index: int, replay: bool) -> tuple[str, float]:
        """Create audit ``index``, retrying while the service asks to; returns (id, submit time)."""
        name, image = self.images[index % len(self.images)]
        options = {"camera_id": "soak", "inspection_region": region(self.rng)}
        body, content_type = multipart(options, name, image)
        headers = {"Content-Type": content_type, "Idempotency-Key": f"soak-{uuid.uuid4().hex}"}
        submitted = time.monotonic()
        while True:
            try:
                status, created = self.smoke.json("POST", "/v1/audits", body=body, headers=headers)
                break
            except urllib.error.HTTPError as exc:
                error = json.loads(exc.read() or b"{}").get("error", {}).get("code")
                if exc.code == 503 and error in RETRYABLE:
                    self.retries += 1
                    time.sleep(2)
                    continue
                raise SystemExit(f"FAIL create audit {index}: HTTP {exc.code} {error}") from exc
        if status != 201:
            raise SystemExit(f"FAIL create audit {index}: HTTP {status}")
        if replay:  # the same request and key must return the same audit
            again_status, again = self.smoke.json("POST", "/v1/audits", body=body, headers=headers)
            self.replays.append(again_status == 200 and again["audit_id"] == created["audit_id"])
        return created["audit_id"], submitted

    def status(self, audit_id: str) -> dict[str, Any]:
        _, audit = self.smoke.json("GET", f"/v1/audits/{audit_id}")
        return audit

    def verify(self, audit: dict[str, Any]) -> list[str]:
        """Problems with one finished audit; empty if none."""
        audit_id, problems = audit["audit_id"], []
        images = audit["images"]
        if len(images) != 1:
            problems.append(f"{audit_id}: {len(images)} image results")
        if audit["status"] == "complete":
            image_id = images[0]["image_id"]
            status, _, data = self.smoke.request(
                "GET", f"/v1/audits/{audit_id}/artifacts/overlay_{image_id}"
            )
            if status != 200 or not data.startswith(b"\x89PNG"):
                problems.append(f"{audit_id}: overlay missing")
            _, report = self.smoke.json("GET", f"/v1/audits/{audit_id}/report")
            if len(report["images"]) != 1:
                problems.append(f"{audit_id}: report has {len(report['images'])} rows")
        return problems


def database_counts(path: Path, audit_ids: list[str]) -> dict[str, int]:
    """Row-level checks in the job store (opened read-only)."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    marks = ",".join("?" * len(audit_ids))
    one = db.execute(
        f"SELECT COUNT(*) FROM (SELECT audit_id FROM images WHERE audit_id IN ({marks}) "
        "GROUP BY audit_id HAVING COUNT(*) != 1)",
        audit_ids,
    ).fetchone()[0]
    pending = db.execute(
        f"SELECT COUNT(*) FROM images i JOIN audits a ON a.id = i.audit_id "
        f"WHERE a.id IN ({marks}) AND i.status = 'pending' "
        "AND a.status IN ('complete', 'partial', 'failed')",
        audit_ids,
    ).fetchone()[0]
    artifacts = db.execute(
        f"SELECT COUNT(*) FROM (SELECT image_id, kind FROM artifacts WHERE audit_id IN ({marks}) "
        "GROUP BY image_id, kind HAVING COUNT(*) > 1)",
        audit_ids,
    ).fetchone()[0]
    incomplete_artifacts = db.execute(
        f"SELECT COUNT(*) FROM images i WHERE i.audit_id IN ({marks}) AND i.status = 'complete' "
        "AND (SELECT COUNT(*) FROM artifacts r WHERE r.image_id = i.id "
        "AND r.kind IN ('overlay', 'mask', 'uncertainty')) != 3",
        audit_ids,
    ).fetchone()[0]
    retried = db.execute(
        f"SELECT COUNT(*) FROM audits WHERE id IN ({marks}) AND attempts > 1", audit_ids
    ).fetchone()[0]
    db.close()
    return {
        "audits_claimed_more_than_once": retried,
        "audits_without_exactly_one_image": one,
        "pending_images_in_finished_audits": pending,
        "duplicate_artifacts": artifacts,
        "complete_images_without_worker_artifacts": incomplete_artifacts,
    }


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reliability soak test of a BeltWatch service.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--audits", type=int, default=200)
    parser.add_argument("--in-flight", type=int, default=10, help="audits submitted, not finished")
    parser.add_argument("--images", type=Path, nargs="*", help="PNG frames (default: synthetic)")
    parser.add_argument("--worker-command", help="run the worker as this command")
    parser.add_argument("--kill-every", type=float, help="SIGKILL the worker every N seconds")
    parser.add_argument("--replay-every", type=int, default=10, help="re-send every Nth request")
    parser.add_argument("--database", type=Path, help="job store, for row-level checks")
    parser.add_argument("--timeout", type=float, default=3600, help="seconds for the whole run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="write the summary as JSON")
    args = parser.parse_args(argv)
    if args.kill_every and not args.worker_command:
        parser.error("--kill-every needs --worker-command")

    images = [(p.name, p.read_bytes()) for p in args.images or []] or [
        (f"synthetic-{i}.png", synthetic_png(seed=i)) for i in range(4)
    ]
    smoke = Smoke(args.base_url, insecure=False, timeout=60)
    worker = WorkerProcess(args.worker_command, args.kill_every) if args.worker_command else None
    if worker:
        worker.start()
    try:
        ready = smoke.wait_ready(300)
        print(f"ready: {ready['checks']}", flush=True)
        _, model = smoke.json("GET", "/v1/model")
        soak = Soak(smoke, images, args.seed)
        started = time.monotonic()
        deadline = started + args.timeout
        submitted: dict[str, float] = {}
        finished: dict[str, dict[str, Any]] = {}
        seconds: dict[str, float] = {}
        next_index = 0
        while len(finished) < args.audits:
            if time.monotonic() > deadline:
                raise SystemExit(f"FAIL timeout: {len(finished)} of {args.audits} finished")
            if worker:
                worker.tick()
            while next_index < args.audits and len(submitted) - len(finished) < args.in_flight:
                replay = args.replay_every > 0 and next_index % args.replay_every == 0
                audit_id, at = soak.create(next_index, replay)
                submitted[audit_id] = at
                next_index += 1
            for audit_id in [a for a in submitted if a not in finished]:
                audit = soak.status(audit_id)
                if audit["status"] in TERMINAL:
                    finished[audit_id] = audit
                    seconds[audit_id] = time.monotonic() - submitted[audit_id]
                    if len(finished) % 20 == 0:
                        print(f"{len(finished)}/{args.audits} finished", flush=True)
            time.sleep(0.5)
        wall = time.monotonic() - started
    finally:
        if worker:
            worker.stop()

    by_status: dict[str, int] = {}
    errors: dict[str, int] = {}
    problems: list[str] = []
    for audit in finished.values():
        by_status[audit["status"]] = by_status.get(audit["status"], 0) + 1
        for image in audit["images"]:
            if image["error"]:
                errors[image["error"]] = errors.get(image["error"], 0) + 1
        if audit["model_version"] != model["version"]:
            problems.append(f"{audit['audit_id']}: model {audit['model_version']}")
        problems += soak.verify(audit)
    counts = database_counts(args.database, list(finished)) if args.database else None
    reclaimed = counts.pop("audits_claimed_more_than_once") if counts else None
    duplicates = len(problems) + (sum(counts.values()) if counts else 0)
    success = by_status.get("complete", 0) / args.audits
    times = list(seconds.values())
    summary = {
        "model_version": model["version"],
        "audits": args.audits,
        "by_status": by_status,
        "image_errors": errors,
        "success_rate": success,
        "problems": problems,
        "database_checks": counts,
        "idempotent_replays": {"sent": len(soak.replays), "same_audit": sum(soak.replays)},
        "create_retries": soak.retries,
        "worker_kills": worker.kills if worker else 0,
        "audits_claimed_more_than_once": reclaimed,
        "seconds_to_finish": {"p50": percentile(times, 0.5), "p95": percentile(times, 0.95)},
        "wall_seconds": round(wall, 1),
        "audits_per_second": args.audits / wall,
        "in_flight": args.in_flight,
        "host": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
        "targets": {"success_rate": 0.99, "duplicate_completions": 0},
        "meets_targets": {
            "success_rate": success >= 0.99,
            "duplicate_completions": duplicates == 0 and all(soak.replays),
        },
    }
    text = json.dumps(summary, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0 if all(summary["meets_targets"].values()) else 1


if __name__ == "__main__":
    sys.exit(main())
