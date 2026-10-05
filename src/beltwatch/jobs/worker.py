"""The inference worker: claims audits and processes their images.

Behaviour, matching the design:

1. Claim a queued audit transactionally (with a lease and attempt counter).
2. Resolve the model release **pinned on the audit**. If it is unavailable, the
   audit fails with an explicit error; another model is never substituted.
3. For each pending image: decode, validate, predict, compute coverage and
   review routing, write artifacts atomically, and store the result (progress
   is saved after every image).
4. Invalid images fail individually with a reason. A transient error releases
   the lease so the audit is retried, up to the attempt limit.
5. Identical requests are served from the result cache (image hash, region,
   preprocessing, model, and review-policy versions).

Usage::

    uv run python -m beltwatch.jobs.worker
"""

import argparse
import logging
import os
import shutil
import socket
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from beltwatch.api.settings import Settings
from beltwatch.inference.predictor import Predictor
from beltwatch.inference.preprocessing import load_rgb
from beltwatch.jobs.pipeline import cache_key, process_image
from beltwatch.jobs.store import Claim, JobStore
from beltwatch.release.bundle import Release, ReleaseError, active_version, load_release

log = logging.getLogger(__name__)

ARTIFACT_KINDS = {"overlay": "image/png", "mask": "image/png", "uncertainty": "image/png"}


def _write_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    Image.fromarray(array).save(tmp, format="PNG")
    tmp.replace(path)


class Worker:
    def __init__(
        self, settings: Settings, store: JobStore | None = None, worker_id: str | None = None
    ):
        self.settings = settings
        self.store = store or JobStore(
            settings.database_path,
            lease_seconds=settings.lease_seconds,
            max_attempts=settings.max_attempts,
        )
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:6]}"
        self._releases: dict[str, tuple[Release, Predictor]] = {}
        self._last_cleanup = 0.0

    def resolve(self, version: str) -> tuple[Release, Predictor]:
        if version not in self._releases:
            release = load_release(self.settings.releases_dir / version)
            self._releases[version] = (release, release.load_predictor(self.settings.device))
            log.info("loaded model release %s", version)
        return self._releases[version]

    def run_once(self) -> bool:
        """Process at most one audit. Returns True if an audit was claimed."""
        loaded = next(iter(self._releases), None)
        self.store.heartbeat(self.worker_id, loaded)
        self._maybe_cleanup()
        claim = self.store.claim_next(self.worker_id)
        if claim is None:
            return False
        try:
            self._process(claim)
        except Exception:
            log.exception("transient failure on %s; releasing for retry", claim.audit_id)
            self.store.release_lease(claim)
        return True

    def _process(self, claim: Claim) -> None:
        audit = self.store.get_audit(claim.audit_id)
        if audit is None:
            return
        try:
            release, predictor = self.resolve(audit["model_version"])
        except ReleaseError as exc:
            self.store.fail_audit(claim.audit_id, f"model_release_unavailable: {exc}")
            log.error("audit %s: pinned release unavailable: %s", claim.audit_id, exc)
            return
        policy = release.manifest.review_policy
        if policy.version != audit["review_policy_version"]:
            self.store.fail_audit(claim.audit_id, "review_policy_mismatch")
            return

        deadline = time.monotonic() + self.settings.job_deadline_seconds
        for image in self.store.list_images(claim.audit_id):
            if image["status"] != "pending":
                continue
            if time.monotonic() > deadline:
                self.store.fail_image(claim, image["id"], "job_deadline_exceeded")
                continue
            if not self.store.extend_lease(claim):
                log.warning("audit %s: lease lost; stopping", claim.audit_id)
                return
            self._process_image(claim, audit, image, release, predictor)
        status = self.store.finish_audit(claim)
        log.info("audit %s finished: %s", claim.audit_id, status)

    def _process_image(
        self,
        claim: Claim,
        audit: dict[str, object],
        image: dict[str, object],
        release: Release,
        predictor: Predictor,
    ) -> None:
        image_id = str(image["id"])
        region = audit["inspection_region"]
        assert isinstance(region, list)
        key = cache_key(
            str(image["sha256"]), region, release.version, release.manifest.review_policy.version
        )
        artifact_dir = Path("artifacts") / claim.audit_id
        paths = {kind: artifact_dir / f"{image_id}_{kind}.png" for kind in ARTIFACT_KINDS}

        cached = self.store.find_cached_result(key)
        if cached is not None and cached["result"] is not None:
            for kind, rel in paths.items():
                source = (
                    self.settings.data_dir
                    / "artifacts"
                    / str(cached["audit_id"])
                    / f"{cached['id']}_{kind}.png"
                )
                target = self.settings.data_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            result = {**cached["result"], "cache_hit": True}
        else:
            try:
                rgb = load_rgb(self.settings.data_dir / str(image["upload_path"]))
                outcome = process_image(
                    rgb,
                    image_sha256=str(image["sha256"]),
                    region_polygon=region,
                    predictor=predictor,
                    policy=release.manifest.review_policy,
                    model_version=release.version,
                )
            except (OSError, ValueError) as exc:
                self.store.fail_image(claim, image_id, f"invalid_image: {exc}")
                return
            _write_png(outcome.overlay, self.settings.data_dir / paths["overlay"])
            _write_png(outcome.mask, self.settings.data_dir / paths["mask"])
            _write_png(outcome.uncertainty, self.settings.data_dir / paths["uncertainty"])
            result = {**outcome.result, "cache_hit": False}

        artifacts = [(kind, str(paths[kind]), ARTIFACT_KINDS[kind]) for kind in ARTIFACT_KINDS]
        if not self.store.complete_image(claim, image_id, result, key, artifacts):
            log.info("image %s already complete or lease lost; result not recorded twice", image_id)

    def _maybe_cleanup(self) -> None:
        now = time.monotonic()
        if now - self._last_cleanup < 300:
            return
        self._last_cleanup = now
        for audit_id in self.store.expired_audits():
            for rel in self.store.delete_audit(audit_id):
                (self.settings.data_dir / rel).unlink(missing_ok=True)
            for sub in ("uploads", "artifacts"):
                shutil.rmtree(self.settings.data_dir / sub / audit_id, ignore_errors=True)
            log.info("expired audit %s deleted", audit_id)

    def run_forever(self, poll_seconds: float = 1.0) -> None:
        log.info("worker %s started", self.worker_id)
        while True:
            if not self.run_once():
                time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the BeltWatch inference worker.")
    parser.add_argument("--poll-seconds", type=float, default=1.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    settings = Settings.from_env()
    if settings.torch_threads:
        torch.set_num_threads(settings.torch_threads)
    worker = Worker(settings)
    try:
        worker.resolve(active_version(settings.releases_dir))
    except (ReleaseError, OSError) as exc:
        log.warning("active release not preloaded: %s", exc)
    worker.run_forever(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
