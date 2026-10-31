"""Fetch a release bundle from a Hugging Face model repository and activate it.

Usage::

    uv run python -m beltwatch.release.fetch --repo <owner>/<repo> --revision <commit> \
        --version beltwatch-0.1.0 --sha256 <weights digest> [--releases-dir model_releases]

The repository holds bundles as top-level directories named by version (the
same layout as ``model_releases/``). The revision must be a full commit hash
so that a moved branch cannot change what is served, and the weights digest
must match the one recorded for the release (docs/results/<version>/). A
private repository needs ``HF_TOKEN`` (a read-only token is enough). Used by
the hosted demo at startup (:mod:`beltwatch.demo`).
"""

import argparse
import logging
import os
import re
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path

from beltwatch.release.bundle import (
    Release,
    ReleaseError,
    activate,
    active_version,
    load_release,
)

log = logging.getLogger(__name__)

COMMIT = re.compile(r"^[0-9a-f]{40}$")

Download = Callable[..., str]


def _hub_download(
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    allow_patterns: list[str],
    local_dir: str,
    token: str | None,
) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(
        repo_id,
        repo_type=repo_type,
        revision=revision,
        allow_patterns=allow_patterns,
        local_dir=local_dir,
        token=token,
    )


def _check(release: Release, version: str, sha256: str) -> Release:
    if release.version != version:
        raise ReleaseError(f"bundle directory {version} holds release {release.version}")
    if release.manifest.model.sha256 != sha256:
        raise ReleaseError(
            f"{version}: weights digest {release.manifest.model.sha256}, expected {sha256}"
        )
    return release


def fetch_release(
    releases_dir: Path,
    *,
    repo_id: str,
    revision: str,
    version: str,
    sha256: str,
    token: str | None = None,
    download: Download = _hub_download,
) -> Release:
    """Download ``version`` (unless present), verify it, and make it the active release."""
    if not COMMIT.fullmatch(revision):
        raise ReleaseError(f"revision must be a full commit hash, not {revision!r}")
    target = releases_dir / version
    if not target.exists():
        staging = releases_dir / f".fetch-{version}"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        try:
            try:
                download(
                    repo_id=repo_id,
                    repo_type="model",
                    revision=revision,
                    allow_patterns=[f"{version}/*"],
                    local_dir=str(staging),
                    token=token,
                )
            except Exception as exc:  # network, authentication, or a missing revision
                raise ReleaseError(f"download from {repo_id}@{revision} failed: {exc}") from exc
            fetched = staging / version
            if not fetched.is_dir():
                raise ReleaseError(f"{repo_id}@{revision} has no {version}/ directory")
            # Check before moving, so a bad download never sits where a release belongs.
            _check(load_release(fetched), version, sha256)
            fetched.rename(target)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
        log.info("fetched %s from %s@%s", version, repo_id, revision)
    release = _check(load_release(target), version, sha256)
    try:
        current = active_version(releases_dir)
    except ReleaseError:
        current = None
    if current != version:
        activate(releases_dir, version, f"fetched from {repo_id}@{revision}")
    return release


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True, help="full commit hash")
    parser.add_argument("--version", required=True)
    parser.add_argument("--sha256", required=True, help="expected weights digest")
    parser.add_argument("--releases-dir", type=Path, default=Path("model_releases"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        release = fetch_release(
            args.releases_dir,
            repo_id=args.repo,
            revision=args.revision,
            version=args.version,
            sha256=args.sha256,
            token=os.environ.get("HF_TOKEN"),
        )
    except ReleaseError as exc:
        log.error("%s", exc)
        return 1
    log.info("active release: %s", release.version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
