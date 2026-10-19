"""Lineage fields shared by training runs, evaluation outputs and release bundles.

Every number BeltWatch reports must trace back to the code and the weights
that produced it, so evaluation, calibration, error-analysis and benchmark
outputs all record the git commit and the SHA-256 of the model file.
"""

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any


def git_commit() -> str:
    """HEAD of the working directory's repository, with ``-dirty`` for tracked changes.

    Outside a repository (the serving image has no ``.git``), the commit the
    image was built from, from ``BELTWATCH_GIT_COMMIT``; otherwise ``unknown``.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.CalledProcessError):
        return os.environ.get("BELTWATCH_GIT_COMMIT") or "unknown"


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(model_path: Path | None) -> dict[str, Any]:
    """The code commit and, for a trained model, its file and SHA-256."""
    return {
        "git_commit": git_commit(),
        "model_file": None if model_path is None else str(model_path),
        "model_sha256": None if model_path is None else file_digest(model_path),
    }
