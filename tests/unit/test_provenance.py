import hashlib
import re
from pathlib import Path

import pytest

from beltwatch.provenance import file_digest, git_commit, provenance


def test_file_digest_is_the_sha256_of_the_bytes(tmp_path: Path) -> None:
    data = bytes(range(256)) * 9000  # spans several read chunks
    path = tmp_path / "weights.bin"
    path.write_bytes(data)

    assert file_digest(path) == hashlib.sha256(data).hexdigest()
    assert provenance(path)["model_sha256"] == hashlib.sha256(data).hexdigest()
    assert provenance(path)["model_file"] == str(path)


def test_git_commit_outside_a_repository_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.delenv("BELTWATCH_GIT_COMMIT", raising=False)

    assert git_commit() == "unknown"
    assert provenance(None) == {"git_commit": "unknown", "model_file": None, "model_sha256": None}


def test_git_commit_outside_a_repository_falls_back_to_the_build_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    monkeypatch.setenv("BELTWATCH_GIT_COMMIT", "a" * 40)

    assert git_commit() == "a" * 40


def test_git_commit_in_this_repository_is_a_full_hash() -> None:
    assert re.fullmatch(r"[0-9a-f]{40}(-dirty)?|unknown", git_commit())
