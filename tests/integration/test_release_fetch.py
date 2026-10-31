import shutil
from pathlib import Path
from typing import Any

import pytest

from beltwatch.release.bundle import ReleaseError, active_version, build_release
from beltwatch.release.fetch import fetch_release

from .conftest import POLICY

REVISION = "0123456789abcdef0123456789abcdef01234567"


@pytest.fixture
def hub(tmp_path: Path) -> tuple[Path, str]:
    """A local stand-in for the model repository: one bundle directory per version."""
    model = tmp_path / "best.pt"
    model.write_bytes(b"weights")
    release = build_release(
        tmp_path / "hub", version="v1", kind="unet", model_path=model, review_policy=POLICY
    )
    assert release.manifest.model.sha256 is not None
    return tmp_path / "hub", release.manifest.model.sha256


def fake_download(hub: Path, calls: list[dict[str, Any]]) -> Any:
    def download(**kwargs: Any) -> str:
        calls.append(kwargs)
        for pattern in kwargs["allow_patterns"]:
            version = pattern.removesuffix("/*")
            if (hub / version).is_dir():
                shutil.copytree(hub / version, Path(kwargs["local_dir"]) / version)
        return str(kwargs["local_dir"])

    return download


def test_fetch_verifies_activates_and_reuses(hub: tuple[Path, str], tmp_path: Path) -> None:
    source, sha = hub
    releases, calls = tmp_path / "releases", []
    args = {"repo_id": "owner/releases", "revision": REVISION, "version": "v1", "sha256": sha}

    release = fetch_release(releases, **args, token="t", download=fake_download(source, calls))
    assert release.version == "v1" and active_version(releases) == "v1"
    assert calls[0]["revision"] == REVISION and calls[0]["token"] == "t"
    assert sorted(p.name for p in releases.iterdir()) == ["active.json", "v1"]  # no staging left

    fetch_release(releases, **args, download=fake_download(source, calls))
    assert len(calls) == 1  # already present: verified, not downloaded again


def test_fetch_refuses_unexpected_weights(hub: tuple[Path, str], tmp_path: Path) -> None:
    source, _ = hub
    releases = tmp_path / "releases"
    with pytest.raises(ReleaseError, match="weights digest"):
        fetch_release(
            releases,
            repo_id="owner/releases",
            revision=REVISION,
            version="v1",
            sha256="0" * 64,
            download=fake_download(source, []),
        )
    assert not (releases / "v1").exists() and not (releases / "active.json").exists()


@pytest.mark.parametrize(
    ("revision", "version", "match"),
    [("main", "v1", "full commit hash"), (REVISION, "v2", "no v2/ directory")],
)
def test_fetch_refuses_moving_revisions_and_missing_bundles(
    hub: tuple[Path, str], tmp_path: Path, revision: str, version: str, match: str
) -> None:
    source, sha = hub
    with pytest.raises(ReleaseError, match=match):
        fetch_release(
            tmp_path / "releases",
            repo_id="owner/releases",
            revision=revision,
            version=version,
            sha256=sha,
            download=fake_download(source, []),
        )


def test_download_errors_become_release_errors(tmp_path: Path) -> None:
    def failing(**_: Any) -> str:
        raise OSError("401 Unauthorized")

    with pytest.raises(ReleaseError, match="401 Unauthorized"):
        fetch_release(
            tmp_path,
            repo_id="owner/releases",
            revision=REVISION,
            version="v1",
            sha256="0" * 64,
            download=failing,
        )
    assert list(tmp_path.iterdir()) == []
