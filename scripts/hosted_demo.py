"""Publish the hosted demo on a Hugging Face Docker Space.

Two steps, run from a clean checkout with a Hugging Face login
(``HF_TOKEN`` or a saved token with write access)::

    # 1. Upload a release bundle to a private model repository (once per release).
    uv run python scripts/hosted_demo.py upload-release \
        --repo <owner>/beltwatch-releases --version beltwatch-0.1.0

    # 2. Assemble the Space from the current commit and push it.
    uv run python scripts/hosted_demo.py build-space --out <dir> \
        --release-repo <owner>/beltwatch-releases --release-revision <commit> \
        --version beltwatch-0.1.0 [--push <owner>/beltwatch]

``upload-release`` refuses a bundle whose weights digest differs from the
release declaration (configs/releases/<version>.yaml) and prints the model
repository's commit, which ``build-space`` pins. ``build-space`` takes only
committed files (``git archive HEAD``), so untracked data never reaches the
Space. The Space then needs the secret ``HF_TOKEN``: a read-only token for the
release repository. See docs/operations.md ("Hosted demo").
"""

import argparse
import io
import re
import subprocess
import sys
import tarfile
from pathlib import Path

from beltwatch.release.bundle import load_release
from beltwatch.release.declaration import load_declaration
from beltwatch.release.fetch import COMMIT

REPO = Path(__file__).resolve().parents[1]
SPACE_DIR = REPO / "docker" / "space"
# What the Space image needs; README.md is replaced by the Space card.
SPACE_PATHS = ("src", "frontend", "configs", "pyproject.toml", "uv.lock", "LICENSE")
PLACEHOLDER = re.compile(r"\{[a-z0-9_]+\}")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def declared_sha256(version: str) -> str:
    return load_declaration(REPO / "configs" / "releases" / f"{version}.yaml").model.sha256


def render(template: str, values: dict[str, str]) -> str:
    text = template
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    if left := PLACEHOLDER.search(text):
        raise SystemExit(f"unfilled placeholder {left.group()}")
    return text


def upload_release(repo_id: str, version: str, releases_dir: Path) -> str:
    from huggingface_hub import HfApi

    release = load_release(releases_dir / version)
    expected = declared_sha256(version)
    if release.manifest.model.sha256 != expected:
        raise SystemExit(
            f"{version}: bundle weights {release.manifest.model.sha256} != declared {expected}"
        )
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    if not api.model_info(repo_id).private:
        raise SystemExit(f"{repo_id} is public; the weights are not distributed")
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(release.directory),
        path_in_repo=version,
        commit_message=f"Add release bundle {version}",
    )
    return str(commit.oid)


def build_space(out: Path, *, release_repo: str, release_revision: str, version: str) -> str:
    if not COMMIT.fullmatch(release_revision):
        raise SystemExit(f"--release-revision must be a full commit hash, not {release_revision!r}")
    if _git("status", "--porcelain"):
        raise SystemExit("the working tree has changes; commit them first")
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty")
    commit = _git("rev-parse", "HEAD")
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD", "--", *SPACE_PATHS],
        cwd=REPO,
        check=True,
        capture_output=True,
    ).stdout
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(out, filter="data")
    values = {
        "git_commit": commit,
        "release_repo": release_repo,
        "release_revision": release_revision,
        "release_version": version,
        "release_sha256": declared_sha256(version),
    }
    for name, target in (("Dockerfile.template", "Dockerfile"), ("README.md", "README.md")):
        (out / target).write_text(render((SPACE_DIR / name).read_text(), values))
    return commit


def push_space(out: Path, space_id: str, commit: str, version: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)
    api.upload_folder(
        repo_id=space_id,
        repo_type="space",
        folder_path=str(out),
        commit_message=f"Deploy {commit[:7]} with {version}",
        delete_patterns="*",  # the Space holds exactly this build
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    up = commands.add_parser("upload-release")
    up.add_argument("--repo", required=True)
    up.add_argument("--version", required=True)
    up.add_argument("--releases-dir", type=Path, default=REPO / "model_releases")
    build = commands.add_parser("build-space")
    build.add_argument("--out", type=Path, required=True)
    build.add_argument("--release-repo", required=True)
    build.add_argument("--release-revision", required=True)
    build.add_argument("--version", required=True)
    build.add_argument("--push", metavar="SPACE_ID")
    args = parser.parse_args(argv)
    if args.command == "upload-release":
        print(upload_release(args.repo, args.version, args.releases_dir))
        return 0
    commit = build_space(
        args.out,
        release_repo=args.release_repo,
        release_revision=args.release_revision,
        version=args.version,
    )
    print(f"assembled {args.out} from {commit}", file=sys.stderr)
    if args.push:
        push_space(args.out, args.push, commit, args.version)
        print(f"pushed to https://huggingface.co/spaces/{args.push}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
