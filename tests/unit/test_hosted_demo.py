import importlib.util
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "hosted_demo.py"
spec = importlib.util.spec_from_file_location("hosted_demo", SCRIPT)
assert spec and spec.loader
hd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hd)

TEMPLATE = (REPO / "docker" / "space" / "Dockerfile.template").read_text()
CARD = (REPO / "docker" / "space" / "README.md").read_text()
VALUES = {
    "git_commit": "c" * 40,
    "release_repo": "owner/beltwatch-releases",
    "release_revision": "r" * 40,
    "release_version": "beltwatch-0.1.0",
    "release_sha256": "s" * 64,
}


def build_lines(dockerfile: str) -> list[str]:
    return [
        line
        for line in dockerfile.splitlines()
        if line.startswith(("FROM ", "COPY --from=", "RUN uv sync"))
    ]


def test_space_image_matches_the_serving_image() -> None:
    assert build_lines(TEMPLATE) == build_lines((REPO / "Dockerfile").read_text())


def test_space_files_fill_every_placeholder() -> None:
    dockerfile, card = hd.render(TEMPLATE, VALUES), hd.render(CARD, VALUES)
    assert f"BELTWATCH_DEMO_RELEASE_REVISION={'r' * 40}" in dockerfile
    assert "beltwatch-0.1.0" in card and "s" * 64 in card
    with pytest.raises(SystemExit, match="release_sha256"):
        hd.render(TEMPLATE, {k: v for k, v in VALUES.items() if k != "release_sha256"})


def test_space_card_matches_the_container() -> None:
    front = CARD.split("---")[1]
    port = re.search(r"^app_port: (\d+)$", front, re.M)
    assert port and f"EXPOSE {port.group(1)}" in TEMPLATE
    description = re.search(r"^short_description: (.+)$", front, re.M)
    assert description and len(description.group(1)) <= 60
    assert "visible coverage" in CARD and "not contamination by" in CARD.replace("\n", " ")


def test_declared_weights_digest_is_read_from_the_declaration() -> None:
    assert hd.declared_sha256("beltwatch-0.1.0").startswith("7be82fde")
