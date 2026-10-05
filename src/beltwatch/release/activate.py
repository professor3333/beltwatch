"""Activate (or roll back to) an existing release.

Usage::

    uv run python -m beltwatch.release.activate beltwatch-0.1.0 --reason "promoted: gates passed"

New audits are pinned to the activated version immediately. Existing audits
keep the version they were created with.
"""

import argparse
from pathlib import Path

from beltwatch.release.bundle import activate, active_version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Activate a model release (also used for rollback)."
    )
    parser.add_argument("version")
    parser.add_argument("--reason", required=True, help="why: promotion, rollback, ...")
    parser.add_argument("--releases-dir", type=Path, default=Path("model_releases"))
    args = parser.parse_args(argv)
    activate(args.releases_dir, args.version, args.reason)
    print(f"active release: {active_version(args.releases_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
