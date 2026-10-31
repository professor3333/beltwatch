"""Run the hosted demo in one container: fetch the release, then the API and the worker.

Usage::

    python -m beltwatch.demo [--host 0.0.0.0] [--port 7860]

The bundle comes from ``BELTWATCH_DEMO_RELEASE_REPO``, ``_REVISION``,
``_VERSION`` and ``_SHA256`` (see :mod:`beltwatch.release.fetch`); a private
repository needs ``HF_TOKEN``, which is not passed on to the API or the
worker. The service runs with ``BELTWATCH_PUBLIC_DEMO=true`` (no training
consent, demo notice in the UI) and, unless one is given, a random
``BELTWATCH_METRICS_TOKEN``, so ``/metrics`` is not public. The process exits
non-zero as soon as the API or the worker stops, so the host restarts the
container. See docs/operations.md ("Hosted demo").
"""

import argparse
import logging
import os
import secrets
import signal
import subprocess
import threading
from collections.abc import Sequence
from pathlib import Path

from beltwatch.api.settings import Settings
from beltwatch.release.bundle import ReleaseError
from beltwatch.release.fetch import fetch_release
from beltwatch.service import running_service

log = logging.getLogger(__name__)

RELEASE_VARIABLES = ("REPO", "REVISION", "VERSION", "SHA256")


def demo_env(environ: dict[str, str]) -> dict[str, str]:
    """The environment for the API and the worker."""
    env = {k: v for k, v in environ.items() if k != "HF_TOKEN"}
    env["BELTWATCH_PUBLIC_DEMO"] = "true"
    if not env.get("BELTWATCH_METRICS_TOKEN"):
        env["BELTWATCH_METRICS_TOKEN"] = secrets.token_urlsafe(32)
    return env


def fetch_configured_release(environ: dict[str, str], releases_dir: Path) -> str:
    values = {name: environ.get(f"BELTWATCH_DEMO_RELEASE_{name}", "") for name in RELEASE_VARIABLES}
    missing = [f"BELTWATCH_DEMO_RELEASE_{name}" for name, value in values.items() if not value]
    if missing:
        raise ReleaseError(f"set {', '.join(missing)}")
    release = fetch_release(
        releases_dir,
        repo_id=values["REPO"],
        revision=values["REVISION"],
        version=values["VERSION"],
        sha256=values["SHA256"],
        token=environ.get("HF_TOKEN") or None,
    )
    return release.version


def supervise(
    processes: Sequence[subprocess.Popen[bytes]], stopped: threading.Event, poll: float = 1.0
) -> int:
    """Wait until ``stopped`` is set (0) or one of the processes exits (1)."""
    while not stopped.wait(poll):
        for process in processes:
            if process.poll() is not None:
                log.error("%s exited with %s", process.args, process.returncode)
                return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="0.0.0.0")  # the container's public port
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per hub request otherwise
    environ = dict(os.environ)
    try:
        version = fetch_configured_release(environ, Settings.from_env().releases_dir)
    except ReleaseError as exc:
        log.error("cannot start the demo: %s", exc)
        return 1
    log.info("serving %s on %s:%d", version, args.host, args.port)
    stopped = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    signal.signal(signal.SIGINT, lambda *_: stopped.set())
    with running_service(args.host, args.port, None, demo_env(environ)) as processes:
        code = supervise(processes, stopped)
    log.info("demo stopped")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
