"""Run a usability-study session from one command.

Usage::

    uv run python -m beltwatch.study.session data/study/<name> [--port 8000]

Starts the API and the inference worker with a long upload retention
(``BELTWATCH_RETENTION_HOURS``, default 720 h, since sessions can span days),
waits until the service is ready, then submits the study (or, after the first
time, replays and checks the recorded audits, see :mod:`beltwatch.study.submit`).
It prints the page to open and the plan files, and stops both processes on
Ctrl-C. Service logs go to ``<study>/service.log``. See docs/usability_study.md.
"""

import argparse
import json
import logging
import os
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from beltwatch.service import running_service
from beltwatch.study.prepare import StudyError, load_study
from beltwatch.study.submit import PILOT, http_transport, submit_study

log = logging.getLogger(__name__)

RETENTION_HOURS = "720"
LOG_FILE = "service.log"


def _wait_ready(
    base_url: str, processes: Sequence[subprocess.Popen[bytes]], timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    last = "no response"
    while time.monotonic() < deadline:
        for process in processes:
            if process.poll() is not None:
                raise StudyError(f"{process.args!r} exited with {process.returncode}")
        try:
            with urllib.request.urlopen(base_url + "/health/ready", timeout=5) as resp:
                if resp.status == 200:
                    return
        except urllib.error.HTTPError as exc:
            last = exc.read()[:300].decode(errors="replace")
        except OSError as exc:
            last = str(exc)
        time.sleep(0.5)
    raise StudyError(f"service not ready after {timeout:.0f} s: {last}")


def _check_port_free(host: str, port: int) -> None:
    with socket.socket() as probe:
        # Like uvicorn: a socket left in TIME_WAIT by the last session does not count.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as exc:
            raise StudyError(f"{host}:{port} is in use ({exc}); choose another --port") from exc


def _wait_for_interrupt() -> None:
    stopped = threading.Event()
    previous = signal.signal(signal.SIGTERM, lambda *_: stopped.set())
    try:
        while not stopped.wait(1.0):
            pass
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGTERM, previous)


def run_session(
    study_dir: Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    participants: int = 8,
    ready_timeout: float = 120.0,
    env: dict[str, str] | None = None,
    wait: Callable[[str, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Start the service, submit or check the study, then ``wait`` (default: until Ctrl-C)."""
    load_study(study_dir)  # fail before starting anything if the folder is not a study
    _check_port_free(host, port)
    base_url = f"http://{host}:{port}"
    env = dict(os.environ if env is None else env)
    env.setdefault("BELTWATCH_RETENTION_HOURS", RETENTION_HOURS)
    with (study_dir / LOG_FILE).open("ab") as logs, running_service(host, port, logs, env) as procs:
        _wait_ready(base_url, procs, ready_timeout)
        record = submit_study(study_dir, http_transport(base_url), participants=participants)
        for set_name, entry in record["audits"].items():
            log.info(
                "%s: audit %s on %s, kept until %s",
                set_name,
                entry["audit_id"],
                entry["model_version"],
                entry["expires_at"],
            )
        plans = study_dir / "plans"
        log.info("\nOpen %s/study.html and load a plan from %s:", base_url, plans)
        log.info("  developer pilot: %s", plans / f"{PILOT}.json")
        log.info(
            "  participants:    %s ... %s", plans / "P01.json", plans / f"P{participants:02d}.json"
        )
        log.info("Do not open %s during a session; it holds the answers.", study_dir / "study.json")
        log.info("Press Ctrl-C to stop the service when the session is over.")
        (wait or (lambda _url, _record: _wait_for_interrupt()))(base_url, record)
    log.info("service stopped")
    return record


def _existing_participants(study_dir: Path) -> int | None:
    plans = sorted((study_dir / "plans").glob("P[0-9]*.json"))
    return len(plans) or None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--participants",
        type=int,
        help="plans to write (default: as many as already exist, else 8)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    participants = args.participants or _existing_participants(args.study_dir) or 8
    try:
        run_session(args.study_dir, host=args.host, port=args.port, participants=participants)
    except (StudyError, json.JSONDecodeError) as exc:
        log.error("%s (service log: %s)", exc, args.study_dir / LOG_FILE)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
