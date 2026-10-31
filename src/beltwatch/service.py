"""Start the API and the inference worker as child processes, and stop them together."""

import os
import signal
import subprocess
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO


def stop_process(process: subprocess.Popen[bytes]) -> None:
    """Stop a process and everything it started (it leads its own process group)."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


@contextmanager
def running_service(
    host: str, port: int, logs: IO[bytes] | None, env: dict[str, str] | None = None
) -> Iterator[list[subprocess.Popen[bytes]]]:
    """Yield ``[api, worker]``; ``logs=None`` lets them write to this process's output."""
    api = ("uvicorn", "beltwatch.api.app:create_app", "--factory", "--log-level", "warning")
    commands = (
        [sys.executable, "-m", *api, "--host", host, "--port", str(port)],
        [sys.executable, "-m", "beltwatch.jobs.worker"],
    )
    stderr = subprocess.STDOUT if logs is not None else None
    processes: list[subprocess.Popen[bytes]] = []
    try:
        for command in commands:
            processes.append(
                subprocess.Popen(
                    command,
                    env=dict(os.environ if env is None else env),
                    stdout=logs,
                    stderr=stderr,
                    start_new_session=True,
                )
            )
        yield processes
    finally:
        for process in processes:
            stop_process(process)
