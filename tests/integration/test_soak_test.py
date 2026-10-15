import importlib.util
import os
import shlex
import sys
import time
from pathlib import Path

import pytest

from beltwatch.api.settings import Settings
from tests.integration.test_deploy_smoke import live_service  # noqa: F401  (fixture)

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))  # the script imports deploy_smoke from its own directory
spec = importlib.util.spec_from_file_location("soak_test", SCRIPTS / "soak_test.py")
assert spec and spec.loader
soak = importlib.util.module_from_spec(spec)
spec.loader.exec_module(soak)


def test_soak_passes_against_a_live_service(
    live_service: str,  # noqa: F811
    settings: Settings,
    tmp_path: Path,
) -> None:
    out = tmp_path / "soak.json"
    code = soak.main(
        [
            "--base-url",
            live_service,
            "--audits",
            "12",
            "--in-flight",
            "4",
            "--replay-every",
            "3",
            "--database",
            str(settings.database_path),
            "--timeout",
            "120",
            "--out",
            str(out),
        ]
    )
    summary = soak.json.loads(out.read_text())
    assert code == 0, summary
    assert summary["by_status"] == {"complete": 12}
    assert summary["idempotent_replays"] == {"sent": 4, "same_audit": 4}
    assert summary["database_checks"] == dict.fromkeys(summary["database_checks"], 0)
    assert summary["audits_claimed_more_than_once"] == 0


def test_regions_are_distinct_and_normalized() -> None:
    rng = soak.random.Random(0)
    regions = [soak.region(rng) for _ in range(200)]
    assert len({str(r) for r in regions}) == 200  # no cache hits between audits
    assert all(0 <= v <= 1 for r in regions for vertex in r for v in vertex)


def test_worker_is_killed_with_its_children_and_restarted(tmp_path: Path) -> None:
    # Like `uv run`, the command starts the real worker as a child; it records its pid.
    pid_file = tmp_path / "child.pid"
    grandchild = (
        f"import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"
    )
    wrapper = f"import subprocess, sys; subprocess.run([sys.executable, '-c', {grandchild!r}])"
    worker = soak.WorkerProcess(f"{sys.executable} -c {shlex.quote(wrapper)}", 0.5)
    worker.start()
    first = worker.process
    assert first is not None
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text():
            break
        time.sleep(0.05)
    child_pid = int(pid_file.read_text())
    time.sleep(0.5)
    worker.tick()
    assert worker.kills == 1 and first.returncode == -9
    time.sleep(0.2)  # let the orphaned child be reaped
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)  # the real worker did not survive its wrapper
    assert worker.process is not first and worker.process.poll() is None
    worker.stop()
    with pytest.raises(ProcessLookupError):
        os.killpg(worker.process.pid, 0)  # stop leaves nothing running
