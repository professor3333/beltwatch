import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_record.py"
spec = importlib.util.spec_from_file_location("benchmark_record", SCRIPT)
assert spec and spec.loader
br = importlib.util.module_from_spec(spec)
spec.loader.exec_module(br)

BENCHMARK = {
    "release": "beltwatch-0.1.0",
    "model_sha256": "7be82fde",
    "meets_targets": {"p95_seconds": True, "images_per_second": True, "peak_rss_mb": True},
}
SOAK = {
    "audits": 4,
    "in_flight": 1,
    "by_status": {"complete": 4},
    "success_rate": 1.0,
    "seconds_to_finish": {"p50": 1.5, "p95": 2.0},
    "audits_per_second": 0.512345,
    "wall_seconds": 7.8,
    "problems": [],
}
INSPECT = [
    {
        "Name": "/beltwatch-worker-1",
        "Image": "sha256:abc",
        "Config": {"Labels": {"com.docker.compose.service": "worker"}},
        "HostConfig": {"Memory": 3 * 2**30, "NanoCpus": 4_000_000_000},
    },
    {"Name": "/bench", "Image": "sha256:abc", "Config": {}, "HostConfig": {"Memory": 0}},
]


def test_record_combines_the_run_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_RUN_ID", "42")
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/BeltWatch")
    monkeypatch.setenv("GITHUB_SHA", "f" * 40)
    files = {"benchmark.json": BENCHMARK, "seq.json": SOAK, "inspect.json": INSPECT}
    for name, content in files.items():
        (tmp_path / name).write_text(json.dumps(content))
    frames = {"split": "val", "frames": [{"member": "a.PNG", "bytes": 3, "sha256": "aa"}]}
    (tmp_path / "frames.json").write_text(json.dumps(frames))
    peaks = tmp_path / "peaks"
    peaks.mkdir()
    (peaks / "worker").write_text(f"{512 * 2**20}\n")
    (peaks / "api").write_text("unavailable\n")
    out = tmp_path / "record.json"

    code = br.main(
        [
            "--benchmark", str(tmp_path / "benchmark.json"),
            "--end-to-end", f"sequential={tmp_path / 'seq.json'}",
            "--containers", str(tmp_path / "inspect.json"),
            "--memory-peaks", str(peaks),
            "--frames", str(tmp_path / "frames.json"),
            "--out", str(out),
        ]
    )  # fmt: skip

    assert code == 0
    result = json.loads(out.read_text())
    assert result["release"] == "beltwatch-0.1.0"
    assert result["containers"]["worker"] == {
        "image": "sha256:abc",
        "memory_limit_mb": 3072,
        "cpus": 4.0,
    }
    assert result["containers"]["bench"]["memory_limit_mb"] is None
    assert result["memory_peak_mb"] == {"api": None, "worker": 512.0}
    assert result["end_to_end"]["sequential"]["audits_per_second"] == 0.5123
    assert result["end_to_end"]["sequential"]["problems"] == 0
    assert result["frames"] == {"split": "val", "count": 1, "sha256": ["aa"]}
    assert result["host"]["run_url"] == "https://github.com/owner/BeltWatch/actions/runs/42"
    assert result["host"]["git_commit"] == "f" * 40
    assert result["meets_targets"] == BENCHMARK["meets_targets"]


def test_host_reads_cpu_model_and_memory_from_proc(tmp_path: Path) -> None:
    (tmp_path / "cpuinfo").write_text("processor\t: 0\nmodel name\t: AMD EPYC 7763 64-Core\n")
    (tmp_path / "meminfo").write_text("MemTotal:       16384000 kB\nMemFree: 1 kB\n")

    info = br.host(tmp_path)

    assert info["cpu_model"] == "AMD EPYC 7763 64-Core"
    assert info["memory_total_mb"] == 16000.0
