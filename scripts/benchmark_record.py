"""Assemble the host benchmark record (standard library only).

Collects the outputs of a host benchmark run (.github/workflows/benchmark.yml)
into one JSON file: the in-container benchmark of the release bundle, the
end-to-end runs through the HTTPS proxy, each container's resource limits and
cgroup memory peak, the frames used, and the host itself (CPU model, cores,
memory, runner image, commit and run URL).

Usage::

    python scripts/benchmark_record.py --benchmark benchmark.json \\
        --end-to-end sequential=sequential.json --end-to-end saturated=saturated.json \\
        --containers inspect.json --memory-peaks peaks/ --frames frames/frames.json \\
        --out benchmark-host.json

The engineering targets (p95, throughput, worker memory) are judged on the
in-container benchmark, as on Kaggle; the end-to-end times also include the
upload, the queue and status polling, and are reported alongside. A run with
audits served from the result cache is refused: its times would not include
inference.
"""

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def host(proc: Path = Path("/proc")) -> dict[str, Any]:
    """The machine the run used, from /proc where available."""
    cpu_model = None
    if (proc / "cpuinfo").exists():
        for line in (proc / "cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    memory_mb = None
    if (proc / "meminfo").exists():
        for line in (proc / "meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory_mb = round(int(line.split()[1]) / 1024, 1)
                break
    env = os.environ
    run_url = None
    if env.get("GITHUB_RUN_ID"):
        run_url = (
            f"{env.get('GITHUB_SERVER_URL', 'https://github.com')}/"
            f"{env.get('GITHUB_REPOSITORY')}/actions/runs/{env['GITHUB_RUN_ID']}"
        )
    return {
        "platform": platform.platform(),
        "cpu_model": cpu_model,
        "cpu_count": os.cpu_count(),
        "memory_total_mb": memory_mb,
        "runner": {
            "name": env.get("RUNNER_NAME"),
            "image_os": env.get("ImageOS"),
            "image_version": env.get("ImageVersion"),
        },
        "git_commit": env.get("GITHUB_SHA"),
        "run_url": run_url,
    }


def containers(inspect: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Image and resource limits of each container, from ``docker inspect`` output."""
    out = {}
    for c in inspect:
        labels = c.get("Config", {}).get("Labels") or {}
        name = labels.get("com.docker.compose.service") or c["Name"].lstrip("/")
        limits = c.get("HostConfig", {})
        out[name] = {
            "image": c.get("Image"),
            "memory_limit_mb": round(limits["Memory"] / 2**20) if limits.get("Memory") else None,
            "cpus": limits["NanoCpus"] / 1e9 if limits.get("NanoCpus") else None,
        }
    return out


def memory_peaks(directory: Path) -> dict[str, float | None]:
    """cgroup ``memory.peak`` per service in MiB (includes page cache, so above RSS)."""
    peaks: dict[str, float | None] = {}
    for path in sorted(directory.iterdir()):
        text = path.read_text().strip()
        peaks[path.name] = round(int(text) / 2**20, 1) if text.isdigit() else None
    return peaks


def end_to_end(summary: dict[str, Any]) -> dict[str, Any]:
    keys = ("audits", "in_flight", "by_status", "success_rate", "seconds_to_finish", "cache_hits")
    return {
        **{k: summary[k] for k in keys},
        "audits_per_second": round(summary["audits_per_second"], 4),
        "wall_seconds": summary["wall_seconds"],
        "problems": len(summary["problems"]),
    }


def record(
    benchmark: dict[str, Any],
    runs: dict[str, dict[str, Any]],
    inspect: list[dict[str, Any]],
    peaks: dict[str, float | None],
    frames: dict[str, Any],
    host_info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "release": benchmark["release"],
        "model_sha256": benchmark["model_sha256"],
        "host": host_info,
        "containers": containers(inspect),
        "benchmark": benchmark,
        "end_to_end": {name: end_to_end(summary) for name, summary in runs.items()},
        "memory_peak_mb": peaks,
        "frames": {
            "split": frames["split"],
            "count": len(frames["frames"]),
            "sha256": [f["sha256"] for f in frames["frames"]],
        },
        "meets_targets": benchmark["meets_targets"],
        "note": (
            "Targets judged on the in-container benchmark (warm, excluding upload and queue). "
            "End-to-end times run from submit to a finished audit through the HTTPS proxy, "
            "with status polled every 0.5 s. A GitHub-hosted runner is a shared cloud VM."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assemble the host benchmark record.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument(
        "--end-to-end", action="append", default=[], metavar="NAME=FILE", help="soak summary"
    )
    parser.add_argument("--containers", type=Path, required=True, help="docker inspect JSON")
    parser.add_argument("--memory-peaks", type=Path, required=True, help="one file per service")
    parser.add_argument("--frames", type=Path, required=True, help="frames.json of the sample")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    runs = {}
    for item in args.end_to_end:
        name, _, path = item.partition("=")
        runs[name] = read_json(Path(path))
        if runs[name]["cache_hits"]:
            parser.error(f"{name}: {runs[name]['cache_hits']} audits served from the result cache")
    result = record(
        read_json(args.benchmark),
        runs,
        read_json(args.containers),
        memory_peaks(args.memory_peaks),
        read_json(args.frames),
        host(),
    )
    text = json.dumps(result, indent=2)
    print(text)
    args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
