"""Submit the study images through the real audit API and write participant plans.

Usage::

    uv run python -m beltwatch.study.submit data/study/<name> --base-url http://localhost:8000 \\
        --participants 8

Each set (practice, A, B) becomes one audit, created with an idempotency key,
so running this again replays the same audits instead of creating new ones.
The assisted condition therefore shows live results of the active release.
Writes ``audits.json``, ``plans/P01.json`` ... and ``plans/PILOT.json`` (a
protocol check that the analysis never includes). The plans hold image IDs
and display order only, never ground truth. See docs/usability_study.md.
"""

import argparse
import json
import logging
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from beltwatch.study.prepare import SETS, StudyError, load_study

log = logging.getLogger(__name__)

PLAN_VERSION = 1
AUDITS_FILE = "audits.json"
# Counterbalanced arms: (set, condition) for block 1 then block 2.
ARMS: tuple[tuple[tuple[str, str], tuple[str, str]], ...] = (
    (("A", "unaided"), ("B", "assisted")),
    (("B", "unaided"), ("A", "assisted")),
    (("A", "assisted"), ("B", "unaided")),
    (("B", "assisted"), ("A", "unaided")),
)

Transport = Callable[[str, str, bytes | None, dict[str, str]], dict[str, Any]]


def http_transport(base_url: str, timeout: float = 60.0) -> Transport:
    def call(method: str, path: str, body: bytes | None, headers: dict[str, str]) -> Any:
        req = urllib.request.Request(base_url.rstrip("/") + path, body, headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            raise StudyError(f"{method} {path}: {exc.code} {exc.read()[:300]!r}") from exc

    return call


def _multipart(files: list[tuple[str, bytes]], options: dict[str, Any]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for name, data in files:
        kind = mimetypes.guess_type(name)[0] or "application/octet-stream"
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\n'
            f"Content-Type: {kind}\r\n\r\n".encode()
            + data
            + b"\r\n"
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="options"\r\n\r\n'
        f"{json.dumps(options)}\r\n--{boundary}--\r\n".encode()
    )
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _suggestion(result: dict[str, Any], threshold: float, present: float) -> dict[str, Any]:
    coverage = result.get("visible_coverage") or {}
    return {
        "follow_up": (result.get("total_target_coverage") or 0.0) > threshold,
        "materials": sorted(m for m, v in coverage.items() if v >= present),
    }


def submit_study(
    study_dir: Path,
    call: Transport,
    *,
    participants: int,
    wait_seconds: float = 600.0,
    poll_seconds: float = 2.0,
) -> dict[str, Any]:
    study = load_study(study_dir)
    rules = study["rules"]
    audits: dict[str, Any] = {}
    for set_name in ("practice", *SETS):
        images = [i for i in study["images"] if i["set"] == set_name]
        files = [(Path(i["file"]).name, (study_dir / i["file"]).read_bytes()) for i in images]
        body, content_type = _multipart(
            files,
            {"camera_id": "usability-study", "inspection_region": study["inspection_region"]},
        )
        created = call(
            "POST",
            "/v1/audits",
            body,
            {
                "Content-Type": content_type,
                "Idempotency-Key": f"{study['name']}-{study['git_commit'][:12]}-{set_name}",
            },
        )
        audits[set_name] = {"audit_id": created["audit_id"], "model_version": None, "images": {}}
    deadline = time.monotonic() + wait_seconds
    for entry in audits.values():
        while True:
            audit = call("GET", f"/v1/audits/{entry['audit_id']}", None, {})
            if audit["status"] in {"complete", "partial", "failed"}:
                break
            if time.monotonic() > deadline:
                raise StudyError(f"audit {entry['audit_id']} did not finish in {wait_seconds} s")
            time.sleep(poll_seconds)
        if audit["status"] != "complete":
            raise StudyError(f"audit {entry['audit_id']} ended {audit['status']}; check the worker")
        entry["model_version"] = audit["model_version"]
        by_name = {Path(i["file"]).name: i["study_id"] for i in study["images"]}
        for item in audit["images"]:
            result = call(
                "GET", f"/v1/audits/{entry['audit_id']}/images/{item['image_id']}", None, {}
            )
            entry["images"][item["image_id"]] = {
                "study_id": by_name[item["filename"]],
                "priority": result.get("priority"),
                "review_required": result.get("review_required"),
                "suggestion": _suggestion(
                    result, rules["audit_positive_threshold"], rules["material_present_threshold"]
                ),
            }
        entry["review_queue"] = audit["review_queue"]
    record = {
        "study": study["name"],
        "submitted_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "audits": audits,
    }
    (study_dir / AUDITS_FILE).write_text(json.dumps(record, indent=2) + "\n")
    write_plans(study_dir, study, audits, participants)
    return record


def _assisted_order(entry: dict[str, Any]) -> list[str]:
    """Review-queue priority first, then the remaining images by suggested coverage."""
    queue = list(entry["review_queue"])
    rest = [i for i in entry["images"] if i not in queue]
    rest.sort(key=lambda i: -(entry["images"][i]["priority"] or 0.0))
    return queue + rest


PILOT = "PILOT"


def _plan(
    study: dict[str, Any],
    audits: dict[str, Any],
    code: str,
    arm_index: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    blocks = []
    for set_name, condition in ARMS[arm_index]:
        entry = audits[set_name]
        ids = list(entry["images"])
        order = (
            _assisted_order(entry)
            if condition == "assisted"
            else [ids[i] for i in rng.permutation(len(ids))]
        )
        practice_ids = list(audits["practice"]["images"])
        blocks.append(
            {
                "condition": condition,
                "set": set_name,
                "audit_id": entry["audit_id"],
                "image_ids": order,
                "practice_audit_id": audits["practice"]["audit_id"],
                "practice_image_ids": [practice_ids[i] for i in rng.permutation(len(practice_ids))],
            }
        )
    return {
        "plan_version": PLAN_VERSION,
        "study": study["name"],
        "participant": code,
        "arm": arm_index,
        "audit_positive_threshold": study["rules"]["audit_positive_threshold"],
        "material_present_threshold": study["rules"]["material_present_threshold"],
        "blocks": blocks,
    }


def write_plans(
    study_dir: Path, study: dict[str, Any], audits: dict[str, Any], participants: int
) -> None:
    """One plan per participant (P01, ...) plus a PILOT plan that is never analysed."""
    plans = study_dir / "plans"
    plans.mkdir(exist_ok=True)
    seed = study["rules"]["seed"]
    for p in range(1, participants + 1):
        plan = _plan(
            study, audits, f"P{p:02d}", (p - 1) % len(ARMS), np.random.default_rng([seed, p])
        )
        (plans / f"P{p:02d}.json").write_text(json.dumps(plan, indent=2) + "\n")
    pilot = _plan(study, audits, PILOT, 0, np.random.default_rng([seed, 0]))
    (plans / f"{PILOT}.json").write_text(json.dumps(pilot, indent=2) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("study_dir", type=Path)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--participants", type=int, default=8)
    parser.add_argument("--wait-seconds", type=float, default=600.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        record = submit_study(
            args.study_dir,
            http_transport(args.base_url),
            participants=args.participants,
            wait_seconds=args.wait_seconds,
        )
    except StudyError as exc:
        log.error("%s", exc)
        return 1
    for set_name, entry in record["audits"].items():
        log.info("%s: %s (%s)", set_name, entry["audit_id"], entry["model_version"])
    log.info("wrote %d plans to %s", args.participants, args.study_dir / "plans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
