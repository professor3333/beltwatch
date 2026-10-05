"""Deployment smoke test for a running BeltWatch service (standard library only).

Checks, in order: readiness, the active model version (optionally the expected
one), a real inference on a generated image, retrieval of the overlay
artifact, the JSON report, and deletion of the test audit.

Usage::

    python scripts/deploy_smoke.py --base-url https://localhost --insecure \\
        --expect-version beltwatch-0.1.0

Exits non-zero with a reason on the first failed check.
"""

import argparse
import json
import random
import ssl
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid
import zlib


def synthetic_png(width: int = 160, height: int = 96, seed: int = 0) -> bytes:
    """A noisy RGB PNG: sharp and well exposed enough to pass input-quality checks."""
    rng = random.Random(seed)
    rows = b"".join(
        b"\x00" + bytes(rng.randint(40, 215) for _ in range(width * 3)) for _ in range(height)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(rows))
        + chunk(b"IEND", b"")
    )


class Smoke:
    def __init__(self, base_url: str, insecure: bool, timeout: float) -> None:
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.context = ssl._create_unverified_context() if insecure else None

    def request(
        self,
        method: str,
        path: str,
        body: bytes | None = None,
        headers: dict[str, str] | None = None,
    ):
        req = urllib.request.Request(
            self.base + path, data=body, method=method, headers=headers or {}
        )
        with urllib.request.urlopen(req, timeout=self.timeout, context=self.context) as resp:
            return resp.status, resp.headers, resp.read()

    def json(self, method: str, path: str, **kwargs):
        status, _, data = self.request(method, path, **kwargs)
        return status, json.loads(data) if data else None

    def wait_ready(self, seconds: float) -> dict:
        deadline = time.monotonic() + seconds
        last = "no response"
        while time.monotonic() < deadline:
            try:
                status, body = self.json("GET", "/health/ready")
                if status == 200:
                    return body
            except urllib.error.HTTPError as exc:
                last = exc.read().decode(errors="replace")[:300]
            except (urllib.error.URLError, ConnectionError, TimeoutError) as exc:
                last = str(exc)
            time.sleep(2)
        raise SystemExit(f"FAIL readiness: not ready after {seconds:.0f}s ({last})")

    def run_audit(self) -> tuple[str, dict]:
        boundary = uuid.uuid4().hex
        options = json.dumps({"camera_id": "deploy-smoke"})
        disposition = "Content-Disposition: form-data; name="
        parts = [
            f'--{boundary}\r\n{disposition}"options"\r\n\r\n{options}\r\n'.encode(),
            f'--{boundary}\r\n{disposition}"files"; filename="smoke.png"\r\n'
            f"Content-Type: image/png\r\n\r\n".encode()
            + synthetic_png(seed=int(time.time()))
            + b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Idempotency-Key": uuid.uuid4().hex,
        }
        status, created = self.json("POST", "/v1/audits", body=b"".join(parts), headers=headers)
        if status != 201:
            raise SystemExit(f"FAIL create audit: HTTP {status}")
        audit_id = created["audit_id"]
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            _, audit = self.json("GET", f"/v1/audits/{audit_id}")
            if audit["status"] in {"complete", "partial", "failed"}:
                return audit_id, audit
            time.sleep(1)
        raise SystemExit(f"FAIL inference: audit {audit_id} did not finish within 120s")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a deployed BeltWatch service.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--expect-version", help="fail unless this model release is active")
    parser.add_argument("--insecure", action="store_true", help="skip TLS verification (local CA)")
    parser.add_argument("--ready-timeout", type=float, default=120)
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args(argv)

    smoke = Smoke(args.base_url, args.insecure, args.timeout)
    ready = smoke.wait_ready(args.ready_timeout)
    print(f"ok   readiness: {ready['checks']}")

    _, model = smoke.json("GET", "/v1/model")
    if args.expect_version and model["version"] != args.expect_version:
        raise SystemExit(f"FAIL model version: {model['version']} != {args.expect_version}")
    kind, preprocessing = model["model_kind"], model["preprocessing_version"]
    print(f"ok   model: {model['version']} ({kind}, preprocessing {preprocessing})")

    audit_id, audit = smoke.run_audit()
    if audit["status"] != "complete":
        raise SystemExit(
            f"FAIL inference: audit {audit_id} ended {audit['status']}: {audit.get('error')}"
        )
    if audit["model_version"] != model["version"]:
        raise SystemExit(
            f"FAIL pinning: audit used {audit['model_version']}, active is {model['version']}"
        )
    image_id = audit["images"][0]["image_id"]
    print(f"ok   inference: {audit_id} complete with {audit['model_version']}")

    status, headers, data = smoke.request(
        "GET", f"/v1/audits/{audit_id}/artifacts/overlay_{image_id}"
    )
    if (
        status != 200
        or headers.get("Content-Type") != "image/png"
        or not data.startswith(b"\x89PNG")
    ):
        raise SystemExit("FAIL artifact: overlay is not a PNG")
    print(f"ok   artifact: overlay {len(data)} bytes")

    _, report = smoke.json("GET", f"/v1/audits/{audit_id}/report")
    if report["audit"]["model_version"] != model["version"] or len(report["images"]) != 1:
        raise SystemExit("FAIL report: unexpected content")
    print("ok   report")

    status, _, _ = smoke.request("DELETE", f"/v1/audits/{audit_id}")
    if status != 204:
        raise SystemExit(f"FAIL delete: HTTP {status}")
    print("ok   delete")
    print("PASS deploy smoke test")
    return 0


if __name__ == "__main__":
    sys.exit(main())
