"""FastAPI application: the audit workflow over HTTP.

Run with::

    uv run uvicorn beltwatch.api.app:create_app --factory --port 8000

Errors use one JSON shape, ``{"error": {"code", "message", "details"}}``, with
explicit codes for invalid input, oversized uploads, a full queue, an
unavailable worker or model release, and idempotency conflicts. Uploaded
image contents never appear in logs.
"""

import base64
import binascii
import csv
import hashlib
import io
import json
import logging
import time
import uuid
from collections.abc import Awaitable, Callable, Iterator
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import FastAPI, File, Form, Header, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from beltwatch import __version__
from beltwatch.api.settings import Settings
from beltwatch.inference.postprocessing import rasterize_region
from beltwatch.inference.preprocessing import PREPROCESSING_VERSION
from beltwatch.jobs.store import IdempotencyConflict, JobStore, NewImage, new_id
from beltwatch.labels import CLASS_NAMES, TARGET_CLASS_IDS
from beltwatch.release.bundle import Release, ReleaseError, load_active_release

log = logging.getLogger("beltwatch.api")

ALLOWED_FORMATS = {
    "PNG": ("image/png", ".png"),
    "JPEG": ("image/jpeg", ".jpg"),
    "WEBP": ("image/webp", ".webp"),
}
TARGET_NAMES = tuple(CLASS_NAMES[c] for c in TARGET_CLASS_IDS)
FRONTEND_DIR = Path(__file__).resolve().parents[3] / "frontend"
MIN_REGION_FRACTION = 0.01
"""An inspection region must cover at least this fraction of each image."""
MEASUREMENT_NOTE = (
    "estimated visible coverage of the inspection region (not contamination by weight)"
)


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.status, self.code, self.message, self.details = status, code, message, details


class AuditOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera_id: str = Field(default="demo-camera", min_length=1, max_length=64, pattern=r"^[\w.-]+$")
    inspection_region: list[list[float]] = Field(
        default=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], min_length=3, max_length=64
    )
    profile: Literal["paper-stream-v1"] = "paper-stream-v1"
    training_consent: bool = False

    @field_validator("inspection_region")
    @classmethod
    def _normalized(cls, region: list[list[float]]) -> list[list[float]]:
        for vertex in region:
            if len(vertex) != 2 or not all(0.0 <= v <= 1.0 for v in vertex):
                raise ValueError("vertices must be [x, y] pairs normalized to [0, 1]")
        return region


class FeedbackIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image_id: str
    decision: Literal["confirmed", "rejected", "corrected", "needs_retake"]
    materials_present: list[Literal["cardboard", "soft_plastic", "rigid_plastic", "metal"]] = []
    reviewer: str | None = Field(default=None, max_length=64)
    note: str | None = Field(default=None, max_length=1000)
    corrected_mask_png_base64: str | None = None


def _decode_upload(data: bytes, filename: str, settings: Settings) -> tuple[str, str, int, int]:
    """Validate the actual image content; returns (content_type, extension, width, height)."""
    try:
        with Image.open(io.BytesIO(data)) as img:
            fmt = img.format or ""
            width, height = img.size
            if fmt not in ALLOWED_FORMATS:
                raise ApiError(
                    415,
                    "unsupported_image_type",
                    f"{filename}: {fmt or 'unknown'} is not PNG, JPEG, or WebP",
                )
            if width * height > settings.max_image_pixels:
                raise ApiError(
                    413,
                    "image_too_large",
                    f"{filename}: {width}x{height} exceeds the pixel limit",
                    max_pixels=settings.max_image_pixels,
                )
            if min(width, height) < 32:
                raise ApiError(422, "image_too_small", f"{filename}: {width}x{height} is too small")
            img.load()
    except ApiError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ApiError(422, "invalid_image", f"{filename}: cannot decode image ({exc})") from exc
    content_type, ext = ALLOWED_FORMATS[fmt]
    return content_type, ext, width, height


class _StoreCollector(Collector):
    def __init__(self, store: JobStore, settings: Settings) -> None:
        self.store, self.settings = store, settings

    def collect(self) -> Iterator[GaugeMetricFamily]:
        yield GaugeMetricFamily(
            "beltwatch_queue_length", "Audits queued or running", value=self.store.pending_count()
        )
        yield GaugeMetricFamily(
            "beltwatch_oldest_pending_audit_seconds",
            "Age of the oldest pending audit",
            value=self.store.oldest_pending_age(),
        )
        yield GaugeMetricFamily(
            "beltwatch_live_workers",
            "Workers with a recent heartbeat",
            value=len(self.store.live_workers(self.settings.worker_stale_seconds)),
        )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    store = JobStore(
        settings.database_path,
        lease_seconds=settings.lease_seconds,
        max_attempts=settings.max_attempts,
    )
    registry = CollectorRegistry()
    registry.register(_StoreCollector(store, settings))
    requests_total = Counter(
        "beltwatch_http_requests_total",
        "HTTP requests",
        ["method", "route", "status"],
        registry=registry,
    )
    latency = Histogram(
        "beltwatch_http_request_seconds", "HTTP request latency", ["route"], registry=registry
    )
    uploads_rejected = Counter(
        "beltwatch_uploads_rejected_total", "Rejected uploads", ["code"], registry=registry
    )

    app = FastAPI(
        title="BeltWatch", version=__version__, description="Visual contamination auditing API"
    )
    app.state.settings, app.state.store = settings, store

    def active_release() -> Release:
        try:
            return load_active_release(settings.releases_dir)
        except ReleaseError as exc:
            raise ApiError(503, "model_unavailable", f"no usable model release: {exc}") from exc

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, exc: ApiError) -> JSONResponse:
        if exc.code in {
            "invalid_image",
            "unsupported_image_type",
            "image_too_large",
            "file_too_large",
        }:
            uploads_rejected.labels(exc.code).inc()
        headers = {"Retry-After": "30"} if exc.status == 503 else None
        return JSONResponse(
            {"error": {"code": exc.code, "message": exc.message, "details": exc.details}},
            status_code=exc.status,
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            {
                "error": {
                    "code": "invalid_request",
                    "message": "request validation failed",
                    "details": {"errors": json.loads(json.dumps(exc.errors(), default=str))},
                }
            },
            status_code=422,
        )

    @app.middleware("http")
    async def _log_requests(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - started
        route = getattr(request.scope.get("route"), "path", "unmatched")
        requests_total.labels(request.method, route, str(response.status_code)).inc()
        latency.labels(route).observe(elapsed)
        response.headers["X-Request-ID"] = request_id
        log.info(
            json.dumps(
                {
                    "event": "http_request",
                    "request_id": request_id,
                    "method": request.method,
                    "route": route,
                    "status": response.status_code,
                    "ms": round(elapsed * 1000, 1),
                }
            )
        )
        return response

    def _audit_or_404(audit_id: str) -> dict[str, Any]:
        audit = store.get_audit(audit_id)
        if audit is None:
            raise ApiError(404, "audit_not_found", f"audit {audit_id} does not exist")
        return audit

    def _image_summary(audit_id: str, image: dict[str, Any]) -> dict[str, Any]:
        result = image["result"] or {}
        return {
            "image_id": image["id"],
            "filename": image["original_filename"],
            "status": image["status"],
            "error": image["error"],
            "total_target_coverage": result.get("total_target_coverage"),
            "review_required": result.get("review_required"),
            "route": result.get("route"),
            "priority": result.get("priority"),
            "detail_url": f"/v1/audits/{audit_id}/images/{image['id']}",
        }

    @app.post("/v1/audits", status_code=201)
    async def create_audit(
        response: Response,
        files: Annotated[
            list[UploadFile], File(description="Conveyor snapshots (PNG, JPEG, WebP)")
        ],
        options: Annotated[str, Form(description="JSON audit options")] = "{}",
        idempotency_key: Annotated[str | None, Header(max_length=128)] = None,
    ) -> dict[str, Any]:
        try:
            opts = AuditOptions.model_validate_json(options)
        except ValidationError as exc:
            raise ApiError(
                422,
                "invalid_options",
                "invalid audit options",
                errors=json.loads(exc.json(include_url=False)),
            ) from exc
        if not 1 <= len(files) <= settings.max_files_per_audit:
            raise ApiError(
                422,
                "invalid_file_count",
                f"send between 1 and {settings.max_files_per_audit} images",
            )
        if settings.require_worker and not store.live_workers(settings.worker_stale_seconds):
            raise ApiError(503, "worker_unavailable", "no inference worker is available")
        release = active_release()
        if store.pending_count() >= settings.max_pending_audits:
            raise ApiError(503, "queue_full", "too many audits are waiting; retry later")

        audit_id = new_id("audit")
        prepared = []
        for upload in files:
            data = await upload.read(settings.max_file_bytes + 1)
            name = (upload.filename or "upload")[:200]
            if len(data) > settings.max_file_bytes:
                raise ApiError(
                    413, "file_too_large", f"{name} exceeds {settings.max_file_bytes} bytes"
                )
            content_type, ext, width, height = _decode_upload(data, name, settings)
            region_px = int(rasterize_region(opts.inspection_region, width, height).sum())
            if region_px < MIN_REGION_FRACTION * width * height:
                raise ApiError(
                    422,
                    "inspection_region_too_small",
                    f"the inspection region covers {region_px} pixels of {name}",
                    min_fraction=MIN_REGION_FRACTION,
                )
            prepared.append(
                (name, data, content_type, ext, width, height, hashlib.sha256(data).hexdigest())
            )

        request_hash = hashlib.sha256(
            json.dumps(
                {"options": opts.model_dump(), "images": [p[6] for p in prepared]}, sort_keys=True
            ).encode()
        ).hexdigest()
        image_ids = [new_id("img") for _ in prepared]
        written: list[Path] = []
        new_images = []
        for image_id, (name, data, content_type, ext, width, height, sha) in zip(
            image_ids, prepared, strict=True
        ):
            rel = Path("uploads") / audit_id / f"{image_id}{ext}"
            path = settings.data_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            written.append(path)
            new_images.append(NewImage(name, sha, width, height, str(rel), content_type))
        try:
            final_id, created = store.create_audit(
                camera_id=opts.camera_id,
                profile=opts.profile,
                inspection_region=opts.inspection_region,
                model_version=release.version,
                preprocessing_version=PREPROCESSING_VERSION,
                review_policy_version=release.manifest.review_policy.version,
                images=new_images,
                request_hash=request_hash,
                idempotency_key=idempotency_key,
                training_consent=opts.training_consent,
                retention_hours=settings.retention_hours,
                audit_id=audit_id,
                image_ids=image_ids,
            )
        except IdempotencyConflict as exc:
            for path in written:
                path.unlink(missing_ok=True)
            raise ApiError(
                409, "idempotency_conflict", "this Idempotency-Key was used for a different request"
            ) from exc
        if not created:
            for path in written:
                path.unlink(missing_ok=True)
            (settings.data_dir / "uploads" / audit_id).rmdir()
            response.status_code = 200
        audit = _audit_or_404(final_id)
        return {
            "audit_id": final_id,
            "status": audit["status"],
            "model_version": audit["model_version"],
            "status_url": f"/v1/audits/{final_id}",
            "images": [
                {"image_id": i["id"], "filename": i["original_filename"]}
                for i in store.list_images(final_id)
            ],
        }

    @app.get("/v1/audits/{audit_id}")
    def get_audit(audit_id: str) -> dict[str, Any]:
        audit = _audit_or_404(audit_id)
        images = store.list_images(audit_id)
        counts = audit["image_counts"]
        summaries = [_image_summary(audit_id, i) for i in images]
        queue = sorted(
            (s for s in summaries if s["review_required"]), key=lambda s: -(s["priority"] or 0)
        )
        return {
            "audit_id": audit_id,
            "status": audit["status"],
            "created_at": audit["created_at"],
            "completed_at": audit["completed_at"],
            "error": audit["error"],
            "camera_id": audit["camera_id"],
            "profile": audit["profile"],
            "inspection_region": audit["inspection_region"],
            "model_version": audit["model_version"],
            "preprocessing_version": audit["preprocessing_version"],
            "review_policy_version": audit["review_policy_version"],
            "expires_at": audit["expires_at"],
            "progress": {
                "total": len(images),
                "complete": counts.get("complete", 0),
                "failed": counts.get("failed", 0),
                "pending": counts.get("pending", 0),
            },
            "images": summaries,
            "review_queue": [s["image_id"] for s in queue],
        }

    @app.get("/v1/audits/{audit_id}/images/{image_id}")
    def get_image(audit_id: str, image_id: str) -> dict[str, Any]:
        _audit_or_404(audit_id)
        image = store.get_image(audit_id, image_id)
        if image is None:
            raise ApiError(
                404, "image_not_found", f"image {image_id} is not part of audit {audit_id}"
            )
        base = f"/v1/audits/{audit_id}/artifacts"
        artifacts = {"original_url": f"{base}/original_{image_id}"}
        if image["status"] == "complete":
            artifacts |= {
                f"{k}_url": f"{base}/{k}_{image_id}" for k in ("overlay", "mask", "uncertainty")
            }
        feedback = [f for f in store.list_feedback(audit_id) if f["image_id"] == image_id]
        return {
            "image_id": image_id,
            "audit_id": audit_id,
            "filename": image["original_filename"],
            "width": image["width"],
            "height": image["height"],
            "sha256": image["sha256"],
            "status": image["status"],
            "error": image["error"],
            **(image["result"] or {}),
            **artifacts,
            "feedback": feedback,
        }

    @app.get("/v1/audits/{audit_id}/artifacts/{artifact_id}")
    def get_artifact(audit_id: str, artifact_id: str) -> FileResponse:
        artifact = store.get_artifact(audit_id, artifact_id)
        if artifact is None:
            raise ApiError(404, "artifact_not_found", f"artifact {artifact_id} does not exist")
        root = settings.data_dir.resolve()
        path = (settings.data_dir / artifact["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ApiError(404, "artifact_not_found", f"artifact {artifact_id} is unavailable")
        return FileResponse(path, media_type=artifact["content_type"])

    @app.post("/v1/audits/{audit_id}/feedback", status_code=201)
    def add_feedback(audit_id: str, body: FeedbackIn) -> dict[str, Any]:
        audit = _audit_or_404(audit_id)
        image = store.get_image(audit_id, body.image_id)
        if image is None:
            raise ApiError(
                404, "image_not_found", f"image {body.image_id} is not part of audit {audit_id}"
            )
        if image["status"] != "complete":
            raise ApiError(
                409, "image_not_complete", "feedback is accepted only for completed images"
            )
        correction_id = None
        if body.corrected_mask_png_base64:
            try:
                raw = base64.b64decode(body.corrected_mask_png_base64, validate=True)
                with Image.open(io.BytesIO(raw)) as img:
                    mask = np.asarray(img)
            except (binascii.Error, UnidentifiedImageError, OSError) as exc:
                raise ApiError(
                    422, "invalid_mask", "corrected mask is not a valid base64 PNG"
                ) from exc
            if mask.ndim != 2 or mask.shape != (image["height"], image["width"]):
                raise ApiError(
                    422,
                    "invalid_mask",
                    "corrected mask must be single-channel and match the image size",
                )
            if not set(np.unique(mask).tolist()) <= set(CLASS_NAMES):
                raise ApiError(422, "invalid_mask", "corrected mask contains unknown class IDs")
            rel = (
                Path("artifacts")
                / audit_id
                / f"{body.image_id}_correction_{uuid.uuid4().hex[:8]}.png"
            )
            (settings.data_dir / rel).parent.mkdir(parents=True, exist_ok=True)
            (settings.data_dir / rel).write_bytes(raw)
            correction_id = store.add_artifact(
                audit_id, body.image_id, "correction", str(rel), "image/png"
            )
        feedback_id = store.add_feedback(
            audit_id=audit_id,
            image_id=body.image_id,
            decision=body.decision,
            materials_present=body.materials_present,
            model_version=audit["model_version"],
            reviewer=body.reviewer,
            note=body.note,
            correction_artifact_id=correction_id,
        )
        return {"feedback_id": feedback_id, "correction_artifact_id": correction_id}

    @app.get("/v1/audits/{audit_id}/report", response_model=None)
    def report(audit_id: str, format: Literal["json", "csv"] = "json") -> dict[str, Any] | Response:
        audit = _audit_or_404(audit_id)
        images = store.list_images(audit_id)
        feedback = store.list_feedback(audit_id)
        latest = {f["image_id"]: f for f in feedback}
        if format == "json":
            return {
                "audit": {
                    k: audit[k]
                    for k in (
                        "id",
                        "created_at",
                        "completed_at",
                        "status",
                        "camera_id",
                        "profile",
                        "inspection_region",
                        "model_version",
                        "preprocessing_version",
                        "review_policy_version",
                        "training_consent",
                    )
                },
                "measurement": MEASUREMENT_NOTE,
                "images": [
                    {
                        "image_id": i["id"],
                        "filename": i["original_filename"],
                        "sha256": i["sha256"],
                        "status": i["status"],
                        "error": i["error"],
                        "result": i["result"],
                        "feedback": [f for f in feedback if f["image_id"] == i["id"]],
                    }
                    for i in images
                ],
            }
        out = io.StringIO()
        fields = [
            "image_id",
            "filename",
            "status",
            "error",
            *[f"coverage_{n}" for n in TARGET_NAMES],
            "total_target_coverage",
            "uncertain_pixel_fraction",
            "review_required",
            "review_reasons",
            "route",
            "reviewer_decision",
            "reviewer_materials",
            "reviewed_at",
            "model_version",
        ]
        writer = csv.DictWriter(out, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for i in images:
            r, f = i["result"] or {}, latest.get(i["id"], {})
            writer.writerow(
                {
                    "image_id": i["id"],
                    "filename": i["original_filename"],
                    "status": i["status"],
                    "error": i["error"],
                    **{
                        f"coverage_{n}": (r.get("visible_coverage") or {}).get(n)
                        for n in TARGET_NAMES
                    },
                    "total_target_coverage": r.get("total_target_coverage"),
                    "uncertain_pixel_fraction": r.get("uncertain_pixel_fraction"),
                    "review_required": r.get("review_required"),
                    "review_reasons": ";".join(r.get("review_reasons", [])),
                    "route": r.get("route"),
                    "reviewer_decision": f.get("decision"),
                    "reviewer_materials": ";".join(f.get("materials_present", [])),
                    "reviewed_at": f.get("created_at"),
                    "model_version": audit["model_version"],
                }
            )
        return Response(
            out.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="{audit_id}.csv"'},
        )

    @app.delete("/v1/audits/{audit_id}", status_code=204)
    def delete_audit(audit_id: str) -> Response:
        _audit_or_404(audit_id)
        for rel in store.delete_audit(audit_id):
            (settings.data_dir / rel).unlink(missing_ok=True)
        for sub in ("uploads", "artifacts"):
            directory = settings.data_dir / sub / audit_id
            if directory.is_dir():
                for leftover in directory.iterdir():
                    leftover.unlink()
                directory.rmdir()
        return Response(status_code=204)

    @app.get("/v1/model")
    def model_info() -> dict[str, Any]:
        release = active_release()
        m = release.manifest
        evaluation = None
        if m.evaluation_report:
            report_data = json.loads((release.directory / m.evaluation_report).read_text())
            evaluation = {k: report_data.get(k) for k in ("split", "n_images", "split_manifest_id")}
            evaluation["foreground_macro_iou"] = report_data.get("segmentation", {}).get(
                "foreground_macro_iou"
            )
            evaluation["iou"] = report_data.get("segmentation", {}).get("iou")
            evaluation["coverage_mae_pp"] = report_data.get("coverage", {}).get("total_mae_pp")
        return {
            "version": m.version,
            "created_at": m.created_at,
            "model_kind": m.model.kind,
            "model_sha256": m.model.sha256,
            "preprocessing_version": m.preprocessing_version,
            "label_map": m.label_map,
            "temperature": m.temperature,
            "calibration": m.calibration_note,
            "review_policy": m.review_policy.model_dump(),
            "provenance": m.provenance,
            "limitations": list(m.limitations),
            "licenses": list(m.licenses),
            "evaluation": evaluation,
            "measurement": "estimated visible coverage (not contamination by weight)",
        }

    @app.get("/health/live")
    def live() -> dict[str, str]:
        return {"status": "alive", "version": __version__}

    @app.get("/health/ready")
    def ready() -> JSONResponse:
        checks: dict[str, Any] = {}
        try:
            store.pending_count()
            checks["database"] = "ok"
        except Exception as exc:  # pragma: no cover - defensive
            checks["database"] = f"error: {exc}"
        try:
            checks["model_release"] = load_active_release(settings.releases_dir).version
        except ReleaseError as exc:
            checks["model_release"] = f"error: {exc}"
        workers = store.live_workers(settings.worker_stale_seconds)
        checks["workers"] = len(workers)
        ok = checks["database"] == "ok" and not str(checks["model_release"]).startswith("error")
        ok = ok and (bool(workers) or not settings.require_worker)
        return JSONResponse(
            {"status": "ready" if ok else "not_ready", "checks": checks},
            status_code=200 if ok else 503,
        )

    @app.get("/metrics")
    def metrics(authorization: Annotated[str | None, Header()] = None) -> PlainTextResponse:
        if settings.metrics_token and authorization != f"Bearer {settings.metrics_token}":
            raise ApiError(401, "unauthorized", "metrics require the configured bearer token")
        return PlainTextResponse(
            generate_latest(registry).decode(), media_type="text/plain; version=0.0.4"
        )

    frontend = settings.frontend_dir or FRONTEND_DIR
    if frontend.is_dir():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return app
