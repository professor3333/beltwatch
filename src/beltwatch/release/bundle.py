"""Immutable model release bundles.

A release is a directory ``model_releases/<version>/`` containing:

* ``manifest.json``: the version, model kind and file with its SHA-256, the
  preprocessing version, the label map, calibration (temperature), the full
  review policy, provenance, limitations, and licenses
* the model file (a ``best.pt`` checkpoint for neural models, a directory for
  the random forest)
* optionally ``evaluation.json``: the evaluation report it was promoted on

``model_releases/active.json`` names the version that new audits are pinned to.
Rolling back means pointing it at a previous bundle, which restores the model,
preprocessing, calibration, and review policy together.

Usage::

    uv run python -m beltwatch.release.bundle --kind unet \\
        --model models/unet-resnet18-ce/best.pt --version beltwatch-0.1.0 \\
        --evaluation reports/unet_resnet18-val/report.json --activate
"""

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from beltwatch import __version__
from beltwatch.inference.predictor import Predictor
from beltwatch.inference.preprocessing import PREPROCESSING_VERSION
from beltwatch.labels import CLASS_NAMES
from beltwatch.review.policy import ReviewPolicy, load_review_policy

ModelKind = Literal["unet", "segformer", "random-forest", "all-background"]
ACTIVE_FILE = "active.json"

DEFAULT_LIMITATIONS = (
    "Estimates visible image coverage only; not contamination by weight or bale purity.",
    "Trained on one facility's paper-stream conveyor (ZeroWaste-f); not validated elsewhere.",
    "Recognizes four materials: cardboard, soft plastic, rigid plastic, metal.",
    "A background prediction does not certify that material is absent.",
)
DEFAULT_LICENSES = (
    "BeltWatch code: MIT.",
    "ZeroWaste-f data: attributed, noncommercial use (CC BY vs CC BY-NC discrepancy).",
    "Pretrained encoder weights: see THIRD_PARTY_NOTICES.md.",
)


class ReleaseError(RuntimeError):
    """A release bundle is missing, malformed, or fails verification."""


class ModelFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: ModelKind
    path: str | None
    sha256: str | None


class ReleaseManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: str
    created_at: str
    model: ModelFile
    preprocessing_version: str
    label_map: dict[str, str]
    temperature: float = 1.0
    calibration_note: str = "not calibrated (temperature 1.0)"
    review_policy: ReviewPolicy
    provenance: dict[str, Any]
    limitations: tuple[str, ...]
    licenses: tuple[str, ...]
    evaluation_report: str | None = None
    calibration_report: str | None = None


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
    for f in files:
        if path.is_dir():
            h.update(f.relative_to(path).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()


@dataclass(frozen=True)
class Release:
    directory: Path
    manifest: ReleaseManifest

    @property
    def version(self) -> str:
        return self.manifest.version

    def load_predictor(self, device: str = "cpu") -> Predictor:
        model = self.manifest.model
        if model.kind == "all-background":
            from beltwatch.baselines.trivial import AllBackground

            return AllBackground()
        assert model.path is not None
        path = self.directory / model.path
        if model.kind == "random-forest":
            from beltwatch.baselines.random_forest import RandomForestSegmenter
            from beltwatch.inference.predictor import TemperatureScaled

            return TemperatureScaled(RandomForestSegmenter.load(path), self.manifest.temperature)
        from beltwatch.inference.neural import NeuralPredictor

        return NeuralPredictor.from_checkpoint(path, device, self.manifest.temperature)


def load_release(directory: Path, *, verify: bool = True) -> Release:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ReleaseError(f"{directory}: no manifest.json")
    try:
        manifest = ReleaseManifest.model_validate_json(manifest_path.read_text())
    except ValueError as exc:
        raise ReleaseError(f"{manifest_path}: invalid manifest: {exc}") from exc
    if manifest.preprocessing_version != PREPROCESSING_VERSION:
        raise ReleaseError(
            f"{manifest.version}: built for preprocessing {manifest.preprocessing_version}, "
            f"code is {PREPROCESSING_VERSION}"
        )
    if manifest.label_map != {str(k): v for k, v in CLASS_NAMES.items()}:
        raise ReleaseError(f"{manifest.version}: label map differs from beltwatch.labels")
    if verify and manifest.model.path is not None:
        model_path = directory / manifest.model.path
        if not model_path.exists():
            raise ReleaseError(f"{manifest.version}: model file {manifest.model.path} missing")
        if _digest(model_path) != manifest.model.sha256:
            raise ReleaseError(f"{manifest.version}: model checksum mismatch")
    return Release(directory, manifest)


def active_version(releases_dir: Path) -> str:
    pointer = releases_dir / ACTIVE_FILE
    if not pointer.is_file():
        raise ReleaseError(f"{releases_dir}: no {ACTIVE_FILE}; build and activate a release")
    return str(json.loads(pointer.read_text())["version"])


def load_active_release(releases_dir: Path) -> Release:
    return load_release(releases_dir / active_version(releases_dir))


def activate(releases_dir: Path, version: str, reason: str) -> None:
    """Point new audits at ``version`` (also used for rollback), keeping a history."""
    load_release(releases_dir / version)
    pointer = releases_dir / ACTIVE_FILE
    previous = json.loads(pointer.read_text()) if pointer.is_file() else {}
    history = previous.get("history", [])
    if previous.get("version"):
        history.append({k: previous[k] for k in ("version", "activated_at", "reason")})
    record = {
        "version": version,
        "activated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "reason": reason,
        "replaces": previous.get("version"),
        "history": history,
    }
    tmp = pointer.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(pointer)


def build_release(
    releases_dir: Path,
    *,
    version: str,
    kind: ModelKind,
    model_path: Path | None,
    review_policy: ReviewPolicy,
    temperature: float = 1.0,
    calibration_note: str | None = None,
    evaluation_report: Path | None = None,
    calibration_report: Path | None = None,
    allow_foreground_calibration_regression: bool = False,
    provenance: dict[str, Any] | None = None,
    limitations: tuple[str, ...] = DEFAULT_LIMITATIONS,
) -> Release:
    """Create an immutable bundle; refuses to overwrite an existing version."""
    directory = releases_dir / version
    if directory.exists():
        raise ReleaseError(f"release {version} already exists; releases are immutable")
    if kind != "all-background" and (model_path is None or not model_path.exists()):
        raise ReleaseError(f"a {kind} release needs an existing model file")
    staging = releases_dir / f".{version}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    model_file: str | None = None
    if model_path is not None and kind != "all-background":
        model_file = "model" if model_path.is_dir() else f"model{model_path.suffix}"
        target = staging / model_file
        if model_path.is_dir():
            shutil.copytree(model_path, target)
        else:
            shutil.copy2(model_path, target)
    report_name = None
    if evaluation_report is not None:
        report_name = "evaluation.json"
        shutil.copy2(evaluation_report, staging / report_name)
    calibration_name = None
    if calibration_report is not None:
        calibration = json.loads(calibration_report.read_text())
        if not calibration.get("fitted") or calibration.get("split") != "calibration":
            raise ReleaseError(
                "the calibration report must come from a fit on the calibration split"
            )
        temperature = float(calibration["temperature"])
        before, after = calibration["before"], calibration["after"]
        fg_before, fg_after = before["foreground"]["ece"], after["foreground"]["ece"]
        if fg_after > fg_before and not allow_foreground_calibration_regression:
            raise ReleaseError(
                f"calibration worsens foreground ECE ({fg_before:.4f} -> {fg_after:.4f}); "
                "refit with --fit-scope foreground, or explicitly allow the regression"
            )
        calibration_note = (
            "temperature scaling fitted on the calibration split "
            f"({calibration['n_pixels']} pixels, "
            f"{calibration['n_images']} images); ECE overall {before['overall']['ece']:.4f} -> "
            f"{after['overall']['ece']:.4f}, foreground {before['foreground']['ece']:.4f} -> "
            f"{after['foreground']['ece']:.4f} (fit scope: {calibration.get('fit_scope', 'all')})"
        )
        calibration_name = "calibration.json"
        shutil.copy2(calibration_report, staging / calibration_name)

    manifest = ReleaseManifest(
        version=version,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        model=ModelFile(
            kind=kind,
            path=model_file,
            sha256=_digest(staging / model_file) if model_file else None,
        ),
        preprocessing_version=PREPROCESSING_VERSION,
        label_map={str(k): v for k, v in CLASS_NAMES.items()},
        temperature=temperature,
        calibration_note=calibration_note
        or ("not calibrated (temperature 1.0)" if temperature == 1.0 else "temperature scaling"),
        review_policy=review_policy,
        provenance={"beltwatch_version": __version__, **(provenance or {})},
        limitations=limitations,
        licenses=DEFAULT_LICENSES,
        evaluation_report=report_name,
        calibration_report=calibration_name,
    )
    (staging / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    staging.replace(directory)
    return load_release(directory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build (and optionally activate) a release.")
    parser.add_argument("--version", required=True)
    parser.add_argument(
        "--kind", required=True, choices=["unet", "segformer", "random-forest", "all-background"]
    )
    parser.add_argument("--model", type=Path, help="checkpoint file or model directory")
    parser.add_argument("--review-policy", type=Path, default=Path("configs/review_policy.yaml"))
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--evaluation", type=Path, help="evaluation report.json to include")
    parser.add_argument("--calibration", type=Path, help="calibration.json; sets the temperature")
    parser.add_argument(
        "--allow-foreground-calibration-regression",
        action="store_true",
        help="accept a temperature that worsens foreground ECE (recorded in the manifest)",
    )
    parser.add_argument("--releases-dir", type=Path, default=Path("model_releases"))
    parser.add_argument("--note", default="", help="provenance note, e.g. why it was built")
    parser.add_argument("--activate", action="store_true")
    args = parser.parse_args(argv)

    release = build_release(
        args.releases_dir,
        version=args.version,
        kind=args.kind,
        model_path=args.model,
        review_policy=load_review_policy(args.review_policy),
        temperature=args.temperature,
        evaluation_report=args.evaluation,
        calibration_report=args.calibration,
        allow_foreground_calibration_regression=args.allow_foreground_calibration_regression,
        provenance={"note": args.note} if args.note else None,
    )
    print(f"built {release.directory}")
    if args.activate:
        activate(args.releases_dir, release.version, args.note or "activated at build time")
        print(f"activated {release.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
