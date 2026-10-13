"""Tune the random-forest baseline on val (``configs/random_forest_tuning.yaml``).

The acceptance gate compares the neural model against a **tuned** classical
baseline, so the forest gets a small, declared search rather than defaults.
Each forest setting is fitted once on train. Every class-frequency correction
(``prior_power``) is then scored on val through the standard evaluation runner,
so candidates and models share one metric path. The candidate with the highest
val foreground macro IoU is copied to ``--out`` and its report to
``<reports-dir>/random-forest-val``. Every candidate's report and the full
search record (``tuning.json``) are kept. The test split is never read.

Usage::

    uv run python -m beltwatch.baselines.tune_random_forest

A rerun reuses fitted forests and finished candidate reports under
``--work-dir``, so an interrupted search continues where it stopped.
"""

import argparse
import json
import logging
import platform
import shutil
import time
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from beltwatch import __version__
from beltwatch.baselines.random_forest import (
    RandomForestConfig,
    RandomForestSegmenter,
    train_and_save,
)
from beltwatch.data.config import load_data_config
from beltwatch.data.dataset import load_samples
from beltwatch.data.masks import load_source_remap
from beltwatch.evaluation.runner import evaluate_images, save_report, unseen_sequence_ids
from beltwatch.training.loop import git_commit

log = logging.getLogger(__name__)

DEFAULTS = RandomForestConfig()


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ForestSetting(_Strict):
    long_side: int = Field(default=DEFAULTS.long_side, ge=32)
    pixels_per_class: int = Field(default=DEFAULTS.pixels_per_class, ge=1)
    n_estimators: int = Field(default=DEFAULTS.n_estimators, ge=1)
    max_depth: int | None = Field(default=DEFAULTS.max_depth, ge=1)
    min_samples_leaf: int = Field(default=DEFAULTS.min_samples_leaf, ge=1)


class TuningConfig(_Strict):
    version: str
    forests: list[ForestSetting] = Field(min_length=1)
    prior_powers: list[float] = Field(min_length=1)


def load_tuning_config(path: Path) -> TuningConfig:
    with path.open(encoding="utf-8") as fh:
        config = TuningConfig.model_validate(yaml.safe_load(fh))
    if any(p < 0 for p in config.prior_powers):
        raise ValueError("prior_powers must be non-negative")
    return config


def candidate_name(forest_index: int, power: float) -> str:
    return f"forest{forest_index}-prior{power:g}"


def tune(
    config_path: Path,
    data_config: Path,
    work_dir: Path,
    out: Path,
    reports_dir: Path,
    *,
    keep_all: bool = False,
) -> dict[str, Any]:
    tuning = load_tuning_config(config_path)
    data = load_data_config(data_config)
    name = data.source.name
    splits_csv = data.paths.splits_dir / f"{name}-splits.csv"
    images_csv = data.paths.manifests_dir / f"{name}-images.csv"
    split_manifest_id = json.loads((data.paths.splits_dir / f"{name}-splits.json").read_text())[
        "split_manifest_id"
    ]
    remap = load_source_remap(data.paths.manifests_dir / f"{name}-validation.json")
    train_samples = load_samples(splits_csv, images_csv, data.paths.raw_dir, "train")
    val_samples = load_samples(splits_csv, images_csv, data.paths.raw_dir, "val")
    slices = {"unseen_sequence": unseen_sequence_ids(splits_csv)}

    candidates: list[dict[str, Any]] = []
    for i, setting in enumerate(tuning.forests):
        forest_dir = work_dir / f"forest{i}"
        forest_config = RandomForestConfig(**setting.model_dump())
        reused = (forest_dir / "model.joblib").exists()
        if reused:
            segmenter = RandomForestSegmenter.load(forest_dir)
            if segmenter.config != forest_config:
                raise ValueError(f"{forest_dir} holds a different forest; remove it to refit")
            log.info("reusing %s", forest_dir)
        else:
            segmenter = train_and_save(
                train_samples, remap, forest_config, forest_dir, split_manifest_id
            )
        fit_seconds = json.loads((forest_dir / "metadata.json").read_text())["fit_seconds"]

        for power in tuning.prior_powers:
            candidate = candidate_name(i, power)
            report_dir = work_dir / "reports" / f"{candidate}-val"
            report_path = report_dir / "report.json"
            if reused and report_path.exists():  # a refitted forest is always rescored
                report = json.loads(report_path.read_text())
            else:
                start = time.perf_counter()
                results = evaluate_images(segmenter.with_prior_power(power), val_samples, remap)
                report = save_report(
                    results,
                    report_dir,
                    model=segmenter.name,
                    split="val",
                    split_manifest_id=split_manifest_id,
                    slices=slices,
                )
                log.info("%s scored in %.0f s", candidate, time.perf_counter() - start)
            seg = report["segmentation"]
            candidates.append(
                {
                    "name": candidate,
                    "forest": setting.model_dump(),
                    "prior_power": power,
                    "fit_seconds": fit_seconds,
                    "val_foreground_macro_iou": seg["foreground_macro_iou"],
                    "val_iou": seg["iou"],
                    "val_coverage_mae_pp": report["coverage"]["total_mae_pp"],
                    "report_dir": str(report_dir),
                }
            )
            log.info("%s: val foreground macro IoU %s", candidate, seg["foreground_macro_iou"])

    scored = [c for c in candidates if c["val_foreground_macro_iou"] is not None]
    if not scored:
        raise RuntimeError("no candidate has a defined val foreground macro IoU")
    best = max(scored, key=lambda c: c["val_foreground_macro_iou"])  # first wins a tie
    forest_index = int(best["name"].removeprefix("forest").split("-")[0])

    best_forest = work_dir / f"forest{forest_index}"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_forest / "model.joblib", out / "model.joblib")
    metadata = json.loads((best_forest / "metadata.json").read_text())
    metadata["config"]["prior_power"] = best["prior_power"]
    metadata["tuning"] = {"version": tuning.version, "candidate": best["name"]}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    final_report = reports_dir / "random-forest-val"
    if final_report.exists():
        shutil.rmtree(final_report)
    shutil.copytree(best["report_dir"], final_report)

    record = {
        "version": tuning.version,
        "selection": "highest val foreground_macro_iou; ties keep the earlier candidate",
        "split": "val",
        "split_manifest_id": split_manifest_id,
        "selected": best["name"],
        "candidates": candidates,
        "git_commit": git_commit(),
        "beltwatch_version": __version__,
        "platform": platform.platform(),
        "grid": tuning.model_dump(),
    }
    (out / "tuning.json").write_text(json.dumps(record, indent=2) + "\n")
    if not keep_all:
        for i in range(len(tuning.forests)):
            (work_dir / f"forest{i}" / "model.joblib").unlink(missing_ok=True)
    log.info("selected %s -> %s and %s", best["name"], out, final_report)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tune the random-forest baseline on val.")
    parser.add_argument("--grid", type=Path, default=Path("configs/random_forest_tuning.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--work-dir", type=Path, default=Path("models/random-forest-tuning"))
    parser.add_argument("--out", type=Path, default=Path("models/random-forest"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument(
        "--keep-all",
        action="store_true",
        help="keep every fitted forest (by default only the selected one is kept, in --out)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    tune(args.grid, args.config, args.work_dir, args.out, args.reports_dir, keep_all=args.keep_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
