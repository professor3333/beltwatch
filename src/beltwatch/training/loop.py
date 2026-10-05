"""The training loop: transfer learning with tracked, resumable runs.

Schedule: the decoder and head train alone for ``frozen_encoder_epochs``, then
the whole network fine-tunes with a smaller encoder learning rate. Learning
rates warm up linearly, then decay along a cosine curve. Gradients accumulate
over ``accumulation_steps`` batches, and mixed precision is used on CUDA.

Every epoch writes a resumable ``latest.pt``. Whenever validation foreground
macro IoU improves, an inference checkpoint ``best.pt`` is written. Runs are
logged to MLflow with the configuration, metrics, and lineage tags: git
commit, split manifest ID, dataset checksum, preprocessing version, and the
checkpoint digest.

Usage::

    uv run python scripts/train.py --config configs/unet.yaml
"""

import argparse
import hashlib
import json
import logging
import math
import platform
import random
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from torch.utils.data import DataLoader

from beltwatch import __version__
from beltwatch.data.config import load_data_config
from beltwatch.data.dataset import SegmentationDataset, load_samples
from beltwatch.data.masks import load_source_remap
from beltwatch.evaluation.coverage import coverage_errors
from beltwatch.evaluation.runner import evaluate_images
from beltwatch.evaluation.segmentation import metrics_from_confusion
from beltwatch.inference.neural import NeuralPredictor, save_inference_checkpoint
from beltwatch.inference.preprocessing import PREPROCESSING_VERSION
from beltwatch.models.base import SegmentationModel
from beltwatch.models.registry import build_model, encoder_weights_description
from beltwatch.training.config import TrainConfig, load_train_config
from beltwatch.training.losses import segmentation_loss

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainResult:
    best_metric: float
    best_epoch: int
    best_checkpoint: Path
    latest_checkpoint: Path
    global_step: int
    run_id: str


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - also seed third-party code using the legacy RNG
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    """Linear warm-up to 1, then cosine decay to 0 at ``total_steps``."""
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def git_commit() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, out)
    else:
        out[prefix] = value
    return out


@contextmanager
def _mlflow_run(config: TrainConfig) -> Iterator[str]:
    mlflow.set_tracking_uri(config.tracking.mlflow_uri)
    mlflow.set_experiment(config.tracking.experiment)
    with mlflow.start_run(run_name=config.run_name) as run:
        yield run.info.run_id


def train(config: TrainConfig, *, resume_from: Path | None = None) -> TrainResult:
    seed_everything(config.runtime.seed, config.runtime.deterministic)
    device = select_device(config.runtime.device)
    use_amp = config.runtime.amp and device.type == "cuda"

    data = load_data_config(config.data_config)
    name = data.source.name
    splits_csv = data.paths.splits_dir / f"{name}-splits.csv"
    images_csv = data.paths.manifests_dir / f"{name}-images.csv"
    split_manifest_id = json.loads((data.paths.splits_dir / f"{name}-splits.json").read_text())[
        "split_manifest_id"
    ]
    remap = load_source_remap(data.paths.manifests_dir / f"{name}-validation.json")
    train_samples = load_samples(splits_csv, images_csv, data.paths.raw_dir, "train")
    val_samples = load_samples(splits_csv, images_csv, data.paths.raw_dir, "val")
    if config.evaluation.max_val_images:
        val_samples = val_samples[: config.evaluation.max_val_images]

    train_set = SegmentationDataset(train_samples, remap, config.preprocess, config.augment)
    generator = torch.Generator().manual_seed(config.runtime.seed)
    loader = DataLoader(
        train_set,
        batch_size=config.optimization.batch_size,
        shuffle=True,
        drop_last=len(train_set) > config.optimization.batch_size,
        num_workers=config.runtime.num_workers,
        generator=generator,
        persistent_workers=config.runtime.num_workers > 0,
        pin_memory=device.type == "cuda",
    )

    model = build_model(config.model).to(device)
    if not isinstance(model, SegmentationModel):
        raise TypeError(f"{config.model.name} does not implement SegmentationModel")
    opt = config.optimization
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.encoder_parameters()), "lr": opt.encoder_lr},
            {"params": list(model.head_parameters()), "lr": opt.head_lr},
        ],
        weight_decay=opt.weight_decay,
    )
    steps_per_epoch = max(1, math.ceil(len(loader) / opt.accumulation_steps))
    total_steps = config.runtime.max_steps or opt.epochs * steps_per_epoch
    warmup_steps = round(opt.warmup_epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_factor(s, total_steps, warmup_steps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    class_weights = (
        torch.tensor(config.loss.class_weights, device=device)
        if config.loss.class_weights
        else None
    )

    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    latest_path, best_path = out_dir / "latest.pt", out_dir / "best.pt"
    start_epoch, global_step, best_metric, best_epoch, stale = 0, 0, -math.inf, -1, 0
    if resume_from is not None:
        state = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, global_step = state["epoch"] + 1, state["global_step"]
        best_metric, best_epoch, stale = state["best_metric"], state["best_epoch"], state["stale"]
        torch.set_rng_state(state["torch_rng"])
        generator.set_state(state["loader_rng"])
        log.info("resumed from %s at epoch %d, step %d", resume_from, start_epoch, global_step)

    lineage = {
        "git_commit": git_commit(),
        "split_manifest_id": split_manifest_id,
        "dataset_md5": data.source.md5,
        "preprocessing_version": PREPROCESSING_VERSION,
        "beltwatch_version": __version__,
        "encoder_weights": encoder_weights_description(config.model),
        "torch_version": str(torch.__version__),
        "device": str(device),
        "platform": platform.platform(),
    }

    with _mlflow_run(config) as run_id:
        mlflow.set_tags(lineage)
        mlflow.log_params(_flatten("", json.loads(config.model_dump_json()), {}))
        mlflow.log_params({"train_images": len(train_samples), "val_images": len(val_samples)})
        epoch_seconds: list[float] = []

        for epoch in range(start_epoch, opt.epochs):
            if global_step >= total_steps:
                break
            model.set_encoder_trainable(epoch >= opt.frozen_encoder_epochs)
            model.train()
            started = time.perf_counter()
            running, seen = 0.0, 0
            optimizer.zero_grad(set_to_none=True)
            for i, batch in enumerate(loader):
                image = batch["image"].to(device, non_blocking=True)
                target = batch["mask"].to(device, non_blocking=True)
                autocast = torch.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
                with autocast:
                    logits = model(image)
                    loss = segmentation_loss(
                        logits, target, config.loss.kind, config.loss.dice_weight, class_weights
                    )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite loss at epoch {epoch}, batch {i}")
                scaled = scaler.scale(loss / opt.accumulation_steps)
                scaled.backward()  # type: ignore[no-untyped-call]
                running += float(loss.detach())
                seen += 1
                last_batch = i + 1 == len(loader)
                if (i + 1) % opt.accumulation_steps == 0 or last_batch:
                    if opt.grad_clip_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1
                    if global_step % 10 == 0:
                        mlflow.log_metrics(
                            {
                                "train/loss": running / seen,
                                "lr/encoder": optimizer.param_groups[0]["lr"],
                                "lr/head": optimizer.param_groups[1]["lr"],
                            },
                            step=global_step,
                        )
                    if global_step >= total_steps:
                        break
            epoch_seconds.append(time.perf_counter() - started)
            mlflow.log_metrics(
                {
                    "train/epoch_loss": running / max(seen, 1),
                    "time/epoch_seconds": epoch_seconds[-1],
                },
                step=global_step,
            )
            log.info(
                "epoch %d: loss %.4f (%.0fs)", epoch, running / max(seen, 1), epoch_seconds[-1]
            )

            evaluate_now = (epoch + 1) % config.evaluation.every_epochs == 0
            finishing = epoch + 1 == opt.epochs or global_step >= total_steps
            if evaluate_now or finishing:
                predictor = NeuralPredictor(
                    model, config.preprocess, device, name=config.model.name
                )
                results = evaluate_images(predictor, val_samples, remap)
                metrics = metrics_from_confusion(np.stack([r.confusion for r in results]).sum(0))
                cov = coverage_errors([r.predicted for r in results], [r.truth for r in results])
                fg = metrics.foreground_macro_iou
                val_metrics = {
                    "val/foreground_macro_iou": fg,
                    "val/mean_iou": metrics.mean_iou,
                    "val/coverage_mae_pp": cov.total_mae_pp,
                    **{f"val/iou_{k}": v for k, v in metrics.iou.items() if not math.isnan(v)},
                }
                mlflow.log_metrics(val_metrics, step=global_step)
                log.info("epoch %d: val foreground macro IoU %.4f", epoch, fg)
                if not math.isnan(fg) and fg > best_metric:
                    best_metric, best_epoch, stale = fg, epoch, 0
                    save_inference_checkpoint(
                        best_path,
                        model,
                        config.model,
                        config.preprocess,
                        {
                            **lineage,
                            "run_id": run_id,
                            "run_name": config.run_name,
                            "epoch": epoch,
                            "global_step": global_step,
                            "val_foreground_macro_iou": fg,
                            "val_metrics": metrics.as_dict(),
                            "train_config": json.loads(config.model_dump_json()),
                        },
                    )
                else:
                    stale += 1
                model.train()

            patience = config.early_stopping_patience
            stopping = patience is not None and stale >= patience
            last_epoch = epoch + 1 == opt.epochs or global_step >= total_steps or stopping
            if (epoch + 1) % config.runtime.checkpoint_every_epochs == 0 or last_epoch:
                _save_latest(
                    latest_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    generator,
                    config,
                    epoch=epoch,
                    global_step=global_step,
                    best_metric=best_metric,
                    best_epoch=best_epoch,
                    stale=stale,
                )
            if stopping:
                log.info("early stopping after %d evaluations without improvement", stale)
                break

        if best_path.exists():
            digest = file_digest(best_path)
            mlflow.set_tags({"best_checkpoint_sha256": digest, "best_epoch": best_epoch})
            mlflow.log_metric("best/val_foreground_macro_iou", best_metric)
        if epoch_seconds:
            mlflow.log_metric("time/mean_epoch_seconds", float(np.mean(epoch_seconds)))
        (out_dir / "run.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "global_step": global_step,
                    **lineage,
                    "preprocess": asdict(config.preprocess),
                },
                indent=2,
            )
            + "\n"
        )

    return TrainResult(best_metric, best_epoch, best_path, latest_path, global_step, run_id)


def _save_latest(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    generator: torch.Generator,
    config: TrainConfig,
    **progress: Any,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "loader_rng": generator.get_state(),
            "config": json.loads(config.model_dump_json()),
            **progress,
        },
        path,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a BeltWatch segmentation model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="path to a latest.pt checkpoint")
    parser.add_argument("--max-steps", type=int, help="override runtime.max_steps")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-val-images", type=int, help="override evaluation.max_val_images")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_train_config(args.config)
    runtime = config.runtime.model_copy(
        update={k: v for k, v in {"max_steps": args.max_steps, "device": args.device}.items() if v}
    )
    evaluation = config.evaluation.model_copy(
        update={"max_val_images": args.max_val_images} if args.max_val_images else {}
    )
    config = config.model_copy(update={"runtime": runtime, "evaluation": evaluation})
    result = train(config, resume_from=args.resume)
    log.info(
        "best val foreground macro IoU %.4f at epoch %d (%s); MLflow run %s",
        result.best_metric,
        result.best_epoch,
        result.best_checkpoint,
        result.run_id,
    )
    return 0
