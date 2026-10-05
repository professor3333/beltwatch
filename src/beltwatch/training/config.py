"""Validated training configuration (``configs/unet.yaml`` and friends)."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from beltwatch.data.augment import AugmentConfig
from beltwatch.inference.preprocessing import PreprocessConfig
from beltwatch.models.registry import ModelConfig


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class OptimizationConfig(_Strict):
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    accumulation_steps: int = Field(default=1, ge=1)
    encoder_lr: float = Field(gt=0)
    head_lr: float = Field(gt=0)
    weight_decay: float = Field(default=1e-4, ge=0)
    warmup_epochs: float = Field(default=1.0, ge=0)
    """Linear learning-rate warm-up, then cosine decay to zero."""
    frozen_encoder_epochs: int = Field(default=0, ge=0)
    """Epochs during which only the decoder and head train."""
    grad_clip_norm: float | None = Field(default=1.0, gt=0)


class LossConfig(_Strict):
    kind: Literal["ce", "ce_dice"] = "ce"
    dice_weight: float = Field(default=1.0, ge=0)
    class_weights: tuple[float, float, float, float, float] | None = None


class EvaluationConfig(_Strict):
    every_epochs: int = Field(default=1, ge=1)
    max_val_images: int | None = Field(default=None, ge=1)


class RuntimeConfig(_Strict):
    seed: int = 0
    num_workers: int = Field(default=4, ge=0)
    amp: bool = True
    """Mixed precision; only used on CUDA."""
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    deterministic: bool = False
    max_steps: int | None = Field(default=None, ge=1)
    """Cap on optimizer steps (smoke runs, step timing)."""
    checkpoint_every_epochs: int = Field(default=1, ge=1)
    """How often to write the resumable latest.pt (always written at the end)."""


class TrackingConfig(_Strict):
    mlflow_uri: str = "sqlite:///mlflow.db"
    experiment: str = "beltwatch"


class TrainConfig(_Strict):
    run_name: str
    data_config: Path = Path("configs/data.yaml")
    output_dir: Path
    model: ModelConfig = ModelConfig()
    preprocess: PreprocessConfig = PreprocessConfig()
    augment: AugmentConfig = AugmentConfig()
    optimization: OptimizationConfig
    loss: LossConfig = LossConfig()
    sampling: Literal["uniform"] = "uniform"
    evaluation: EvaluationConfig = EvaluationConfig()
    early_stopping_patience: int | None = Field(default=None, ge=1)
    runtime: RuntimeConfig = RuntimeConfig()
    tracking: TrackingConfig = TrackingConfig()


def load_train_config(path: Path) -> TrainConfig:
    with path.open(encoding="utf-8") as fh:
        return TrainConfig.model_validate(yaml.safe_load(fh))
