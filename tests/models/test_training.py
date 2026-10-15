import json
from pathlib import Path

import mlflow
import numpy as np
import pytest
import torch

from beltwatch.data.augment import AugmentConfig
from beltwatch.data.masks import load_mask
from beltwatch.evaluation.segmentation import confusion_matrix, metrics_from_confusion
from beltwatch.inference.neural import NeuralPredictor, atomic_torch_save
from beltwatch.inference.preprocessing import PreprocessConfig, load_rgb
from beltwatch.models.registry import ModelConfig
from beltwatch.training.config import (
    EvaluationConfig,
    OptimizationConfig,
    RuntimeConfig,
    TrackingConfig,
    TrainConfig,
    load_train_config,
)
from beltwatch.training.loop import train

REMAP = {0: 0, 1: 3, 2: 1, 3: 4, 4: 2}


def tiny_config(data_config: Path, tmp_path: Path, **optimization: object) -> TrainConfig:
    opt = {
        "epochs": 2,
        "batch_size": 2,
        "encoder_lr": 1e-3,
        "head_lr": 1e-3,
        "warmup_epochs": 0,
        "frozen_encoder_epochs": 1,
        **optimization,
    }
    return TrainConfig(
        run_name="tiny",
        data_config=data_config,
        output_dir=tmp_path / "out",
        model=ModelConfig(pretrained=False),
        preprocess=PreprocessConfig(long_side=96),
        augment=AugmentConfig(
            crop_size=64,
            scale_range=(1.0, 1.0),
            blur_prob=0,
            jpeg_prob=0,
            brightness=0.0,
            contrast=0.0,
        ),
        optimization=OptimizationConfig(**opt),  # type: ignore[arg-type]
        evaluation=EvaluationConfig(every_epochs=1),
        runtime=RuntimeConfig(num_workers=0, device="cpu", amp=False),
        tracking=TrackingConfig(mlflow_uri=f"sqlite:///{tmp_path}/mlflow.db", experiment="test"),
    )


def test_repo_unet_config_is_valid() -> None:
    config = load_train_config(Path("configs/unet.yaml"))
    assert config.optimization.encoder_lr < config.optimization.head_lr
    assert config.model.pretrained


def test_segformer_config_matches_unet_except_the_model() -> None:
    """Matched conditions: only the architecture (and output paths) may differ."""
    unet = load_train_config(Path("configs/unet.yaml")).model_dump()
    segformer = load_train_config(Path("configs/segformer_b0.yaml")).model_dump()
    for key in ("model", "run_name", "output_dir"):
        unet.pop(key), segformer.pop(key)
    assert unet == segformer


CLASS_WEIGHTS = (1.0, 2.8, 4.0, 10.0, 10.0)
EXPERIMENT_CHANGES = {
    "segformer-b0-ce-seed1": {("runtime", "seed"): 1},
    "segformer-b0-ce-seed2": {("runtime", "seed"): 2},
    "segformer-b0-ce-dice": {("loss", "kind"): "ce_dice"},  # dice_weight stays at its default 1
    "segformer-b0-ce-material": {("augment", "material_crop_prob"): 0.5},
    "segformer-b0-ce-classweights": {("loss", "class_weights"): CLASS_WEIGHTS},
    # Round 2: fresh seeds of the class-weight change (docs/training.md §6).
    "segformer-b0-ce-classweights-seed1": {
        ("loss", "class_weights"): CLASS_WEIGHTS,
        ("runtime", "seed"): 1,
    },
    "segformer-b0-ce-classweights-seed2": {
        ("loss", "class_weights"): CLASS_WEIGHTS,
        ("runtime", "seed"): 2,
    },
}


@pytest.mark.parametrize("name", sorted(EXPERIMENT_CHANGES))
def test_experiment_differs_from_the_segformer_baseline_only_as_declared(name: str) -> None:
    base = load_train_config(Path("configs/segformer_b0.yaml")).model_dump()
    experiment = load_train_config(Path(f"configs/experiments/{name}.yaml")).model_dump()

    assert experiment.pop("run_name") == name
    assert experiment.pop("output_dir") == Path(f"models/{name}")
    base.pop("run_name"), base.pop("output_dir")
    for (section, key), value in EXPERIMENT_CHANGES[name].items():
        assert experiment[section][key] == value != base[section][key]
        experiment[section][key] = base[section][key]
    assert experiment == base


def test_every_experiment_config_is_declared() -> None:
    names = {path.stem for path in Path("configs/experiments").glob("*.yaml")}
    assert names == set(EXPERIMENT_CHANGES)


def test_training_smoke_run_writes_checkpoints_and_tracks(
    tiny_data_config: Path, tmp_path: Path
) -> None:
    config = tiny_config(tiny_data_config, tmp_path)

    result = train(config)

    assert result.global_step == 4  # 4 train images / batch 2 x 2 epochs
    assert result.best_checkpoint.exists() and result.latest_checkpoint.exists()
    assert np.isfinite(result.best_metric)
    run = mlflow.get_run(result.run_id)
    assert run.data.tags["split_manifest_id"] == "f" * 64
    assert run.data.tags["preprocessing_version"] == "pp-1"
    assert "best_checkpoint_sha256" in run.data.tags
    assert run.data.params["optimization.head_lr"] == "0.001"
    assert "val/foreground_macro_iou" in run.data.metrics
    record = json.loads((config.output_dir / "run.json").read_text())
    assert record["run_id"] == result.run_id


def test_resume_continues_from_latest(tiny_data_config: Path, tmp_path: Path) -> None:
    first = train(tiny_config(tiny_data_config, tmp_path, epochs=1))
    assert first.global_step == 2

    resumed = train(
        tiny_config(tiny_data_config, tmp_path, epochs=2), resume_from=first.latest_checkpoint
    )

    assert resumed.global_step == 4


ACCELERATOR = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
)


@pytest.mark.skipif(ACCELERATOR is None, reason="needs a CUDA or MPS device")
def test_resume_on_an_accelerator(tiny_data_config: Path, tmp_path: Path) -> None:
    # RNG states must be restored from CPU tensors; CPU-only CI cannot catch this.
    def on_accelerator(epochs: int) -> TrainConfig:
        config = tiny_config(tiny_data_config, tmp_path, epochs=epochs)
        runtime = config.runtime.model_copy(update={"device": ACCELERATOR})
        return config.model_copy(update={"runtime": runtime})

    first = train(on_accelerator(1))
    resumed = train(on_accelerator(2), resume_from=first.latest_checkpoint)

    assert resumed.global_step == 4


def test_interrupted_checkpoint_write_keeps_previous_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "latest.pt"
    atomic_torch_save({"epoch": 1}, path)

    def die_mid_write(obj: object, f: Path) -> None:
        Path(f).write_bytes(b"trunc")
        raise RuntimeError("session disconnected")

    monkeypatch.setattr(torch, "save", die_mid_write)
    with pytest.raises(RuntimeError):
        atomic_torch_save({"epoch": 2}, path)

    monkeypatch.undo()
    assert torch.load(path, weights_only=True) == {"epoch": 1}


def test_checkpoint_round_trip_predicts_full_size(tiny_data_config: Path, tmp_path: Path) -> None:
    result = train(tiny_config(tiny_data_config, tmp_path, epochs=1))

    predictor = NeuralPredictor.from_checkpoint(result.best_checkpoint)
    proba = predictor.predict_proba(np.zeros((54, 96, 3), dtype=np.uint8))

    assert proba.shape == (5, 54, 96)
    assert np.allclose(proba.sum(0), 1.0, atol=1e-4)


def test_checkpoint_with_other_preprocessing_version_is_refused(
    tiny_data_config: Path, tmp_path: Path
) -> None:
    result = train(tiny_config(tiny_data_config, tmp_path, epochs=1))
    ckpt = torch.load(result.best_checkpoint, weights_only=True)
    ckpt["preprocessing_version"] = "pp-0"
    torch.save(ckpt, tmp_path / "old.pt")

    with pytest.raises(ValueError, match="preprocessing"):
        NeuralPredictor.from_checkpoint(tmp_path / "old.pt")


def test_tiny_set_overfit(tiny_data_config: Path, tmp_path: Path) -> None:
    """The network must be able to memorize a tiny dataset: catches label/loss wiring bugs."""
    torch.manual_seed(0)
    config = tiny_config(
        tiny_data_config,
        tmp_path,
        epochs=40,
        batch_size=4,
        encoder_lr=3e-3,
        head_lr=3e-3,
        frozen_encoder_epochs=0,
    )
    config = config.model_copy(
        update={
            # Full-frame "crops" so every pixel, including the small metal block, is seen.
            "augment": config.augment.__class__(
                crop_size=96,
                scale_range=(1.0, 1.0),
                hflip_prob=0.0,
                blur_prob=0,
                jpeg_prob=0,
                brightness=0.0,
                contrast=0.0,
            ),
            "evaluation": EvaluationConfig(every_epochs=40),
            "runtime": config.runtime.model_copy(update={"checkpoint_every_epochs": 40}),
        }
    )
    result = train(config)

    predictor = NeuralPredictor.from_checkpoint(result.best_checkpoint)
    raw = tiny_data_config.parent / "raw"
    conf = sum(
        confusion_matrix(
            predictor.predict_proba(load_rgb(raw / f"img/{i}.png")).argmax(0),
            load_mask(raw / f"msk/{i}.png", REMAP),
        )
        for i in range(4)
    )
    metrics = metrics_from_confusion(np.asarray(conf))
    assert metrics.iou["cardboard"] > 0.8
    assert metrics.iou["metal"] > 0.6
