"""Neural-network predictor and the checkpoint format it loads.

A checkpoint carries everything needed to reproduce predictions: the model
configuration, the preprocessing configuration and version, the label map, and
the weights. Loading refuses a checkpoint whose preprocessing version differs
from the running code, so training and serving can never silently diverge.
"""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from beltwatch.inference.preprocessing import (
    PREPROCESSING_VERSION,
    PreprocessConfig,
    prepare_image,
    restore_probabilities,
)
from beltwatch.labels import CLASS_NAMES
from beltwatch.models.registry import ModelConfig, build_model

CHECKPOINT_FORMAT = "beltwatch-checkpoint-v1"


def save_inference_checkpoint(
    path: Path,
    model: nn.Module,
    model_config: ModelConfig,
    preprocess: PreprocessConfig,
    metadata: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": CHECKPOINT_FORMAT,
            "model_config": model_config.model_dump(),
            "preprocess": asdict(preprocess),
            "preprocessing_version": PREPROCESSING_VERSION,
            "label_map": {k: v for k, v in CLASS_NAMES.items()},
            # JSON round trip: only plain types, so weights_only loading works.
            "metadata": json.loads(json.dumps(metadata)),
            "state_dict": model.state_dict(),
        },
        path,
    )


class NeuralPredictor:
    def __init__(
        self,
        model: nn.Module,
        preprocess: PreprocessConfig,
        device: torch.device | str = "cpu",
        temperature: float = 1.0,
        name: str = "unet_resnet18",
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.preprocess = preprocess
        self.temperature = temperature
        self.name = name

    @torch.inference_mode()
    def predict_proba(self, rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        image, geometry = prepare_image(rgb, self.preprocess)
        logits = self.model(torch.from_numpy(image)[None].to(self.device))[0]
        probs = restore_probabilities(logits.float().cpu(), geometry, self.temperature)
        return probs.numpy().astype(np.float32)

    @classmethod
    def from_checkpoint(
        cls, path: Path, device: torch.device | str = "cpu", temperature: float = 1.0
    ) -> "NeuralPredictor":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if ckpt.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"{path}: not a {CHECKPOINT_FORMAT} file")
        if ckpt["preprocessing_version"] != PREPROCESSING_VERSION:
            raise ValueError(
                f"{path}: trained with preprocessing {ckpt['preprocessing_version']}, "
                f"code is {PREPROCESSING_VERSION}"
            )
        if {int(k): v for k, v in ckpt["label_map"].items()} != {
            k: v for k, v in CLASS_NAMES.items()
        }:
            raise ValueError(f"{path}: label map differs from beltwatch.labels")
        model_config = ModelConfig(**ckpt["model_config"])
        model = build_model(model_config, pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        p = ckpt["preprocess"]
        mean, std = p["mean"], p["std"]
        preprocess = PreprocessConfig(
            long_side=int(p["long_side"]),
            pad_multiple=int(p["pad_multiple"]),
            mean=(float(mean[0]), float(mean[1]), float(mean[2])),
            std=(float(std[0]), float(std[1]), float(std[2])),
        )
        return cls(model, preprocess, device, temperature, name=model_config.name)
