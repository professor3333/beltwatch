from pathlib import Path

import numpy as np
import pytest
import torch

from beltwatch.inference.neural import NeuralPredictor
from beltwatch.labels import NUM_CLASSES
from beltwatch.models.base import SegmentationModel
from beltwatch.models.registry import ModelConfig, build_model, encoder_weights_description
from beltwatch.models.segformer import SegFormerB0
from beltwatch.training.loop import train
from tests.models.test_training import tiny_config


@pytest.fixture(scope="module")
def model() -> SegFormerB0:
    return SegFormerB0(pretrained=False)


def test_forward_returns_full_resolution_logits(model: SegFormerB0) -> None:
    assert model(torch.randn(2, 3, 64, 96)).shape == (2, NUM_CLASSES, 64, 96)


def test_implements_the_training_interface(model: SegFormerB0) -> None:
    assert isinstance(model, SegmentationModel)
    enc = {id(p) for p in model.encoder_parameters()}
    head = {id(p) for p in model.head_parameters()}
    assert enc and head and not enc & head
    assert enc | head == {id(p) for p in model.parameters()}


def test_encoder_can_be_frozen_and_unfrozen() -> None:
    m = SegFormerB0(pretrained=False)
    m.set_encoder_trainable(False)
    assert not any(p.requires_grad for p in m.encoder_parameters())
    assert all(p.requires_grad for p in m.head_parameters())
    m.set_encoder_trainable(True)
    assert all(p.requires_grad for p in m.encoder_parameters())


def test_registry_and_provenance() -> None:
    config = ModelConfig(name="segformer_b0", pretrained=False)
    assert isinstance(build_model(config), SegFormerB0)
    assert "noncommercial" in encoder_weights_description(ModelConfig(name="segformer_b0"))
    assert encoder_weights_description(config).startswith("none")


def test_segformer_trains_and_round_trips_through_the_shared_predictor(
    tiny_data_config: Path, tmp_path: Path
) -> None:
    config = tiny_config(tiny_data_config, tmp_path, epochs=1)
    config = config.model_copy(update={"model": ModelConfig(name="segformer_b0", pretrained=False)})

    result = train(config)

    predictor = NeuralPredictor.from_checkpoint(result.best_checkpoint)
    assert predictor.name == "segformer_b0"
    proba = predictor.predict_proba(np.zeros((54, 96, 3), dtype=np.uint8))
    assert proba.shape == (NUM_CLASSES, 54, 96)
    assert np.allclose(proba.sum(0), 1.0, atol=1e-4)
