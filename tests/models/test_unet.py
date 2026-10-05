import math

import pytest
import torch
from torch import nn

from beltwatch.labels import IGNORE_INDEX, NUM_CLASSES
from beltwatch.models.registry import ModelConfig, build_model
from beltwatch.models.unet import UNetResNet18
from beltwatch.training.loop import lr_factor
from beltwatch.training.losses import cross_entropy, foreground_dice_loss, segmentation_loss


@pytest.fixture(scope="module")
def model() -> UNetResNet18:
    return UNetResNet18(pretrained=False)


def test_forward_shape(model: UNetResNet18) -> None:
    out = model(torch.randn(2, 3, 64, 96))
    assert out.shape == (2, NUM_CLASSES, 64, 96)


def test_rejects_sizes_not_divisible_by_32(model: UNetResNet18) -> None:
    with pytest.raises(ValueError, match="divisible by 32"):
        model(torch.randn(1, 3, 50, 64))


def test_encoder_batchnorm_stays_frozen_in_train_mode(model: UNetResNet18) -> None:
    model.train()
    encoder_bn = [
        m for mod in model.encoder_modules() for m in mod.modules() if isinstance(m, nn.BatchNorm2d)
    ]
    assert encoder_bn
    assert all(not m.training for m in encoder_bn)
    assert all(not p.requires_grad for m in encoder_bn for p in m.parameters())
    assert model.classifier.training


def test_parameter_groups_partition_all_parameters(model: UNetResNet18) -> None:
    enc = {id(p) for p in model.encoder_parameters()}
    head = {id(p) for p in model.head_parameters()}
    assert enc and head
    assert not enc & head
    assert enc | head == {id(p) for p in model.parameters()}


def test_set_encoder_trainable_keeps_bn_frozen() -> None:
    m = UNetResNet18(pretrained=False)
    m.set_encoder_trainable(False)
    assert not any(p.requires_grad for p in m.encoder_parameters())
    m.set_encoder_trainable(True)
    convs = [
        p
        for mod in m.encoder_modules()
        for c in mod.modules()
        if isinstance(c, nn.Conv2d)
        for p in c.parameters()
    ]
    assert all(p.requires_grad for p in convs)
    bns = [
        p
        for mod in m.encoder_modules()
        for b in mod.modules()
        if isinstance(b, nn.BatchNorm2d)
        for p in b.parameters()
    ]
    assert not any(p.requires_grad for p in bns)


def test_build_model_from_config() -> None:
    assert isinstance(build_model(ModelConfig(pretrained=False)), UNetResNet18)


def test_cross_entropy_ignores_ignore_pixels() -> None:
    logits = torch.zeros(1, NUM_CLASSES, 1, 2)
    logits[0, 1, 0, 0] = 10.0
    target = torch.tensor([[[1, IGNORE_INDEX]]])
    assert cross_entropy(logits, target) < 1e-3


def test_dice_is_low_for_perfect_and_high_for_wrong_predictions() -> None:
    target = torch.tensor([[[1, 2], [0, IGNORE_INDEX]]])
    perfect = (
        torch.nn.functional.one_hot(target.clamp(0, 4), NUM_CLASSES).permute(0, 3, 1, 2) * 20.0
    )
    wrong = torch.roll(perfect, shifts=1, dims=1)
    assert foreground_dice_loss(perfect.float(), target) < 0.4  # smoothing keeps absent classes ~1
    assert foreground_dice_loss(wrong.float(), target) > foreground_dice_loss(
        perfect.float(), target
    )


def test_segmentation_loss_kinds() -> None:
    logits, target = torch.randn(1, NUM_CLASSES, 4, 4), torch.randint(0, NUM_CLASSES, (1, 4, 4))
    assert segmentation_loss(logits, target, "ce_dice") > segmentation_loss(logits, target, "ce")
    with pytest.raises(ValueError, match="unknown loss"):
        segmentation_loss(logits, target, "focal")


def test_lr_schedule_warms_up_then_decays() -> None:
    factors = [lr_factor(s, total_steps=100, warmup_steps=10) for s in range(100)]
    assert factors[0] == pytest.approx(0.1)
    assert factors[9] == pytest.approx(1.0)
    assert factors[10] == pytest.approx(1.0)
    assert factors[55] == pytest.approx(0.5, abs=0.02)
    assert factors[-1] < 0.01
    assert all(not math.isnan(f) for f in factors)
