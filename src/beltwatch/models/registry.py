"""Build models by name from configuration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict
from torch import nn

from beltwatch.labels import NUM_CLASSES
from beltwatch.models.unet import UNetResNet18


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["unet_resnet18", "segformer_b0"] = "unet_resnet18"
    pretrained: bool = True
    freeze_bn: bool = True
    """U-Net only: keep the pretrained encoder's BatchNorm statistics frozen."""


def build_model(config: ModelConfig, *, pretrained: bool | None = None) -> nn.Module:
    """Build a model; ``pretrained`` overrides the config (e.g. False when loading weights)."""
    use_pretrained = config.pretrained if pretrained is None else pretrained
    if config.name == "unet_resnet18":
        return UNetResNet18(NUM_CLASSES, pretrained=use_pretrained, freeze_bn=config.freeze_bn)
    if config.name == "segformer_b0":
        from beltwatch.models.segformer import SegFormerB0

        return SegFormerB0(NUM_CLASSES, pretrained=use_pretrained)
    raise ValueError(f"unknown model {config.name!r}")


def encoder_weights_description(config: ModelConfig) -> str:
    """Provenance string for the pretrained weights a model starts from."""
    if not config.pretrained:
        return "none (random initialization)"
    if config.name == "segformer_b0":
        from beltwatch.models.segformer import ENCODER_WEIGHTS as SEGFORMER_WEIGHTS

        return SEGFORMER_WEIGHTS
    from beltwatch.models.unet import ENCODER_WEIGHTS as UNET_WEIGHTS

    return UNET_WEIGHTS
