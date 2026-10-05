"""Build models by name from configuration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict
from torch import nn

from beltwatch.labels import NUM_CLASSES
from beltwatch.models.unet import UNetResNet18


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: Literal["unet_resnet18"] = "unet_resnet18"
    pretrained: bool = True
    freeze_bn: bool = True


def build_model(config: ModelConfig, *, pretrained: bool | None = None) -> nn.Module:
    """Build a model; ``pretrained`` overrides the config (e.g. False when loading weights)."""
    use_pretrained = config.pretrained if pretrained is None else pretrained
    if config.name == "unet_resnet18":
        return UNetResNet18(NUM_CLASSES, pretrained=use_pretrained, freeze_bn=config.freeze_bn)
    raise ValueError(f"unknown model {config.name!r}")
