"""The interface the training loop needs from a segmentation model."""

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

import torch
from torch import nn


@runtime_checkable
class SegmentationModel(Protocol):
    """``forward(x)`` maps ``B x 3 x H x W`` images to ``B x C x H x W`` logits."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...

    def encoder_parameters(self) -> Iterator[nn.Parameter]: ...

    def head_parameters(self) -> Iterator[nn.Parameter]: ...

    def set_encoder_trainable(self, trainable: bool) -> None: ...
