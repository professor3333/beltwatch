"""SegFormer-B0: the transformer challenger.

A hierarchical Mix Transformer (MiT-B0) encoder with a lightweight all-MLP
decoder (Hugging Face Transformers). The encoder starts from NVIDIA's
ImageNet-pretrained ``nvidia/mit-b0`` weights, and the decoder plus a new
five-class head are trained from scratch. SegFormer predicts at 1/4 of the
input resolution, so logits are upsampled bilinearly to the input size here,
which keeps the same interface as the U-Net.

**License:** the pretrained MiT weights are released under NVIDIA's license
for noncommercial research and evaluation. Any release built from them
records this. See THIRD_PARTY_NOTICES.md.
"""

from collections.abc import Iterator

import torch
import torch.nn.functional as F
from torch import nn
from transformers import SegformerConfig, SegformerForSemanticSegmentation

from beltwatch.labels import CLASS_NAMES, NUM_CLASSES

PRETRAINED_ID = "nvidia/mit-b0"
ENCODER_WEIGHTS = f"huggingface {PRETRAINED_ID} (NVIDIA license: noncommercial research/evaluation)"


class SegFormerB0(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> None:
        super().__init__()
        id2label = {i: CLASS_NAMES[i] for i in range(num_classes)}
        label2id = {name: i for i, name in id2label.items()}
        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(
                PRETRAINED_ID, num_labels=num_classes, id2label=id2label, label2id=label2id
            )
        else:
            # The default configuration is exactly the MiT-B0 architecture, so
            # checkpoints load offline without downloading anything.
            config = SegformerConfig(id2label=id2label, label2id=label2id)
            if config.num_labels != num_classes:
                raise ValueError("SegFormer config did not take the label map")
            self.net = SegformerForSemanticSegmentation(config)  # type: ignore[no-untyped-call]

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.net.segformer.parameters()

    def head_parameters(self) -> Iterator[nn.Parameter]:
        yield from self.net.decode_head.parameters()

    def set_encoder_trainable(self, trainable: bool) -> None:
        for p in self.net.segformer.parameters():
            p.requires_grad_(trainable)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.net(pixel_values=x).logits
        return F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
