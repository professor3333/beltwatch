"""U-Net with an ImageNet-pretrained ResNet-18 encoder.

Encoder: torchvision's ResNet-18 stages (strides 2, 4, 8, 16, 32). Decoder: at
each scale, upsample, concatenate the encoder skip connection, then two 3x3
convolutions. The decoder uses GroupNorm, which behaves the same for any batch
size. The encoder's pretrained BatchNorm statistics are frozen
(:meth:`UNetResNet18.train` keeps them in eval mode), because batches of 2 to 4
crops would give noisy estimates.

Inputs must have height and width divisible by 32; the shared preprocessing
pads to that.
"""

from collections.abc import Iterator

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from beltwatch.labels import NUM_CLASSES

ENCODER_WEIGHTS = "torchvision ResNet18_Weights.IMAGENET1K_V1"


class _ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.ReLU(inplace=True),
        )


class _UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = _ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor | None) -> torch.Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        out: torch.Tensor = self.conv(x)
        return out


class UNetResNet18(nn.Module):
    def __init__(
        self, num_classes: int = NUM_CLASSES, pretrained: bool = True, freeze_bn: bool = True
    ) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2, 64
        self.pool = backbone.maxpool
        self.layer1, self.layer2 = backbone.layer1, backbone.layer2  # /4 64, /8 128
        self.layer3, self.layer4 = backbone.layer3, backbone.layer4  # /16 256, /32 512
        self.up4 = _UpBlock(512, 256, 256)
        self.up3 = _UpBlock(256, 128, 128)
        self.up2 = _UpBlock(128, 64, 64)
        self.up1 = _UpBlock(64, 64, 32)
        self.up0 = _UpBlock(32, 0, 16)
        self.classifier = nn.Conv2d(16, num_classes, 1)
        self.freeze_bn = freeze_bn
        if freeze_bn:
            for module in self.encoder_modules():
                for m in module.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.requires_grad_(False)

    def encoder_modules(self) -> list[nn.Module]:
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]

    def encoder_parameters(self) -> Iterator[nn.Parameter]:
        for module in self.encoder_modules():
            yield from module.parameters()

    def head_parameters(self) -> Iterator[nn.Parameter]:
        encoder = {id(p) for p in self.encoder_parameters()}
        return (p for p in self.parameters() if id(p) not in encoder)

    def set_encoder_trainable(self, trainable: bool) -> None:
        for module in self.encoder_modules():
            for m in module.modules():
                if isinstance(m, nn.BatchNorm2d) and self.freeze_bn:
                    continue
                for p in m.parameters(recurse=False):
                    p.requires_grad_(trainable)

    def train(self, mode: bool = True) -> "UNetResNet18":
        super().train(mode)
        if self.freeze_bn:
            for module in self.encoder_modules():
                for m in module.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 32 or x.shape[-2] % 32:
            raise ValueError(f"input size {tuple(x.shape[-2:])} must be divisible by 32")
        s1 = self.stem(x)
        s2 = self.layer1(self.pool(s1))
        s3 = self.layer2(s2)
        s4 = self.layer3(s3)
        s5 = self.layer4(s4)
        d = self.up4(s5, s4)
        d = self.up3(d, s3)
        d = self.up2(d, s2)
        d = self.up1(d, s1)
        d = self.up0(d, None)
        logits: torch.Tensor = self.classifier(d)
        return logits
