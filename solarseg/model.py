"""A compact U-Net written from scratch (no pretrained weights or external downloads).

Also provides a ConvNeXt-UNet variant that swaps the custom encoder for a
torchvision ``convnext_tiny`` backbone while keeping the same decoder/forward
interface, so it plugs into the existing training and inference scripts.
"""
from __future__ import annotations

from torchvision.models import convnext_tiny

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        super().__init__(*layers)


class UNet(nn.Module):
    """U-Net with ``depth`` downsampling steps; widths base, 2*base, 4*base, ..."""

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 32, depth: int = 4, dropout: float = 0.1):
        super().__init__()
        widths = [base * 2**i for i in range(depth + 1)]
        self.encoders = nn.ModuleList(
            [ConvBlock(in_ch, widths[0])]
            + [ConvBlock(widths[i - 1], widths[i], dropout if i == depth else 0.0) for i in range(1, depth + 1)]
        )
        self.decoders = nn.ModuleList(
            [ConvBlock(widths[i + 1] + widths[i], widths[i]) for i in reversed(range(depth))]
        )
        self.head = nn.Conv2d(widths[0], out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for i, enc in enumerate(self.encoders):
            x = enc(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        x = skips.pop()
        for dec in self.decoders:
            skip = skips.pop()
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skip], dim=1))
        return self.head(x)


# ---------------------------------------------------------------------------
# ConvNeXt-UNet
# ---------------------------------------------------------------------------

class ConvNeXtUNet(nn.Module):
    """U-Net whose encoder is a torchvision ConvNeXt-tiny backbone.

    The ConvNeXt stem uses a 4× stride, so for a 1024×1024 input the
    four encoder levels produce features at 1/4, 1/8, 1/16, and 1/32
    resolution.  A lightweight ConvBlock decoder with skip connections
    mirrors the original U-Net's forward pass.

    Encoder channels (ConvNeXt-tiny): 96 → 192 → 384 → 768
    """

    ENC_CH = [96, 192, 384, 768]

    def __init__(self, in_ch: int = 1, out_ch: int = 1, base: int = 48):
        super().__init__()
        backbone = convnext_tiny(weights=None)

        # Adapt the stem for single-channel grayscale input (GONG H-alpha).
        backbone.features[0][0] = nn.Conv2d(in_ch, 96, kernel_size=4, stride=4)

        # Group the 8 feature modules into 4 encoder levels.
        # Each ConvNeXt stage downsamples by 2× except the first (after the stem).
        self.stem = nn.Sequential(backbone.features[0], backbone.features[1])   # 96  ch, 1/4
        self.enc1 = nn.Sequential(backbone.features[2], backbone.features[3])   # 192 ch, 1/8
        self.enc2 = nn.Sequential(backbone.features[4], backbone.features[5])  # 384 ch, 1/16
        self.enc3 = nn.Sequential(backbone.features[6], backbone.features[7]) # 768 ch, 1/32

        # Bottleneck (no extra downsampling)
        self.bottleneck = ConvBlock(768, 768, dropout=0.1)

        # Decoder: upsample → concat skip → ConvBlock
        # dec3: 1/32 → 1/16, concat(e2=384)
        # dec2: 1/16 → 1/8,  concat(e1=192)
        # dec1: 1/8 → 1/4,   concat(e0=96)
        # dec0: 1/4 → 1/1,   no skip (final upsampling)
        self.decoders = nn.ModuleList([
            ConvBlock(768 + 384, 384),                          # dec3
            ConvBlock(384 + 192, 192),                          # dec2
            ConvBlock(192 + 96, 96),                            # dec1
            ConvBlock(96, base),                                # dec0
        ])
        self.head = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.stem(x)       #  96 ch, 1/4
        e1 = self.enc1(e0)      # 192 ch, 1/8
        e2 = self.enc2(e1)      # 384 ch, 1/16
        e3 = self.enc3(e2)      # 768 ch, 1/32

        # Bottleneck
        b = self.bottleneck(e3)  # 768 ch, 1/32

        # Decoder with skip connections
        skips = [e2, e1, e0]     # from coarsest to finest
        for dec in self.decoders:
            skip = skips.pop(0) if skips else None
            target_h, target_w = skip.shape[-2:] if skip is not None else x.shape[-2:]
            b = F.interpolate(b, size=(target_h, target_w), mode="bilinear", align_corners=False)
            if skip is not None:
                b = dec(torch.cat([b, skip], dim=1))
            else:
                b = dec(b)
        return self.head(b)


def build_model(base: int = 32, depth: int = 4, encoder: str = "unet") -> nn.Module:
    """Build a segmentation model.

    Parameters
    ----------
    base : int
        Width multiplier for the original U-Net (ignored for ConvNeXt).
    depth : int
        Number of U-Net downsampling steps (ignored for ConvNeXt).
    encoder : str
        ``"unet"`` (default) → from-scratch U-Net; ``"convnext"`` → ConvNeXt-tiny encoder
        with a lightweight ConvBlock decoder.
    """
    if encoder == "convnext":
        return ConvNeXtUNet(in_ch=1, out_ch=1, base=base)
    return UNet(in_ch=1, out_ch=1, base=base, depth=depth)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
