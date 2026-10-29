"""Tests for the ConvNeXt-UNet model variant."""
from __future__ import annotations

import torch

from solarseg.model import ConvNeXtUNet, build_model, count_parameters


def test_convnext_unet_output_shape():
    """Output must match input spatial dimensions."""
    model = ConvNeXtUNet(in_ch=1, out_ch=1, base=32)
    for h, w in [(256, 256), (512, 512), (1024, 1024)]:
        x = torch.randn(1, 1, h, w)
        out = model(x)
        assert out.shape == (1, 1, h, w), f"Expected {(1,1,h,w)}, got {out.shape}"


def test_convnext_unet_params_reasonable():
    """ConvNeXt-tiny encoder should have a substantial parameter count."""
    model = build_model(encoder="convnext")
    n = count_parameters(model)
    assert n > 10_000_000, f"ConvNeXt-UNet should have >10M params, has {n:,}"


def test_convnext_unet_backward_compat():
    """build_model(encoder='unet') should still return the original U-Net."""
    from solarseg.model import UNet
    model = build_model(encoder="unet", base=32, depth=4)
    assert isinstance(model, UNet)
    assert model(torch.randn(1, 1, 128, 128)).shape == (1, 1, 128, 128)


def test_convnext_unet_default_encoder():
    """Default encoder should be 'unet' for backward compatibility."""
    model = build_model(base=32, depth=4)
    from solarseg.model import UNet
    assert isinstance(model, UNet)


def test_convnext_unet_single_channel():
    """Model must accept 1-channel input (GONG H-alpha images)."""
    model = ConvNeXtUNet(in_ch=1, out_ch=1)
    x = torch.randn(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)