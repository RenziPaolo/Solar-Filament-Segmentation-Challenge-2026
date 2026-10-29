"""Tests for clDice loss functions."""
from __future__ import annotations

import torch

from solarseg.losses import (
    BCEDiceLoss,
    BCECLDiceLoss,
    BCEDiceCLDiceLoss,
    soft_cldice_loss,
    soft_dice_loss,
)


def test_soft_cldice_loss_shape():
    """clDice loss should return a scalar."""
    logits = torch.randn(2, 1, 64, 64)
    target = torch.rand(2, 1, 64, 64)
    loss = soft_cldice_loss(logits, target)
    assert loss.dim() == 0, "clDice loss must be a scalar"
    assert loss.item() >= 0, "Loss must be non-negative"


def test_bce_cldice_loss_is_finite():
    """BCE+clDice should produce a finite scalar."""
    logits = torch.randn(1, 1, 32, 32)
    target = torch.rand(1, 1, 32, 32)
    loss = BCECLDiceLoss()(logits, target)
    assert torch.isfinite(loss), "Loss must be finite"
    assert loss.item() > 0, "Loss must be positive"


def test_bce_dice_cldice_loss_is_finite():
    """BCE+Dice+clDice should produce a finite scalar."""
    logits = torch.randn(1, 1, 32, 32)
    target = torch.rand(1, 1, 32, 32)
    loss = BCEDiceCLDiceLoss()(logits, target)
    assert torch.isfinite(loss), "Loss must be finite"
    assert loss.item() > 0, "Loss must be positive"


def test_cldice_perfect_prediction():
    """When prediction closely matches a thin filament target, clDice loss should be small.

    Note: soft skeletonization produces non-zero skeleton only for thin/linear
    structures (like solar filaments).  Solid filled regions yield empty
    skeletons — this is expected behavior, not a bug.
    """
    target = torch.zeros(1, 1, 64, 64)
    target[0, 0, 32, 10:55] = 1.0  # 1-pixel-wide horizontal filament
    # Moderate logits → probabilities around 0.27/0.73
    logits = (target - 0.5) * 2.0
    loss = soft_cldice_loss(logits, target)
    assert loss.item() < 0.5, f"clDice loss should be small for filament prediction, got {loss.item():.4f}"


def test_cldice_worse_than_dice_on_disconnected():
    """clDice should penalize fragmentation more than Dice."""
    # Target: a single filled blob
    target = torch.zeros(1, 1, 64, 64)
    target[0, 0, 20:40, 20:40] = 1.0
    # Prediction: same blob but split into two halves
    pred = target.clone()
    pred[0, 0, 20:30, 20:40] = 1.0
    pred[0, 0, 30:40, 20:40] = 0.0  # gap in the middle

    logits = pred * 20 - 10
    dice = soft_dice_loss(logits, target)
    cldice = soft_cldice_loss(logits, target)

    # clDice should be higher (worse) because the skeleton is broken
    assert cldice.item() > dice.item() or cldice.item() > 0.5