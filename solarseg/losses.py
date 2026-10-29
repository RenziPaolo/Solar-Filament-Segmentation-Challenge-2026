"""Loss for thin, sparse structures: weighted BCE, soft Dice, and clDice.

clDice (centerline Dice) explicitly rewards topological connectivity, which is
ideal for long, snake-like filaments.  See:
    - ``clDice: A Framework for Skeleton-Preserving Medical Image Segmentation``
      (Sudre et al., MICCAI 2020 / arXiv:1911.03374)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from skimage.morphology import skeletonize
except ImportError:  # pragma: no cover
    skeletonize = None


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum()
    denom = prob.sum() + target.sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


class BCEDiceLoss(nn.Module):
    def __init__(self, pos_weight: float = 3.0, dice_weight: float = 1.0):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        return bce + self.dice_weight * soft_dice_loss(logits, target)


@torch.no_grad()
def dice_score(prob: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred = (prob >= threshold).float()
    tgt = (target >= 0.5).float()
    inter = (pred * tgt).sum()
    denom = pred.sum() + tgt.sum()
    return float((2 * inter / denom).item()) if denom > 0 else 1.0


# ---------------------------------------------------------------------------
# clDice (centerline Dice) loss
# ---------------------------------------------------------------------------

def _soft_erode(img: torch.Tensor) -> torch.Tensor:
    """Soft morphological erosion via min-pooling (3×1 + 1×3 minimum)."""
    p1 = -F.max_pool2d(-img, (3, 1), 1, (1, 0))
    p2 = -F.max_pool2d(-img, (1, 3), 1, (0, 1))
    return torch.min(p1, p2)


def _soft_dilate(img: torch.Tensor) -> torch.Tensor:
    """Soft morphological dilation via max-pooling (3×3 maximum)."""
    return F.max_pool2d(img, (3, 3), 1, (1, 1))


def _soft_open(img: torch.Tensor) -> torch.Tensor:
    """Soft opening: erode then dilate."""
    return _soft_dilate(_soft_erode(img))


def _soft_skel(img: torch.Tensor, iter_: int = 3) -> torch.Tensor:
    """Soft iterative skeletonization.

    Repeatedly erodes the image and tracks the difference from the opened
    result, accumulating the skeleton.
    """
    img1 = _soft_open(img)
    skel = F.relu(img - img1)
    for _ in range(iter_):
        img = _soft_erode(img)
        img1 = _soft_open(img)
        delta = F.relu(img - img1)
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_loss(logits: torch.Tensor, target: torch.Tensor, iter_: int = 3) -> torch.Tensor:
    """Soft clDice loss — rewards topologically correct connectivity.

    The loss is 1 - clDice, where clDice is the geometric mean of skeleton
    precision and skeleton sensitivity.  Uses soft iterative skeletonization
    for the prediction (probability map) and ``skimage.morphology.skeletonize``
    for the ground-truth target, matching the original clDice implementation.
    """
    prob = torch.sigmoid(logits)
    skel_pred = _soft_skel(prob, iter_)

    # Skeletonize each ground-truth plane with skimage (handles binary/soft correctly)
    target_np = target.detach().cpu().numpy()
    skel_np = np.zeros_like(target_np, dtype=np.float32)
    for b in range(target_np.shape[0]):
        for c in range(target_np.shape[1]):
            binary = target_np[b, c] > 0.5
            if binary.any():
                skel_np[b, c] = skeletonize(binary).astype(np.float32) if skeletonize else binary.astype(np.float32)
    skel_target = torch.from_numpy(skel_np).to(target.device)

    tprec = (skel_pred * target).sum() / (skel_pred.sum() + 1e-6)
    tsens = (skel_target * prob).sum() / (skel_target.sum() + 1e-6)
    cl_dice = 2 * tprec * tsens / (tprec + tsens + 1e-6)
    return 1.0 - cl_dice


class BCECLDiceLoss(nn.Module):
    """BCE + soft clDice — drop-in replacement for BCEDiceLoss."""

    def __init__(self, pos_weight: float = 3.0, cldice_weight: float = 1.0, skel_iter: int = 3):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.cldice_weight = cldice_weight
        self.skel_iter = skel_iter

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        cldice = soft_cldice_loss(logits, target, iter_=self.skel_iter)
        return bce + self.cldice_weight * cldice


class BCEDiceCLDiceLoss(nn.Module):
    """BCE + Dice + clDice — combines pixel-level and topology-level signals."""

    def __init__(self, pos_weight: float = 3.0, dice_weight: float = 0.5,
                 cldice_weight: float = 0.5, skel_iter: int = 3):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.dice_weight = dice_weight
        self.cldice_weight = cldice_weight
        self.skel_iter = skel_iter

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=self.pos_weight)
        dice = soft_dice_loss(logits, target)
        cldice = soft_cldice_loss(logits, target, iter_=self.skel_iter)
        return bce + self.dice_weight * dice + self.cldice_weight * cldice
