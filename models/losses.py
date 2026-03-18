"""Loss functions for Phase 3 classifier training."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Binary focal loss on top of logits."""

    def __init__(
        self,
        *,
        alpha: float = 0.75,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.to(dtype=logits.dtype)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, probabilities, 1.0 - probabilities)
        focal_weight = self.alpha * torch.pow(1.0 - pt, self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


def build_loss(
    name: str,
    *,
    pos_weight: torch.Tensor | None = None,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> nn.Module:
    """Factory for supported binary classification losses."""
    normalized_name = name.lower()
    if normalized_name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if normalized_name == "focal":
        return FocalLoss(alpha=alpha, gamma=gamma)
    raise ValueError(f"Unsupported loss '{name}'. Expected 'bce' or 'focal'.")
