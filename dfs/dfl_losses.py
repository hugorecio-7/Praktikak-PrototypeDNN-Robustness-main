"""Losses for distributional prototype-distance experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionFocalDistanceLoss(nn.Module):
    """Distribution focal loss for squared prototype-distance targets."""

    def __init__(
        self,
        num_bins: int = 32,
        d_min: float = 0.0,
        d_max: float = 100.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()

        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}")
        if d_max <= d_min:
            raise ValueError(f"d_max must be > d_min, got d_min={d_min}, d_max={d_max}")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                "reduction must be one of {'mean', 'sum', 'none'}, "
                f"got {reduction!r}"
            )

        self.num_bins = num_bins
        self.d_min = d_min
        self.d_max = d_max
        self.reduction = reduction

    def _validate_inputs(
        self, bin_logits: torch.Tensor, target_distance: torch.Tensor
    ) -> None:
        if bin_logits.dim() != 3:
            raise ValueError(
                "bin_logits must be 3D with shape [B, M, num_bins], "
                f"got {tuple(bin_logits.shape)}"
            )
        if target_distance.dim() != 2:
            raise ValueError(
                "target_distance must be 2D with shape [B, M], "
                f"got {tuple(target_distance.shape)}"
            )
        if bin_logits.shape[-1] != self.num_bins:
            raise ValueError(
                f"bin_logits last dimension must be num_bins={self.num_bins}, "
                f"got {bin_logits.shape[-1]}"
            )
        if bin_logits.shape[:2] != target_distance.shape:
            raise ValueError(
                "bin_logits leading dimensions must match target_distance, "
                f"got {tuple(bin_logits.shape[:2])} and {tuple(target_distance.shape)}"
            )
        if bin_logits.device != target_distance.device:
            raise ValueError(
                "bin_logits and target_distance must be on the same device, "
                f"got {bin_logits.device} and {target_distance.device}"
            )

    def forward(
        self,
        bin_logits: torch.Tensor,
        target_distance: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(bin_logits, target_distance)

        target_distance = target_distance.to(dtype=bin_logits.dtype)
        target_clamped = torch.clamp(target_distance, min=self.d_min, max=self.d_max)
        scaled = (
            (target_clamped - self.d_min)
            / (self.d_max - self.d_min)
            * (self.num_bins - 1)
        )

        left = torch.floor(scaled).long().clamp(min=0, max=self.num_bins - 1)
        right = (left + 1).clamp(max=self.num_bins - 1)

        w_right = scaled - left.to(dtype=scaled.dtype)
        last_bin = left == (self.num_bins - 1)
        w_right = torch.where(last_bin, torch.zeros_like(w_right), w_right)
        w_left = 1.0 - w_right

        flat_logits = bin_logits.reshape(-1, self.num_bins)
        flat_left = left.reshape(-1)
        flat_right = right.reshape(-1)

        loss_left = F.cross_entropy(flat_logits, flat_left, reduction="none")
        loss_right = F.cross_entropy(flat_logits, flat_right, reduction="none")

        flat_loss = (
            w_left.reshape(-1) * loss_left
            + w_right.reshape(-1) * loss_right
        )

        if self.reduction == "mean":
            return flat_loss.mean()
        if self.reduction == "sum":
            return flat_loss.sum()
        return flat_loss.reshape_as(target_distance)
