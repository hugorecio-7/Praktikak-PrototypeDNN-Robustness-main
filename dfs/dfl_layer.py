"""Distributional prototype-distance head for DFS/DFL experiments."""

from typing import Dict

import torch
import torch.nn as nn


class DistributionalPrototypeDistanceLayer(nn.Module):
    """Map each sample-prototype pair to a distance distribution.

    This layer is intentionally standalone. It does not replace the B30
    classifier distances yet; it only produces distributional distance
    summaries that can be wired into later experiments.
    """

    def __init__(
        self,
        latent_dim: int,
        num_bins: int = 32,
        d_min: float = 0.0,
        d_max: float = 100.0,
        hidden_dim: int = 64,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {latent_dim}")
        if num_bins < 2:
            raise ValueError(f"num_bins must be >= 2, got {num_bins}")
        if d_max <= d_min:
            raise ValueError(f"d_max must be > d_min, got d_min={d_min}, d_max={d_max}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if eps < 0:
            raise ValueError(f"eps must be >= 0, got {eps}")

        self.latent_dim = latent_dim
        self.num_bins = num_bins
        self.d_min = d_min
        self.d_max = d_max
        self.hidden_dim = hidden_dim
        self.eps = eps

        self.register_buffer("bin_centers", torch.linspace(d_min, d_max, num_bins))

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_bins),
        )

    def _validate_inputs(self, z: torch.Tensor, prototypes: torch.Tensor) -> None:
        if z.dim() != 2:
            raise ValueError(f"z must be 2D with shape [B, latent_dim], got {tuple(z.shape)}")
        if prototypes.dim() != 2:
            raise ValueError(
                "prototypes must be 2D with shape [M, latent_dim], "
                f"got {tuple(prototypes.shape)}"
            )
        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"z latent dimension must match latent_dim={self.latent_dim}, "
                f"got {z.shape[1]}"
            )
        if prototypes.shape[1] != self.latent_dim:
            raise ValueError(
                "prototype latent dimension must match "
                f"latent_dim={self.latent_dim}, got {prototypes.shape[1]}"
            )
        if prototypes.device != z.device:
            raise ValueError(
                f"z and prototypes must be on the same device, got {z.device} and "
                f"{prototypes.device}"
            )

    def forward(
        self,
        z: torch.Tensor,
        prototypes: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        self._validate_inputs(z, prototypes)

        prototypes = prototypes.to(dtype=z.dtype)
        batch_size, num_prototypes = z.shape[0], prototypes.shape[0]
        first_parameter = next(self.mlp.parameters())
        if first_parameter.device != z.device or first_parameter.dtype != z.dtype:
            self.mlp.to(device=z.device, dtype=z.dtype)

        diff = z[:, None, :] - prototypes[None, :, :]
        diff2 = diff.pow(2)

        flat_features = diff2.reshape(batch_size * num_prototypes, self.latent_dim)
        bin_logits = self.mlp(flat_features)
        bin_logits = bin_logits.reshape(batch_size, num_prototypes, self.num_bins)

        prob = torch.softmax(bin_logits, dim=-1)
        bin_centers = self.bin_centers.to(device=z.device, dtype=z.dtype)
        mu = torch.sum(prob * bin_centers, dim=-1)
        centered_bins = bin_centers.view(1, 1, -1) - mu.unsqueeze(-1)
        var = torch.sum(prob * centered_bins.pow(2), dim=-1)
        sigma = torch.sqrt(var + self.eps)

        out = {
            "bin_logits": bin_logits,
            "prob": prob,
            "mu": mu,
            "var": var,
            "sigma": sigma,
        }
        if return_features:
            out["diff2"] = diff2
        return out
