"""Inference-time Reconstruction-Risk Shield for B30/CAEModel_Balanced."""

from __future__ import annotations

import torch
import torch.nn as nn

try:
    from torchmetrics.functional.image import structural_similarity_index_measure
except ImportError as exc:  # pragma: no cover - import guard for local setups.
    raise ImportError(
        "ReconstructionRiskWrapper requires torchmetrics. Install it with "
        "`pip install torchmetrics` before using this wrapper."
    ) from exc


class ReconstructionRiskWrapper(nn.Module):
    """Add a structural reconstruction penalty to B30 prototype distances.

    The wrapped ``base_model`` is expected to be an already trained
    ``CAEModel_Balanced``. The prototype reconstructions are computed once at
    construction time and stored as a non-trainable buffer.
    """

    def __init__(
        self,
        base_model: nn.Module,
        gamma: float = 1.0,
        power: int = 1,
        use_shift: bool = False,
    ) -> None:
        super().__init__()

        required_attrs = (
            "encoder",
            "decoder",
            "prototype_layer",
            "fc",
            "in_channels_prototype",
        )
        missing = [name for name in required_attrs if not hasattr(base_model, name)]
        if missing:
            raise AttributeError(
                "base_model is missing required CAEModel_Balanced attributes: "
                + ", ".join(missing)
            )
        if not hasattr(base_model.prototype_layer, "prototype_distances"):
            raise AttributeError(
                "base_model.prototype_layer is missing 'prototype_distances'."
            )

        self.base_model = base_model
        self.gamma = float(gamma)
        self.power = int(power)
        self.use_shift = bool(use_shift)
        self.in_channels_prototype = base_model.in_channels_prototype

        if self.power < 1:
            raise ValueError(f"power must be >= 1, got {self.power}.")

        # The shield is an inference wrapper. Freezing parameters keeps PGD
        # gradients focused on the input while preserving differentiability.
        self.base_model.eval()
        for param in self.base_model.parameters():
            param.requires_grad_(False)

        proto_vectors = self.base_model.prototype_layer.prototype_distances.detach()
        device = proto_vectors.device
        dtype = proto_vectors.dtype

        with torch.no_grad():
            dummy = torch.randn(1, 1, 28, 28, device=device, dtype=dtype)
            dummy_latent = self.base_model.encoder(dummy)
            self.prototype_feature_shape = tuple(dummy_latent.shape[1:])
            flat_dim = dummy_latent.view(1, -1).shape[1]

            if proto_vectors.shape[1] != flat_dim:
                raise ValueError(
                    "Prototype vector dimension does not match encoder output: "
                    f"{proto_vectors.shape[1]} != {flat_dim}."
                )

            proto_latents = proto_vectors.view(-1, *self.prototype_feature_shape)
            prototype_reconstructions = self.base_model.decoder(proto_latents).detach()

        self.register_buffer(
            "prototype_reconstructions",
            prototype_reconstructions.requires_grad_(False),
        )
        self.last_d_latente: torch.Tensor | None = None
        self.last_d_final: torch.Tensor | None = None

    @property
    def n_prototypes(self) -> int:
        return int(self.prototype_reconstructions.shape[0])

    def _pairwise_ssim_for_shifted_x(
        self,
        x: torch.Tensor,
        prototype_pairs: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        n_prototypes = self.n_prototypes
        x_pairs = (
            x.unsqueeze(1)
            .expand(batch_size, n_prototypes, *x.shape[1:])
            .reshape(batch_size * n_prototypes, *x.shape[1:])
        )
        ssim_values = structural_similarity_index_measure(
            x_pairs,
            prototype_pairs,
            data_range=1.0,
            reduction="none",
        )
        return ssim_values.view(batch_size, n_prototypes)

    def _pairwise_ssim(self, x: torch.Tensor) -> torch.Tensor:
        """Return SSIM(x_i, prototype_j) as a [B, P] tensor."""
        if x.dim() != 4:
            raise ValueError(f"x must have shape [B, C, H, W], got {tuple(x.shape)}.")
        if x.shape[1:] != self.prototype_reconstructions.shape[1:]:
            raise ValueError(
                "Input and reconstructed prototypes must have matching image "
                f"shape, got {tuple(x.shape[1:])} and "
                f"{tuple(self.prototype_reconstructions.shape[1:])}."
            )

        batch_size = x.shape[0]
        n_prototypes = self.n_prototypes
        prototypes = self.prototype_reconstructions.to(device=x.device, dtype=x.dtype)
        prototype_pairs = (
            prototypes.unsqueeze(0)
            .expand(batch_size, n_prototypes, *prototypes.shape[1:])
            .reshape(batch_size * n_prototypes, *prototypes.shape[1:])
        )

        if not self.use_shift:
            return self._pairwise_ssim_for_shifted_x(x, prototype_pairs)

        shifts = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
        best_ssim = None
        for dx, dy in shifts:
            shifted_x = torch.roll(x, shifts=(dx, dy), dims=(2, 3))
            ssim_values = self._pairwise_ssim_for_shifted_x(
                shifted_x,
                prototype_pairs,
            )
            best_ssim = (
                ssim_values
                if best_ssim is None
                else torch.maximum(best_ssim, ssim_values)
            )

        return best_ssim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoder_out = self.base_model.encoder(x)
        self.base_model.feature_vectors = encoder_out

        z = encoder_out.view(-1, self.in_channels_prototype)
        d_latente = self.base_model.prototype_layer(z)

        ssim_matrix = self._pairwise_ssim(x)
        d_estructural = (1.0 - ssim_matrix) ** self.power
        d_final = d_latente + self.gamma * d_estructural

        self.last_d_latente = d_latente
        self.last_d_final = d_final

        return self.base_model.fc(d_final)
