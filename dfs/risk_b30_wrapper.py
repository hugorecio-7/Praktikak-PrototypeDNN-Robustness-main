"""Safe interception wrapper for B30 prototype-distance experiments."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class RiskAwareB30Wrapper(nn.Module):
    """Reference-based wrapper around an already-created B30 model.

    The wrapper reuses the original model submodules by reference to keep the
    deterministic path mathematically identical and avoid registering a second
    full copy of the base model.
    """

    VALID_MODES = {"deterministic", "mean", "risk"}

    def __init__(
        self,
        base_model: nn.Module,
        mode: str = "deterministic",
        beta: float = 0.0,
        dfl_layer: Optional[nn.Module] = None,
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
                "base_model is missing required B30 attributes: "
                + ", ".join(missing)
            )

        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode={mode!r}. Expected one of {sorted(self.VALID_MODES)}."
            )
        if mode in {"mean", "risk"} and dfl_layer is None:
            raise ValueError(
                f"mode={mode!r} requires a dfl_layer. Use mode='deterministic' "
                "to reproduce original B30 distances without DFL."
            )

        self.mode = mode
        self.beta = beta
        self.dfl_layer = dfl_layer

        self.encoder = base_model.encoder
        self.decoder = base_model.decoder
        self.prototype_layer = base_model.prototype_layer
        self.fc = base_model.fc
        self.in_channels_prototype = base_model.in_channels_prototype
        self.feature_vectors = getattr(base_model, "feature_vectors", None)

    def _encode_and_get_deterministic_distances(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoder_out = self.encoder(x)
        self.feature_vectors = encoder_out
        z = encoder_out.view(-1, self.in_channels_prototype)
        d_det = self.prototype_layer(z)
        return encoder_out, z, d_det

    def _get_prototypes(self) -> torch.Tensor:
        if not hasattr(self.prototype_layer, "prototype_distances"):
            raise AttributeError(
                "prototype_layer is missing expected attribute "
                "'prototype_distances'."
            )
        return self.prototype_layer.prototype_distances

    def compute_distance_outputs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        encoder_out, z, d_det = self._encode_and_get_deterministic_distances(x)
        outputs = {
            "encoder_out": encoder_out,
            "z": z,
            "d_det": d_det,
        }

        if self.mode == "deterministic":
            d_used = d_det
        else:
            prototypes = self._get_prototypes()
            dfl_out = self.dfl_layer(z, prototypes)
            mu = dfl_out["mu"]
            sigma = dfl_out["sigma"]

            if mu.shape != d_det.shape:
                raise ValueError(
                    f"DFL mu shape must match deterministic distances, got "
                    f"{tuple(mu.shape)} and {tuple(d_det.shape)}"
                )
            if sigma.shape != d_det.shape:
                raise ValueError(
                    f"DFL sigma shape must match deterministic distances, got "
                    f"{tuple(sigma.shape)} and {tuple(d_det.shape)}"
                )

            if self.mode == "mean":
                d_used = mu
            else:
                d_used = mu + self.beta * sigma

            outputs.update(
                {
                    "mu": mu,
                    "sigma": sigma,
                    "bin_logits": dfl_out["bin_logits"],
                    "prob": dfl_out["prob"],
                }
            )

        logits = self.fc(d_used)
        outputs["d_used"] = d_used
        outputs["logits"] = logits
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.compute_distance_outputs(x)
        return outputs["logits"]
