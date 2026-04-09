"""
metric_extractor.py
===================

Design rationale
----------------
Separation of concerns: this module answers ONLY "what is happening inside
the model?". metric_calculators.py answers "how bad is it?". Keeping these
two responsibilities apart means:

  1. Adding a new architecture requires touching only this file.
  2. metric_calculators.py has no import dependency on any model class;
     it is pure vectorial mathematics.
  3. Tests for metric correctness can be written against synthetic
     dictionaries without instantiating any model.

Wrapper stripping
-----------------
ProtoVAEWrapper and SENNWrapper (defined in run_test.py) renormalise inputs
from [0,1] to [-1,1] and expose only .forward(). We strip them here to access
sub-modules directly, applying the renormalisation ourselves where needed.

"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchgen import model

THETA: float = 1e-8   # Shared numerical stability constant

def _unwrap_base(model: nn.Module) -> nn.Module:
    return model.base if hasattr(model, "base") else model


def _is_protovae_model(model: nn.Module) -> bool:
    base = _unwrap_base(model)
    return (
        hasattr(model, "base")
        and hasattr(base, "prototype_class_identity")
        and hasattr(base, "prototype_vectors")
        and hasattr(base, "pred_class")
        and hasattr(base, "calc_sim_scores")
        and hasattr(base, "decoder")
    )


def _is_senn_model(model: nn.Module) -> bool:
    base = _unwrap_base(model)
    return (
        hasattr(model, "base")
        and hasattr(base, "conceptizer")
        and hasattr(base, "parameterizer")
        and hasattr(base, "aggregator")
    )

# =============================================================================
# Public API
# =============================================================================

def extract_internals(model: nn.Module, x: torch.Tensor) -> dict:
    """
    Dispatch to the correct architecture-specific extractor.

    All extractors run inside torch.no_grad() — calling this function never
    accumulates gradients, even when x.requires_grad is True.
    """
    with torch.no_grad():
        if _is_protovae_model(model):
            return _extract_protovae(model.base, x)
        elif _is_senn_model(model):
            return _extract_senn(model.base, x)
        else:
            return _extract_b30(model, x)


def get_proto_labels(model: nn.Module) -> torch.Tensor:
    """
    Return the class label assigned to each prototype as a (n_proto,) LongTensor.

    Used by calc_m_proto() to partition prototypes into correct (in P_y) and
    wrong (not in P_y) sets. Strategy differs per architecture.

    B30 (CAEModel_Balanced)
    -----------------------
    Uses IdentityLayer as its fc head. Its weight matrix has shape
    (n_classes, n_proto), initialised with -1 at [i, j] where prototype j
    belongs to class i, and fill_with (positive) elsewhere.
    argmin(dim=0) over the columns recovers the correct class per prototype.

    ProtoVAE
    --------
    Stores a buffer `prototype_class_identity` of shape (n_proto, n_classes)
    as a one-hot matrix. argmax(dim=1) gives the class per prototype.

    SENN
    ----
    Has no prototype layer. Raises ValueError.

    """

    if _is_senn_model(model):
        raise ValueError(
            "SENN has no prototype layer. Do not call get_proto_labels on SENN models."
        )

    if _is_protovae_model(model):
        return model.base.prototype_class_identity.detach().argmax(dim=1).cpu()

    return model.fc.linear.weight.detach().argmin(dim=0).cpu()


# =============================================================================
# Architecture-specific extractors (private)
# =============================================================================

def _extract_b30(model: nn.Module, x: torch.Tensor) -> dict:
    """
    PrototypeDL / B30 (CAEModel_Balanced) extractor.

    Real sub-module graph (verified from modules.py)
    -------------------------------------------------
    x  ──► EncoderLayer  ──► encoder_out  (B, n_classes, H', W')  [spatial]
                          └──► DecoderLayer ──► x_hat  (B, C, H, W)
    encoder_out.view(-1, in_channels_prototype) ──► z  (B, flat_dim)
    z  ──► PrototypeLayer.prototype_distances ──► d_proto  (B, n_proto)
    d_proto ──► IdentityLayer (model.fc) ──► logits  (B, n_classes)

    Two tensors from one encoder
    ----------------------------
    The encoder output serves two purposes with DIFFERENT shapes:
      - Spatial (B, n_classes, H', W') : fed into the decoder unchanged.
      - Flat    (B, flat_dim)          : fed into the prototype layer.
    We store the flat version as "z" because it is the representation that
    lives in prototype space and is what R_enc measures.

    Prototype distance computation
    ------------------------------
    The real PrototypeLayer calls list_of_distances(z, prototype_distances),
    which computes L2 distances. torch.cdist(z, protos, p=2) is mathematically
    identical and keeps the extractor self-contained.

    Why two forward passes?
    -----------------------
    We need z and x_hat via a partial pass, plus logits via the full forward.
    Running both under no_grad is safe and avoids reconstructing IdentityLayer
    logic manually.
    """
    _check_attr(model, "encoder",              "B30 (CAEModel_Balanced)")
    _check_attr(model, "decoder",              "B30 (CAEModel_Balanced)")
    _check_attr(model, "prototype_layer",      "B30 (CAEModel_Balanced)")
    _check_attr(model, "in_channels_prototype","B30 (CAEModel_Balanced)")

    encoder_out = model.encoder(x)                                    # (B, n_classes, H', W')
    z           = encoder_out.view(-1, model.in_channels_prototype)   # (B, flat_dim)
    x_hat       = model.decoder(encoder_out)                          # spatial input for decoder

    logits = model.forward(x)                                         # (B, n_classes)

    # prototype_distances is the Parameter inside PrototypeLayer: (n_proto, flat_dim).
    protos  = model.prototype_layer.prototype_distances               # (n_proto, flat_dim)
    d_proto = torch.cdist(z, protos, p=2)                             # (B, n_proto)

    return {
        "z":       z.detach(),
        "d_proto": d_proto.detach(),
        "x_hat":   x_hat.detach(),
        "logits":  logits.detach(),
    }


def _extract_protovae(base: nn.Module, x: torch.Tensor) -> dict:
    """
    ProtoVAE extractor.

    Real architecture (verified from model.py)
    -------------------------------------------
    base.features(x_norm)  →  conv_features  (B, latent*2)
      mu      = conv_features[:, :latent]
      log_var = conv_features[:, latent:].clamp(log(1e-8), -log(1e-8))
    z = mu   (deterministic — see below)
    base.decoder(z)              →  x_hat_norm  (B, C, H, W) in [-1,1]  via tanh
    base.prototype_vectors       →  (n_proto, latent) — the learnable prototype Parameter
    torch.cdist(z, protos, p=2)  →  d_proto (B, n_proto)  [raw L2 distances]
    base.last_layer(base.calc_sim_scores(z))  →  logits (B, n_classes)

    Why z = mu (no reparameterisation)?
    -------------------------------------
    ProtoVAE.pred_class() — the method used for evaluation — explicitly sets
    z = mu without sampling. We mirror that: a stochastic sample would make
    R_mu non-reproducible for the same input, corrupting the metric.

    Why raw distances and NOT sim_scores for d_proto?
    --------------------------------------------------
    calc_sim_scores computes log((d+1)/(d+ε)), a monotone DECREASING function
    of distance. calc_m_proto expects values where SMALLER = CLOSER = BETTER.
    Feeding similarities would invert the margin sign and give wrong results.
    We compute raw L2 distances directly.

    Latent dimension inference
    --------------------------
    We read `latent` from base.prototype_vectors.shape[1] instead of importing
    from ProtoVAE's settings.py, keeping this file dependency-free.

    Input normalisation
    -------------------
    ProtoVAE was trained on [-1,1]; we apply x_norm = x*2-1.
    Reconstructions (tanh output) are mapped back to [0,1] for R_dec.
    """
    x_norm = x * 2.0 - 1.0

    latent = base.prototype_vectors.shape[1]                  # inferred, e.g. 32

    conv_features = base.features(x_norm)                     # (B, latent*2)
    mu      = conv_features[:, :latent]                       # (B, latent)
    log_var = conv_features[:, latent:].clamp(
        float(np.log(1e-8)), float(-np.log(1e-8))
    )                                                         # (B, latent)
    sigma = torch.exp(0.5 * log_var)                          # posterior std-dev (B, latent)
    z     = mu                                                # deterministic

    x_hat_norm = base.decoder(z)                              # (B, C, H, W) tanh → [-1,1]
    x_hat      = (x_hat_norm + 1.0) / 2.0                    # → [0,1] for R_dec

    # Raw L2 distances — NOT similarities (see docstring).
    d_proto = torch.cdist(z, base.prototype_vectors, p=2)     # (B, n_proto)

    # Logits: replicate pred_class path.
    sim_scores = base.calc_sim_scores(z)                      # (B, n_proto) similarities
    logits     = base.last_layer(sim_scores)                  # (B, n_classes)

    return {
        "mu":      mu.detach(),
        "sigma":   sigma.detach(),
        "z":       z.detach(),         # canonical alias used by shared metric code
        "d_proto": d_proto.detach(),   # raw distances, not similarities
        "x_hat":   x_hat.detach(),
        "logits":  logits.detach(),
    }


def _extract_senn(base: nn.Module, x: torch.Tensor) -> dict:
    """
    SENN extractor.

    Real architecture (verified from senn.py + conceptizers.py)
    ------------------------------------------------------------
    base.conceptizer(x_norm)    →  (h_concept, recon_x)
      h_concept  : (B, n_concepts, 1)     — concept activations (ConvConceptizer)
      recon_x    : (B, C, H, W)           — reconstruction (discarded here)
    base.parameterizer(x_norm)  →  theta_param  (B, n_concepts, n_classes)
    base.aggregator(h_concept, theta_param) → logits (B, n_classes)
      SumAggregator applies log_softmax, so outputs are log-probabilities (≤ 0).
      argmax ordering is preserved — calc_m_pred still works correctly,
      but margin magnitudes differ from B30 which outputs raw logits.

    Critical fix vs. first implementation
    ----------------------------------------
    conceptizer.forward() returns a TUPLE (encoded, decoded) per the
    Conceptizer base class. The previous version assigned the whole tuple
    to h_concept and passed it to the aggregator — torch.bmm would crash.
    Always unpack explicitly.

    Why track h_concept AND theta_param separately?
    -----------------------------------------------
    H4 of the TFG predicts that the Parameterizer (theta) carries the
    dominant semantic load in SENN, analogous to the encoder in B30.
    If H4 is correct, freezing the Parameterizer (FT-2 for SENN) should
    drastically reduce robustness, and R_param will show the largest variance.
    Tracking both allows empirical falsification or support of H4.

    SENN has no prototype layer — "d_proto" is absent from the returned dict.
    calc_m_proto must NOT be called for SENN models.
    """
    x_norm = x * 2.0 - 1.0

    # Unpack the tuple — conceptizer.forward() always returns (encoded, decoded).
    h_concept, _recon_x = base.conceptizer(x_norm)   # (B, n_concepts, 1), (B, C, H, W)
    theta_param = base.parameterizer(x_norm)          # (B, n_concepts, n_classes)
    logits      = base.aggregator(h_concept, theta_param)  # (B, n_classes) log_softmax

    return {
        "h_concept":   h_concept.detach(),
        "theta_param": theta_param.detach(),
        "logits":      logits.detach(),
        # No "z", "d_proto", "x_hat" — SENN has none of these.
    }


# =============================================================================
# Internal helpers
# =============================================================================

def _check_attr(model: nn.Module, attr: str, arch_name: str) -> None:
    """Raise a descriptive AttributeError if a required sub-module is missing."""
    if not hasattr(model, attr):
        raise AttributeError(
            f"[metric_extractor] Model '{type(model).__name__}' is treated as "
            f"{arch_name} but is missing attribute '{attr}'. "
            f"Available: {[a for a in dir(model) if not a.startswith('_')]}"
        )
