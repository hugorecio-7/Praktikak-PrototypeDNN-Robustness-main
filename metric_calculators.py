"""
metric_calculators.py
=====================

Mathematical conventions
------------------------
THETA : Small additive constant in all denominators. Prevents division by
        zero without meaningfully biasing the result (1e-8 << any realistic norm).

Sign convention for margins
---------------------------
  Positive margin  ↔  model is correct / semantically coherent.
  Negative margin  ↔  model has failed.

The *early failure* signature — the core of RQ1 — is the regime where
m_proto < 0 AND m_pred > 0: the semantic representation has already broken
while the classifier still outputs the right label. EarlyRate quantifies
how often this happens across the test set and all epsilons.

Note on SENN logits
-------------------
SENN's SumAggregator applies log_softmax, so its "logits" are log-probabilities
(all ≤ 0). The argmax ordering is preserved, so calc_m_pred still identifies
the correct class correctly. However, margin *magnitudes* will be on a different
scale than B30/ProtoVAE (raw logits). Interpret m_pred values within each
architecture; do not compare magnitudes directly across architectures.

"""

from __future__ import annotations

import numpy as np
import torch

THETA: float = 1e-8


# =============================================================================
# Per-sample metrics
# Called INSIDE the epsilon loop. Return (B,) numpy arrays.
# =============================================================================

def calc_m_proto(
    d_proto: torch.Tensor,
    labels: torch.Tensor,
    proto_labels: torch.Tensor,
) -> np.ndarray:
    """
    Prototype Margin  m̃_proto(x).

    Intuition
    ---------
    The numerator is the gap between the nearest WRONG prototype and the
    nearest CORRECT prototype. 

      +1 → far from wrong prototypes, close to correct ones (semantically coherent).
       0 → equidistant from both groups (semantically ambiguous).
      -1 → closer to a wrong prototype than to any correct one (semantic failure).

    A sample with m_proto < 0 has had its *interpretability* broken even if
    the classifier has not yet changed its output — the early failure signature.

    """
    d  = d_proto.cpu().numpy().astype(np.float64)   # (B, n_proto)
    y  = labels.cpu().numpy()                         # (B,)
    pl = proto_labels.cpu().numpy()                   # (n_proto,)

    B = d.shape[0]
    margins = np.empty(B, dtype=np.float32)

    for i in range(B):
        correct_mask = (pl == y[i])
        wrong_mask   = ~correct_mask

        d_correct = d[i, correct_mask].min() if correct_mask.any() else 0.0
        d_wrong   = d[i, wrong_mask  ].min() if wrong_mask.any()   else np.inf

        num = d_wrong   - d_correct
        den = d_wrong   + d_correct + THETA
        margins[i] = float(num / den)

    return margins


def calc_m_pred(logits: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """
    Prediction Margin  m̃_pred(x).

    Intuition
    ---------
    Positive → true class still wins the logit race → classifier correct.
    Negative → some other class has a higher logit → classifier wrong.
    
    The magnitude encodes how comfortably the classifier is correct (or wrong),
    normalised by the total logit energy (scale-invariant across architectures).

    """
    lg = logits.cpu().numpy().astype(np.float64)   # (B, n_classes)
    y  = labels.cpu().numpy()                        # (B,)

    B = lg.shape[0]
    margins = np.empty(B, dtype=np.float32)

    for i in range(B):
        logit_y     = lg[i, y[i]]
        competitors = np.delete(lg[i], y[i])
        max_comp    = competitors.max()
        norm        = np.linalg.norm(lg[i]) + THETA   # ||logits||_2
        margins[i]  = float((logit_y - max_comp) / norm)

    return margins


def calc_R_enc(z_clean: torch.Tensor, z_adv: torch.Tensor) -> np.ndarray:
    """
    Relative Encoder Drift  R_enc(x, ε).

    Intuition
    ---------
    How far the adversarial perturbation displaces the sample in latent space,
    *relative* to where the clean sample already sits. A latent code of norm 10
    that moves by 1 is far less disrupted than a code of norm 0.5 that moves by 1.

    For B30 : z is the flattened encoder bottleneck.
    For ProtoVAE : z = mu (posterior mean) → aliased as calc_R_mu.

    A value of 0 means no drift (perfect robustness in latent space).
    A value of 1 means the adversarial code moved as far from the clean code
    as the clean code is from the origin.

    """
    zc = z_clean.cpu().numpy().astype(np.float64)
    za = z_adv.cpu().numpy().astype(np.float64)

    num = np.linalg.norm(za - zc, axis=-1)           # (B,)
    den = np.linalg.norm(zc,      axis=-1) + THETA   # (B,)
    return (num / den).astype(np.float32)


def calc_R_dec(x_hat_clean: torch.Tensor, x_hat_adv: torch.Tensor) -> np.ndarray:
    """
    Decoder Reconstruction Drift  R_dec(x, ε).

    The reconstruction x̂ is the decoder's visual explanation. R_dec measures
    whether that explanation has changed under attack — a pixel-level proxy for
    interpretability degradation.

    """
    xc = x_hat_clean.cpu().numpy().reshape(x_hat_clean.shape[0], -1).astype(np.float64)
    xa = x_hat_adv.cpu().numpy().reshape(x_hat_adv.shape[0],   -1).astype(np.float64)
    return np.linalg.norm(xa - xc, axis=-1).astype(np.float32)   # (B,)


# --- ProtoVAE-specific (same formula as R_enc, different semantic role) ------

def calc_R_mu(mu_clean: torch.Tensor, mu_adv: torch.Tensor) -> np.ndarray:
    """
    Relative drift of the variational posterior mean  R_μ(x, ε).

    The mean μ is the "signal" in the latent code; its drift measures how much
    the *expected* representation changes under attack.

    A very robust ProtoVAE should keep R_μ near 0 even at large ε.
    """
    return calc_R_enc(mu_clean, mu_adv)


def calc_R_sigma(sigma_clean: torch.Tensor, sigma_adv: torch.Tensor) -> np.ndarray:
    """
    Relative drift of the variational posterior std-dev  R_σ(x, ε).

    Intuition
    ---------
    σ encodes the encoder's *uncertainty* about where to place a sample in
    latent space. Even if μ barely moves under attack (low R_μ), the encoder
    might inflate σ — hedging — signalling that the input is unusual even if
    the classification hasn't changed. High R_σ with low R_μ is a diagnostic
    for gradient masking: the encoder absorbs the attack through uncertainty
    rather than resisting it. Useful for the FT-E anomaly analysis.
    """
    return calc_R_enc(sigma_clean, sigma_adv)


# --- SENN-specific -----------------------------------------------------------

def calc_R_concept(h_clean: torch.Tensor, h_adv: torch.Tensor) -> np.ndarray:
    """
    Relative drift of the Conceptizer output  R_concept(x, ε).

    h is the concept vector: (B, n_concepts, 1). We flatten before computing
    the norm to treat the whole concept tensor as a single vector per sample.
    H4 predicts this will be *less* sensitive than R_param.
    """
    hc = h_clean.reshape(h_clean.shape[0], -1)
    ha = h_adv.reshape(h_adv.shape[0], -1)
    return calc_R_enc(hc, ha)


def calc_R_param(theta_clean: torch.Tensor, theta_adv: torch.Tensor) -> np.ndarray:
    """
    Relative drift of the Parameterizer output  R_param(x, ε).

    θ shape is (B, n_concepts, n_classes). We flatten before the norm.

    H4 of the TFG predicts theta carries the dominant semantic load in SENN,
    analogous to the encoder in B30. If H4 is correct, R_param should show the
    largest variance across ablation scenarios, and freezing the Parameterizer
    (FT-2 for SENN) should drastically reduce robustness.
    """
    tc = theta_clean.reshape(theta_clean.shape[0], -1)
    ta = theta_adv.reshape(theta_adv.shape[0], -1)
    return calc_R_enc(tc, ta)


def compute_empirical_robustness_interval(
    correct_matrix: np.ndarray,
    eps_grid: np.ndarray,
) -> dict:
    """
    Per-sample Empirical Robustness Interval  I_PGD(x) = [r_PGD−, r_PGD+].

    Definitions
    -----------
      r_PGD−(x) = max { ε ∈ E | ∀ε' ≤ ε,  ŷ(x_adv_ε') = y }
      r_PGD+(x) = min { ε ∈ E | ŷ(x_adv_ε) ≠ y }

        Returns
    -------
    dict:
        "r_minus"            : (N,) float32 — safe radius per sample.
                               Samples that never fail get r_minus = eps_grid[-1].
        "r_plus"             : (N,) float32 — first-failure epsilon.
                               Samples that never fail get r_plus = nan.
        "mean_r_minus_failing" : float — mean r_minus restricted to samples that
                               fail at some epsilon. Comparable to mean_r_plus.
        "mean_r_plus"        : float — mean r_plus over failing samples (nanmean).
        "frac_never_fail"    : float — fraction of samples that never fail in [0, max_eps].
        "mean_r_minus_all"   : float — mean r_minus over ALL samples (including
                               never-failing ones, which get r_minus = max_eps).
                               Useful for summarising overall robustness but NOT
                               directly comparable to mean_r_plus.
        "hist_r_minus"       : dict {"counts", "bin_edges"} for histogram plotting.

    Note on population consistency
    --------------------------------
    mean_r_minus_failing and mean_r_plus are computed over the SAME population
    (samples that fail at least once), so mean_r_minus_failing <= mean_r_plus
    is always guaranteed by construction.
    mean_r_minus_all is larger because it includes never-failing samples at max_eps,
    while mean_r_plus (nanmean) excludes them — making a direct comparison invalid.

    Intuition
    ---------
    r_PGD− is the largest ε up to which the sample remains correctly classified — the "robustness radius" under PGD attack. 
    r_PGD+ is the smallest ε at which the sample first becomes misclassified. 
    The interval [r_PGD−, r_PGD+] captures the "robustness window" for each sample.
    """
    N, E = correct_matrix.shape
    r_minus = np.full(N, eps_grid[-1], dtype=np.float32)
    r_plus  = np.full(N, np.nan,       dtype=np.float32)

    for i in range(N):
        row            = correct_matrix[i]
        first_fail_idx = int(np.argmax(~row))

        if row[first_fail_idx]:
            # All True — never fails across all epsilons.
            continue

        r_minus[i] = float(eps_grid[first_fail_idx - 1]) if first_fail_idx > 0 else 0.0
        r_plus[i]  = float(eps_grid[first_fail_idx])

    # Population masks.
    failing_mask  = ~np.isnan(r_plus)          # samples that fail at some epsilon
    frac_never    = float((~failing_mask).mean())

    # Consistent-population means: both computed over failing samples only.
    mean_r_minus_failing = (
        float(r_minus[failing_mask].mean()) if failing_mask.any() else float(eps_grid[-1])
    )
    mean_r_plus = (
        float(r_plus[failing_mask].mean()) if failing_mask.any() else float("nan")
    )

    # All-sample mean (inflated by never-failing samples, useful for Table 5.x).
    mean_r_minus_all = float(r_minus.mean())

    n_bins = min(E, 20)
    counts, edges = np.histogram(r_minus, bins=n_bins)

    return {
        "r_minus":               r_minus,
        "r_plus":                r_plus,
        "mean_r_minus_failing":  mean_r_minus_failing,
        "mean_r_plus":           mean_r_plus,
        "frac_never_fail":       frac_never,
        "mean_r_minus_all":      mean_r_minus_all,
        "hist_r_minus": {
            "counts":    counts.tolist(),
            "bin_edges": edges.tolist(),
        },
    }


def compute_early_rate(
    m_proto_matrix: np.ndarray,
    m_pred_matrix:  np.ndarray,
) -> float:
    """
    EarlyRate — fraction of test samples that exhibit early interpretability
    failure at any epsilon in the sweep.

    Interpretation of the condition
    --------------------------------
    m̃_proto < 0  →  sample drifted closer to a WRONG prototype than to a
                     correct one → interpretability has failed.
    m̃_pred  > 0  →  classifier still outputs the correct label.

    Their conjunction is the early failure signature (H3): the model's internal
    semantic representation is already broken, but the classification head has
    not "noticed" yet.

    Why max over ε (i.e., any())?
    ------------------------------
    The condition is transient: it occurs between the point where the latent
    space first breaks and the point where the classifier also fails. A fixed-ε
    EarlyRate would be threshold-sensitive and miss samples where the phenomenon
    occurs at a different epsilon. any() counts a sample if the condition holds
    at *at least one* ε — the most inclusive and robust definition.

    Parameters
    ----------
    m_proto_matrix : (N, E) float array — prototype margin per sample per ε.
    m_pred_matrix  : (N, E) float array — prediction margin per sample per ε.

    Returns
    -------
    float in [0, 1]. Higher = more samples exhibit early interpretability failure.
    """
    condition  = (m_proto_matrix < 0) & (m_pred_matrix > 0)   # (N, E) bool
    per_sample = condition.any(axis=1)                          # (N,) bool
    return float(per_sample.mean())


# =============================================================================
# Cohesion Ratio (Chapter 6.1)
# =============================================================================

def calc_cohesion_ratio(
    z: torch.Tensor,
    labels: torch.Tensor,
    proto_labels: torch.Tensor,
    prototypes: torch.Tensor,
) -> np.ndarray:
    """
    Intra/Inter-cluster cohesion ratio in latent space.

    Formulas
    --------
      D_intra(x) = min_{p ∈ P_y}  ||z(x) − p||_2
      D_inter(x) = min_{c ≠ y}  min_{p ∈ P_c}  ||z(x) − p||_2
      Ratio(x)   = D_intra(x) / ( D_inter(x) + θ )

    Purpose (Chapter 6.1)
    ----------------------
    Proves that fine-tuning does NOT destroy the geometric cluster structure
    despite observable metric-scale changes in the latent space. A ratio
    consistently below 1 means correct prototypes are always nearer than
    wrong ones — the semantic organisation is intact.

    Unlike m_proto (which normalises by the sum), the raw ratio here detects
    scale changes: if D_intra and D_inter both grow 10x but their ratio stays
    constant, the structure is preserved even if R_enc is high.

    Parameters
    ----------
    z           : (B, D)       — latent codes.
    labels      : (B,)         — ground-truth class indices.
    proto_labels: (n_proto,)   — class assignment of each prototype.
    prototypes  : (n_proto, D) — prototype embeddings.

    Returns
    -------
    (B,) float32 array. Values < 1 indicate correct cohesion.
    """
    z_np = z.cpu().numpy().astype(np.float64)
    y_np = labels.cpu().numpy()
    pl   = proto_labels.cpu().numpy()
    p_np = prototypes.cpu().numpy().astype(np.float64)

    B = z_np.shape[0]
    ratios = np.empty(B, dtype=np.float32)

    for i in range(B):
        correct_mask = (pl == y_np[i])
        wrong_mask   = ~correct_mask

        correct_protos = p_np[correct_mask]
        wrong_protos   = p_np[wrong_mask]

        d_intra = np.linalg.norm(z_np[i] - correct_protos, axis=-1).min()
        d_inter = (
            np.linalg.norm(z_np[i] - wrong_protos, axis=-1).min()
            if wrong_mask.any() else np.inf
        )

        ratios[i] = float(d_intra / (d_inter + THETA))

    return ratios

# =============================================================================
# SENN-specific global metrics (Chapter 7.2)
# =============================================================================

def compute_tau_param(R_param_matrix: np.ndarray, eps_idx: int = 1) -> float:
    """
    Empirical threshold tau_param for SENN explanation collapse detection.

    Definition
    ----------
      tau_param = percentile_95 { R_param(x_i, eps_min) | i = 1..N }

    where eps_min is the smallest NON-ZERO epsilon in the grid (eps_idx=1).

    Why not use eps=0?
    ------------------
    At eps=0 there is no perturbation, so R_param = 0 for every sample by
    construction — the 95th percentile would be trivially 0, useless as a
    threshold. The smallest non-zero epsilon represents minimum adversarial
    pressure, and the distribution of R_param values there characterises the
    baseline drift — the irreducible noise floor of the Parameterizer.

    Any R_param value that exceeds this 95th-percentile baseline at larger
    epsilon is considered an anomalous, attack-induced drift that constitutes
    explanation collapse.

    """
    if R_param_matrix.shape[1] <= eps_idx:
        raise ValueError(
            f"R_param_matrix has only {R_param_matrix.shape[1]} epsilon columns "
            f"but eps_idx={eps_idx} was requested. Need at least {eps_idx + 1} columns."
        )
    baseline_col = R_param_matrix[:, eps_idx]   # (N,) — R_param at eps_min
    return float(np.percentile(baseline_col, 95))


def compute_early_rate_senn(
    R_param_matrix: np.ndarray,
    m_pred_matrix:  np.ndarray,
    tau_param:      float,
) -> float:
    """
    EarlyRate_SENN — fraction of samples where SENN's explanation collapses
    before its classification output changes.

    Formula
    -------
      EarlyRate_SENN = (1/N) * sum_{i=1}^{N} max_{eps in E}
                         I[ R_param(x_i, eps) > tau_param  AND  m_pred(x_i, eps) > 0 ]

    Condition components
    --------------------
    R_param(x_i, eps) > tau_param
        The Parameterizer output has drifted beyond the empirical noise floor
        established on minimally-perturbed samples. The explanation has collapsed:
        the adversarial perturbation has significantly altered how SENN weighs its
        concepts.

    m_pred(x_i, eps) > 0
        The classifier still outputs the correct label.

    Their conjunction is the SENN analogue of the early failure signature (H4):
    the explanation mechanism breaks down while the final prediction has not yet
    changed. This is what H4 predicts — the Parameterizer is the semantically
    critical component in SENN, so it degrades first.

    Note: tau_param is model-specific. A FT-0 model will have a different
    tau_param than a FT-2 (frozen Parameterizer) model — that difference is
    itself diagnostic evidence for H4.

    """
    condition  = (R_param_matrix > tau_param) & (m_pred_matrix > 0)  # (N, E) bool
    per_sample = condition.any(axis=1)                                  # (N,) bool
    return float(per_sample.mean())