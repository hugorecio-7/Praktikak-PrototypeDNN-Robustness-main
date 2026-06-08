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
while the classifier still outputs the right label. ICR (Interpretability
Collapse Rate) quantifies how often this happens across the test set and all
epsilons.

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


# =============================================================================
# Adversarial latent trajectory metrics
# =============================================================================

def compute_pgd_final_shift(
    z_clean: np.ndarray,
    z_pgd_final: np.ndarray,
) -> np.ndarray:
    """
    Per-sample final PGD displacement in the original latent space.

    Returns ||z_pgd_final - z_clean||_2 for each sample.
    """
    zc = np.asarray(z_clean, dtype=np.float64)
    zp = np.asarray(z_pgd_final, dtype=np.float64)
    return np.linalg.norm(zp - zc, axis=-1).astype(np.float32)


def compute_autoattack_shift(
    z_clean: np.ndarray,
    z_auto: np.ndarray,
) -> np.ndarray:
    """
    Per-sample AutoAttack final displacement in the original latent space.

    Returns ||z_auto - z_clean||_2 for each sample.
    """
    zc = np.asarray(z_clean, dtype=np.float64)
    za = np.asarray(z_auto, dtype=np.float64)
    return np.linalg.norm(za - zc, axis=-1).astype(np.float32)


def compute_pgd_path_length(
    z_pgd_trajectory: list[np.ndarray] | np.ndarray,
) -> np.ndarray:
    """
    Per-sample PGD latent trajectory path length.

    z_pgd_trajectory may be a list of T arrays shaped (B, D), or one array
    shaped (T, B, D). The returned value is
    sum_t ||z_pgd_t - z_pgd_{t-1}||_2 for each sample.
    """
    traj = np.asarray(z_pgd_trajectory, dtype=np.float64)
    if traj.ndim < 3:
        raise ValueError(
            "z_pgd_trajectory must have shape (T, B, D) or be a list of "
            "(B, D) arrays."
        )
    step_lengths = np.linalg.norm(np.diff(traj, axis=0), axis=-1)
    return step_lengths.sum(axis=0).astype(np.float32)


def compute_pgd_path_efficiency(
    z_clean: np.ndarray,
    z_pgd_final: np.ndarray,
    z_pgd_trajectory: list[np.ndarray] | np.ndarray,
) -> np.ndarray:
    """
    Per-sample PGD path efficiency in latent space.

    Efficiency = ||z_pgd_final - z_clean||_2 / (path_length + theta_traj).
    Values closer to 1 indicate a direct path; lower values indicate a more
    curved trajectory. theta_traj is only for numerical stability.
    """
    theta_traj = 1e-12
    final_shift = compute_pgd_final_shift(z_clean, z_pgd_final).astype(np.float64)
    path_length = compute_pgd_path_length(z_pgd_trajectory).astype(np.float64)
    efficiency = final_shift / (path_length + theta_traj)
    return efficiency.astype(np.float32)


def _as_proto_labels_tensor(
    proto_labels: torch.Tensor | list[int] | np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Convert prototype labels to a 1D tensor on the requested device."""
    if torch.is_tensor(proto_labels):
        labels = proto_labels.to(device=device)
    else:
        labels = torch.as_tensor(proto_labels, device=device)
    return labels.long().reshape(-1)


def _correct_wrong_min_distances(
    distance_matrix: torch.Tensor,
    y: torch.Tensor,
    proto_labels: torch.Tensor | list[int] | np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return nearest correct-class and wrong-class prototype distances."""
    if distance_matrix.dim() != 2:
        raise ValueError(
            "distance_matrix must have shape [B, M], "
            f"got {tuple(distance_matrix.shape)}"
        )
    if y.dim() != 1:
        raise ValueError(f"y must have shape [B], got {tuple(y.shape)}")
    if y.shape[0] != distance_matrix.shape[0]:
        raise ValueError(
            "y length must match distance_matrix batch size, "
            f"got {y.shape[0]} and {distance_matrix.shape[0]}"
        )

    labels = _as_proto_labels_tensor(proto_labels, distance_matrix.device)
    if labels.numel() != distance_matrix.shape[1]:
        raise ValueError(
            "proto_labels length must match number of prototypes, "
            f"got {labels.numel()} and {distance_matrix.shape[1]}"
        )

    y = y.to(device=distance_matrix.device, dtype=labels.dtype)
    correct_mask = labels.unsqueeze(0).eq(y.unsqueeze(1))
    wrong_mask = ~correct_mask

    if not correct_mask.any(dim=1).all():
        missing = torch.nonzero(~correct_mask.any(dim=1), as_tuple=False).flatten()
        raise ValueError(
            "At least one sample has no correct-class prototypes. "
            f"Sample indices: {missing[:10].detach().cpu().tolist()}"
        )
    if not wrong_mask.any(dim=1).all():
        missing = torch.nonzero(~wrong_mask.any(dim=1), as_tuple=False).flatten()
        raise ValueError(
            "At least one sample has no wrong-class prototypes. "
            f"Sample indices: {missing[:10].detach().cpu().tolist()}"
        )

    inf = torch.tensor(float("inf"), device=distance_matrix.device, dtype=distance_matrix.dtype)
    correct_distances = torch.where(correct_mask, distance_matrix, inf)
    wrong_distances = torch.where(wrong_mask, distance_matrix, inf)
    correct_min = correct_distances.min(dim=1).values
    wrong_min = wrong_distances.min(dim=1).values
    return correct_min, wrong_min


def calc_m_proto_from_distances(
    distance_matrix: torch.Tensor,
    y: torch.Tensor,
    proto_labels: torch.Tensor | list[int] | np.ndarray,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Compute prototype margins directly from a [B, M] distance matrix.

    Lower distances are better. For each sample, the margin is

        (wrong_min - correct_min) / (wrong_min + correct_min + eps)

    where correct_min is the nearest prototype with label y and wrong_min is the
    nearest prototype with any other label. Positive margins mean the nearest
    correct-class prototype is closer than the nearest wrong-class prototype.
    """
    correct_min, wrong_min = _correct_wrong_min_distances(
        distance_matrix=distance_matrix,
        y=y,
        proto_labels=proto_labels,
    )
    return (wrong_min - correct_min) / (wrong_min + correct_min + eps)


def calc_artificial_prototype_match_suppression(
    mu_distances: torch.Tensor,
    used_distances: torch.Tensor,
    y: torch.Tensor,
    proto_labels: torch.Tensor | list[int] | np.ndarray,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """
    Diagnose corrected and damaged prototype matches by risk-aware distances.

    A false artificial match under ``mu`` occurs when the nearest wrong-class
    prototype under ``mu`` is closer than the nearest correct-class prototype.
    It is corrected if the risk-aware distances reverse that ordering. The harm
    diagnostic marks the opposite case: ``mu`` preferred the correct prototype,
    but the used distance prefers the wrong prototype.
    """
    del eps  # kept for signature symmetry and future numerical variants
    if mu_distances.shape != used_distances.shape:
        raise ValueError(
            "mu_distances and used_distances must have the same shape, "
            f"got {tuple(mu_distances.shape)} and {tuple(used_distances.shape)}"
        )

    correct_mu, wrong_mu = _correct_wrong_min_distances(
        distance_matrix=mu_distances,
        y=y,
        proto_labels=proto_labels,
    )
    correct_used, wrong_used = _correct_wrong_min_distances(
        distance_matrix=used_distances,
        y=y,
        proto_labels=proto_labels,
    )

    false_match_mu = wrong_mu < correct_mu
    latent_correct_match = correct_mu < wrong_mu
    corrected_match = false_match_mu & (correct_used < wrong_used)
    risk_harm_match = latent_correct_match & (wrong_used < correct_used)

    return {
        "false_match_mu": false_match_mu,
        "latent_correct_match": latent_correct_match,
        "corrected_match": corrected_match,
        # Backward-compatible alias for older DFS/analysis code.
        "suppressed_match": corrected_match,
        "risk_harm_match": risk_harm_match,
        "correct_mu": correct_mu,
        "wrong_mu": wrong_mu,
        "correct_used": correct_used,
        "wrong_used": wrong_used,
    }


def calc_corrected_prototype_match_diagnostics(
    mu_distances: torch.Tensor,
    used_distances: torch.Tensor,
    y: torch.Tensor,
    proto_labels: torch.Tensor | list[int] | np.ndarray,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Return corrected-match and harm diagnostics for CMR/Harm Rate."""
    return calc_artificial_prototype_match_suppression(
        mu_distances=mu_distances,
        used_distances=used_distances,
        y=y,
        proto_labels=proto_labels,
        eps=eps,
    )


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


def _pgd_linf_attack_per_sample_eps(
    model: torch.nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    eps: torch.Tensor,
    iters: int,
    alpha: float,
    random_start: bool,
) -> torch.Tensor:
    """
    PGD-Linf attack with a different epsilon for each sample in the batch.

    This mirrors PGDLInf_attack's default CE objective and fixed alpha, but it
    accepts a vector of epsilons so the r-PGD binary search can stay batched.
    """
    device = batch_x.device
    eps = eps.to(device=device, dtype=batch_x.dtype).view(-1, *([1] * (batch_x.ndim - 1)))

    ori_images = batch_x.clone().detach()
    perturbed_batch_x = ori_images.clone().detach()

    if random_start:
        perturbed_batch_x = perturbed_batch_x + torch.empty_like(perturbed_batch_x).uniform_(-1.0, 1.0) * eps
        perturbed_batch_x = torch.clamp(perturbed_batch_x, min=0, max=1).detach()

    loss_function = torch.nn.CrossEntropyLoss()

    for _ in range(int(iters)):
        perturbed_batch_x.requires_grad_(True)
        logits = model(perturbed_batch_x)
        loss = loss_function(logits, batch_y)
        gradients = torch.autograd.grad(loss, perturbed_batch_x)[0]

        perturbed_batch_x = perturbed_batch_x.detach() + float(alpha) * torch.sign(gradients)
        delta = torch.clamp(perturbed_batch_x - ori_images, min=-eps, max=eps)
        perturbed_batch_x = torch.clamp(ori_images + delta, min=0, max=1).detach()

    return perturbed_batch_x


def _predict_correct(
    model: torch.nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        logits = model(batch_x)
        preds = torch.argmax(logits, dim=1)
    return (preds == batch_y).detach().cpu().numpy().astype(bool)


def _coarse_correct_from_model(
    model: torch.nn.Module,
    data_loader,
    eps_grid: np.ndarray,
    pgd_iters: int,
    pgd_alpha: float,
    pgd_random_start: bool,
    verbose: bool,
) -> np.ndarray:
    cols: list[list[np.ndarray]] = [[] for _ in range(len(eps_grid))]
    device = next(model.parameters()).device

    for batch_idx, (batch_x, batch_y) in enumerate(data_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        cols[0].append(_predict_correct(model, batch_x, batch_y))

        for ide in range(1, len(eps_grid)):
            eps_value = float(eps_grid[ide])
            eps_batch = torch.full((batch_x.shape[0],), eps_value, device=device, dtype=batch_x.dtype)
            x_adv = _pgd_linf_attack_per_sample_eps(
                model=model,
                batch_x=batch_x,
                batch_y=batch_y,
                eps=eps_batch,
                iters=pgd_iters,
                alpha=pgd_alpha,
                random_start=pgd_random_start,
            )
            cols[ide].append(_predict_correct(model, x_adv, batch_y))

        if verbose:
            print(f"    coarse r-PGD grid batch {batch_idx + 1}/{len(data_loader)}")

    return np.stack([np.concatenate(col, axis=0) for col in cols], axis=1)


def _summarize_r_pgd(
    r_pgd: np.ndarray,
    failing_mask: np.ndarray,
    max_eps: float,
    clean_correct_mask: np.ndarray,
    n_bins: int = 20,
) -> dict:
    r_pgd = r_pgd.astype(np.float32)
    clean_correct_mask = clean_correct_mask.astype(bool)
    valid_mask = clean_correct_mask & np.isfinite(r_pgd)
    valid_failing_mask = valid_mask & failing_mask.astype(bool)

    valid_r_pgd = r_pgd[valid_mask]
    mean_all = float(valid_r_pgd.mean()) if valid_r_pgd.size else float("nan")
    mean_failing = (
        float(r_pgd[valid_failing_mask].mean())
        if valid_failing_mask.any()
        else float(max_eps)
    )
    min_r_pgd = float(valid_r_pgd.min()) if valid_r_pgd.size else float("nan")
    counts, edges = np.histogram(valid_r_pgd, bins=n_bins, range=(0.0, float(max_eps)))

    return {
        "mean_r_pgd": mean_all,
        "mean_r_pgd_failing": mean_failing,
        "min_r_pgd": min_r_pgd,
        "frac_never_fail": (
            float((valid_mask & ~failing_mask.astype(bool)).sum() / valid_mask.sum())
            if valid_mask.any()
            else float("nan")
        ),
        "n_total": int(r_pgd.size),
        "n_clean_correct": int(valid_mask.sum()),
        "n_excluded_clean_incorrect": int((~clean_correct_mask).sum()),
        "frac_clean_correct": (
            float(clean_correct_mask.mean()) if clean_correct_mask.size else float("nan")
        ),
        "hist_r_pgd": {
            "counts": counts.tolist(),
            "bin_edges": edges.tolist(),
        },
    }


def compute_r_pgd_binary_search(
    model: torch.nn.Module,
    data_loader,
    eps_grid: np.ndarray | None = None,
    coarse_correct: np.ndarray | None = None,
    eps_min: float = 0.0,
    eps_max: float = 0.4,
    pgd_iters: int = 80,
    pgd_alpha: float = 0.01,
    pgd_random_start: bool = True,
    binary_steps: int = 10,
    tol: float | None = None,
    seed: int | None = None,
    verbose: bool = False,
    **legacy_kwargs,
) -> dict:
    """
    Refine r-PGD with binary search inside the interval found by the epsilon grid.

    For every sample, the coarse grid first gives:
        low  = last epsilon where all tested epsilons up to that point are correct
        high = first epsilon where the sample fails

    The binary search only evaluates epsilons in [low, high]. The returned r_pgd
    is the final lower bound, i.e. the largest tested epsilon that still keeps
    the sample correctly classified. Samples that never fail on the coarse grid
    keep r_pgd = eps_grid[-1].
    """
    if legacy_kwargs and verbose:
        ignored = ", ".join(sorted(legacy_kwargs.keys()))
        print(f"    Ignoring legacy r-PGD options: {ignored}")

    was_training = model.training
    model.eval()

    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    if eps_grid is None:
        if eps_max < eps_min:
            raise ValueError(f"eps_max ({eps_max}) must be >= eps_min ({eps_min}).")
        step = legacy_kwargs.get("sweep_step", None)
        if step is None or float(step) <= 0:
            step = eps_max - eps_min
        eps_grid = np.arange(float(eps_min), float(eps_max) + 1e-12, float(step), dtype=np.float64)
        if eps_grid[-1] < float(eps_max):
            eps_grid = np.append(eps_grid, float(eps_max))
    else:
        eps_grid = np.asarray(eps_grid, dtype=np.float64)

    if eps_grid.ndim != 1 or len(eps_grid) < 2:
        raise ValueError("eps_grid must be a 1D array with at least two epsilon values.")
    if not np.all(np.diff(eps_grid) > 0):
        raise ValueError("eps_grid must be strictly increasing.")

    eps_max = float(eps_grid[-1])

    if coarse_correct is None:
        coarse_correct = _coarse_correct_from_model(
            model=model,
            data_loader=data_loader,
            eps_grid=eps_grid,
            pgd_iters=pgd_iters,
            pgd_alpha=pgd_alpha,
            pgd_random_start=pgd_random_start,
            verbose=verbose,
        )
    else:
        coarse_correct = np.asarray(coarse_correct).astype(bool)

    if coarse_correct.ndim != 2 or coarse_correct.shape[1] != len(eps_grid):
        raise ValueError(
            "coarse_correct must have shape (N, len(eps_grid)); "
            f"got {coarse_correct.shape}, len(eps_grid)={len(eps_grid)}."
        )

    n_samples = coarse_correct.shape[0]
    clean_correct_mask = coarse_correct[:, 0].astype(bool)
    low = np.full(n_samples, np.nan, dtype=np.float64)
    high = np.full(n_samples, np.nan, dtype=np.float64)
    failing_mask = np.zeros(n_samples, dtype=bool)

    for i in range(n_samples):
        if not clean_correct_mask[i]:
            continue

        row = coarse_correct[i]
        if row.all():
            low[i] = eps_max
            high[i] = eps_max
            continue

        failing_mask[i] = True
        first_fail_idx = int(np.argmax(~row))
        if first_fail_idx == 0:
            low[i] = 0.0
            high[i] = 0.0
        else:
            low[i] = float(eps_grid[first_fail_idx - 1])
            high[i] = float(eps_grid[first_fail_idx])

    device = next(model.parameters()).device
    start = 0
    for batch_idx, (batch_x, batch_y) in enumerate(data_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        end = start + batch_x.shape[0]

        local_low = low[start:end].copy()
        local_high = high[start:end].copy()

        for step_idx in range(int(binary_steps)):
            active = local_high > local_low
            if tol is not None:
                active &= (local_high - local_low) > float(tol)
            active_idx = np.flatnonzero(active)
            if active_idx.size == 0:
                break

            mid_np = (local_low[active_idx] + local_high[active_idx]) / 2.0
            mid = torch.as_tensor(mid_np, device=device, dtype=batch_x.dtype)

            x_adv = _pgd_linf_attack_per_sample_eps(
                model=model,
                batch_x=batch_x[active_idx],
                batch_y=batch_y[active_idx],
                eps=mid,
                iters=pgd_iters,
                alpha=pgd_alpha,
                random_start=pgd_random_start,
            )
            correct_mid = _predict_correct(model, x_adv, batch_y[active_idx])

            local_low[active_idx[correct_mid]] = mid_np[correct_mid]
            local_high[active_idx[~correct_mid]] = mid_np[~correct_mid]

            if tol is not None and np.all((local_high - local_low) <= float(tol)):
                break

        low[start:end] = local_low
        high[start:end] = local_high
        start = end

        if verbose:
            print(f"    binary r-PGD batch {batch_idx + 1}/{len(data_loader)}")

    if was_training:
        model.train()

    r_pgd = low.astype(np.float32)
    summary = _summarize_r_pgd(
        r_pgd=r_pgd,
        failing_mask=failing_mask,
        max_eps=eps_max,
        clean_correct_mask=clean_correct_mask,
        n_bins=min(len(eps_grid), 20),
    )
    summary.update({
        "r_pgd": r_pgd,
        "r_pgd_upper": high.astype(np.float32),
        "failing_mask": failing_mask.astype(np.uint8),
        "clean_correct_mask": clean_correct_mask.astype(np.uint8),
        "eps_grid": eps_grid.astype(np.float32),
        "binary_steps": int(binary_steps),
        "tol": None if tol is None else float(tol),
        "pgd_iters": int(pgd_iters),
        "pgd_alpha": float(pgd_alpha),
        "pgd_random_start": bool(pgd_random_start),
    })
    return summary


def _population_mask(n_samples: int, clean_correct_mask: np.ndarray | None) -> np.ndarray:
    """
    Resolve the population used by sample-level aggregate metrics.

    With clean_correct_mask, aggregate only over samples classified correctly at
    eps=0. This matches the r-PGD convention: samples already wrong on clean
    data have no meaningful robustness or early-collapse population.
    """
    if clean_correct_mask is None:
        return np.ones(n_samples, dtype=bool)

    mask = np.asarray(clean_correct_mask).astype(bool)
    if mask.ndim != 1 or mask.shape[0] != n_samples:
        raise ValueError(
            "clean_correct_mask must be a 1D array with one entry per sample; "
            f"got shape {mask.shape}, expected ({n_samples},)."
        )
    return mask


def compute_icr(
    m_proto_matrix: np.ndarray,
    m_pred_matrix:  np.ndarray,
    clean_correct_mask: np.ndarray | None = None,
) -> float:
    """
    ICR (Interpretability Collapse Rate) — fraction of test samples that exhibit interpretability collapse
    at any epsilon in the sweep.

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
    ICR would be threshold-sensitive and miss samples where the phenomenon
    occurs at a different epsilon. any() counts a sample if the condition holds
    at *at least one* ε — the most inclusive and robust definition.

    Parameters
    ----------
    m_proto_matrix : (N, E) float array — prototype margin per sample per ε.
    m_pred_matrix  : (N, E) float array — prediction margin per sample per ε.

    Returns
    -------
    float in [0, 1]. Higher = more samples exhibit interpretability collapse.
    """
    condition  = (m_proto_matrix < 0) & (m_pred_matrix > 0)   # (N, E) bool
    per_sample = condition.any(axis=1)                          # (N,) bool
    mask = _population_mask(per_sample.shape[0], clean_correct_mask)
    return float(per_sample[mask].mean()) if mask.any() else float("nan")


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

def compute_tau_param(
    R_param_matrix: np.ndarray,
    eps_idx: int = 1,
    clean_correct_mask: np.ndarray | None = None,
) -> float:
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
    mask = _population_mask(R_param_matrix.shape[0], clean_correct_mask)
    if not mask.any():
        return float("nan")
    baseline_col = R_param_matrix[mask, eps_idx]
    return float(np.percentile(baseline_col, 95))


def compute_icr_senn(
    R_param_matrix: np.ndarray,
    m_pred_matrix:  np.ndarray,
    tau_param:      float,
    clean_correct_mask: np.ndarray | None = None,
) -> float:
    """
    ICR_SENN — fraction of samples where SENN's explanation collapses
    before its classification output changes.

    Formula
    -------
    ICR_SENN = (1/N) * sum_{i=1}^{N} max_{eps in E}
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
    mask = _population_mask(per_sample.shape[0], clean_correct_mask)
    return float(per_sample[mask].mean()) if mask.any() else float("nan")
