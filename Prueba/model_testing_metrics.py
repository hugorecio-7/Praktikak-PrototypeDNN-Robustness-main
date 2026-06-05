"""
model_testing_metrics.py
========================
Drop-in companion to model_testing.py.

Exports a single public function:

    adversarial_metrics_eps_collect(...)

It is a strict *superset* of adversarial_attacks_eps_plot_test():
it produces every file that the original produces (curve.csv, per_example_correct.npz,
metrics.json, accuracy plot) AND adds:

    per_example_metrics.npz   — per-sample metric matrices (N, E) for every
                                 internal metric available for that architecture.
    metrics.json              — extended with ICR, empirical robustness
                                 interval, and mean metric curves per epsilon.

Architecture routing
--------------------
Detection is done via isinstance at the start of each model iteration, mirroring
the strategy in metric_extractor.py.  The three families and their metrics are:

    B30  (CAEModel_Balanced, raw)
        m_proto, m_pred, R_enc, R_dec

    ProtoVAE  (ProtoVAEWrapper)
        m_proto, m_pred, R_enc (= R_mu), R_dec, R_mu, R_sigma

    SENN  (SENNWrapper)
        m_pred, R_concept, R_param
        (no m_proto — SENN has no prototype layer)

Loop structure
--------------
Identical to adversarial_attacks_eps_plot_test():
    attack loop
      └─ batch loop
           └─ model loop
                ├─ clean internals (computed ONCE per batch×model)
                └─ epsilon loop
                     └─ adv internals (computed ONCE per batch×model×eps)

This means extract_internals() is called at most
    (n_batches × n_models × (1 + n_epsilons))
times — no redundant forward passes.
"""

from __future__ import annotations

import os
import json
import time
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from model_testing import _ensure_dir, _save_json, bootstrap_ci_mean_matrix
from metric_extractor import extract_internals, get_proto_labels
from metric_calculators import (
    calc_m_proto,
    calc_m_pred,
    calc_R_enc,
    calc_R_dec,
    calc_R_mu,
    calc_R_sigma,
    calc_R_concept,
    calc_R_param,
    calc_artificial_prototype_match_suppression,
    compute_empirical_robustness_interval,
    compute_icr,
    compute_tau_param,
    compute_icr_senn,
)


# =============================================================================
# Architecture detection helpers
# =============================================================================

def _arch_flags(model: nn.Module) -> tuple[bool, bool, bool]:
    """
    Return (is_b30, is_protovae, is_senn) flags for a model.

    Exactly one of the three will be True. All three False is a bug.
    """
    base = model.base if hasattr(model, "base") else model

    is_protovae = (
        hasattr(model, "base")
        and hasattr(base, "prototype_class_identity")
        and hasattr(base, "prototype_vectors")
        and hasattr(base, "pred_class")
        and hasattr(base, "calc_sim_scores")
        and hasattr(base, "decoder")
    )

    is_senn = (
        hasattr(model, "base")
        and hasattr(base, "conceptizer")
        and hasattr(base, "parameterizer")
        and hasattr(base, "aggregator")
    )

    is_b30 = not is_protovae and not is_senn
    return is_b30, is_protovae, is_senn


def _is_shield_model(model: nn.Module) -> bool:
    return (
        hasattr(model, "base_model")
        and hasattr(model, "prototype_reconstructions")
        and hasattr(model, "gamma")
        and hasattr(model, "power")
        and hasattr(model, "_pairwise_ssim")
    )


def _extract_shield_internals(model: nn.Module, x: torch.Tensor) -> dict:
    """
    Extract B30 Shield internals.

    For Shield models, d_proto is the distance actually used by the classifier:

        d_final = d_latent + gamma * (1 - SSIM(x, Dec(p_j))) ** power

    d_proto_latent is kept separately so suppression/harm diagnostics can compare
    the original latent distance against the risk-aware distance.
    """
    base = model.base_model
    encoder_out = base.encoder(x)
    base.feature_vectors = encoder_out
    z = encoder_out.view(-1, base.in_channels_prototype)
    x_hat = base.decoder(encoder_out)
    d_latent = base.prototype_layer(z)
    ssim_matrix = model._pairwise_ssim(x)
    d_final = d_latent + float(model.gamma) * ((1.0 - ssim_matrix) ** int(model.power))
    logits = base.fc(d_final)
    model.last_d_latente = d_latent
    model.last_d_final = d_final

    return {
        "z": z.detach(),
        "d_proto": d_final.detach(),
        "d_proto_latent": d_latent.detach(),
        "d_proto_final": d_final.detach(),
        "x_hat": x_hat.detach(),
        "logits": logits.detach(),
    }


def _extract_internals_for_metrics(model: nn.Module, x: torch.Tensor) -> dict:
    if _is_shield_model(model):
        with torch.no_grad():
            return _extract_shield_internals(model, x)
    return extract_internals(model, x)


def _tensor_to_numpy(tensor: torch.Tensor, dtype=np.float32) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(dtype)


def _append_shield_match_metrics(
    store: dict,
    eps_idx: int,
    internals: dict,
    labels: torch.Tensor,
    proto_labels: torch.Tensor,
) -> None:
    outputs = calc_artificial_prototype_match_suppression(
        internals["d_proto_latent"],
        internals["d_proto_final"],
        labels,
        proto_labels,
    )
    store["false_match_mu"][eps_idx].append(
        _tensor_to_numpy(outputs["false_match_mu"], dtype=np.float32)
    )
    store["suppressed_match"][eps_idx].append(
        _tensor_to_numpy(outputs["suppressed_match"], dtype=np.float32)
    )
    store["risk_harm_match"][eps_idx].append(
        _tensor_to_numpy(outputs["risk_harm_match"], dtype=np.float32)
    )


def _shield_metric_summary(M: dict[str, np.ndarray]) -> dict | None:
    required = {"false_match_mu", "suppressed_match", "risk_harm_match"}
    if not required.issubset(M):
        return None

    false_match = M["false_match_mu"].astype(bool)
    suppressed = M["suppressed_match"].astype(bool)
    risk_harm = M["risk_harm_match"].astype(bool)
    n_samples = int(false_match.shape[0])

    suppressed_curve = []
    harm_curve = []
    false_counts = []
    suppressed_counts = []
    harm_counts = []

    for eps_idx in range(false_match.shape[1]):
        false_count = int(false_match[:, eps_idx].sum())
        suppressed_count = int(suppressed[:, eps_idx].sum())
        harm_count = int(risk_harm[:, eps_idx].sum())
        false_counts.append(false_count)
        suppressed_counts.append(suppressed_count)
        harm_counts.append(harm_count)
        suppressed_curve.append(
            float((suppressed_count / false_count) * 100.0)
            if false_count > 0
            else 0.0
        )
        harm_curve.append(
            float((harm_count / n_samples) * 100.0)
            if n_samples > 0
            else float("nan")
        )

    return {
        "suppressed_matches": suppressed_curve,
        "harm_rate": harm_curve,
        "false_match_mu_count": false_counts,
        "suppressed_match_count": suppressed_counts,
        "risk_harm_match_count": harm_counts,
        "definition": (
            "suppressed_matches is 100 * suppressed_match / false_match_mu per "
            "epsilon; harm_rate is 100 * risk_harm_match / N per epsilon."
        ),
    }


def _init_metric_store(dim: int, is_b30: bool, is_protovae: bool, is_senn: bool, is_shield: bool = False) -> dict:
    """
    Initialise the per-epsilon accumulator for one model.

    Returns a dict  { metric_name: [[] for _ in range(dim)] }
    where each inner list will collect (B,) numpy arrays, one per batch.

    m_pred is always present. All others depend on architecture.
    """
    store: dict[str, list] = {"m_pred": [[] for _ in range(dim)]}

    if is_b30 or is_protovae:
        store["m_proto"] = [[] for _ in range(dim)]
        store["R_enc"]   = [[] for _ in range(dim)]
        store["R_dec"]   = [[] for _ in range(dim)]

    if is_shield:
        store["m_proto_latent"] = [[] for _ in range(dim)]
        store["false_match_mu"] = [[] for _ in range(dim)]
        store["suppressed_match"] = [[] for _ in range(dim)]
        store["risk_harm_match"] = [[] for _ in range(dim)]

    if is_protovae:
        store["R_mu"]    = [[] for _ in range(dim)]
        store["R_sigma"] = [[] for _ in range(dim)]

    if is_senn:
        store["R_concept"] = [[] for _ in range(dim)]
        store["R_param"]   = [[] for _ in range(dim)]

    return store


def _module_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def _pgd_linf_attack_per_sample_eps(
    attack_model: nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    eps: torch.Tensor,
    iters: int,
    alpha: float,
    random_start: bool,
) -> torch.Tensor:
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
        logits = attack_model(perturbed_batch_x)
        loss = loss_function(logits, batch_y)
        gradients = torch.autograd.grad(loss, perturbed_batch_x)[0]

        perturbed_batch_x = perturbed_batch_x.detach() + float(alpha) * torch.sign(gradients)
        delta = torch.clamp(perturbed_batch_x - ori_images, min=-eps, max=eps)
        perturbed_batch_x = torch.clamp(ori_images + delta, min=0, max=1).detach()

    return perturbed_batch_x


def _predict_correct(
    eval_model: nn.Module,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
) -> np.ndarray:
    with torch.no_grad():
        logits = eval_model(batch_x)
        preds = torch.argmax(logits, dim=1)
    return (preds == batch_y).detach().cpu().numpy().astype(bool)


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


def compute_r_pgd_binary_search_for_eval_model(
    attack_model: nn.Module,
    eval_model: nn.Module,
    data_loader,
    eps_grid: np.ndarray,
    coarse_correct: np.ndarray,
    pgd_iters: int = 80,
    pgd_alpha: float = 0.01,
    pgd_random_start: bool = True,
    binary_steps: int = 10,
    tol: float | None = None,
    seed: int | None = None,
    verbose: bool = False,
) -> dict:
    """
    Binary-search r-PGD where adversarial examples may be generated on one
    model and evaluated on another. This is the Shield grey-box path.
    """
    was_training_attack = attack_model.training
    was_training_eval = eval_model.training
    attack_model.eval()
    eval_model.eval()

    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    eps_grid = np.asarray(eps_grid, dtype=np.float64)
    if eps_grid.ndim != 1 or len(eps_grid) < 2:
        raise ValueError("eps_grid must be a 1D array with at least two epsilon values.")
    if not np.all(np.diff(eps_grid) > 0):
        raise ValueError("eps_grid must be strictly increasing.")

    coarse_correct = np.asarray(coarse_correct).astype(bool)
    if coarse_correct.ndim != 2 or coarse_correct.shape[1] != len(eps_grid):
        raise ValueError(
            "coarse_correct must have shape (N, len(eps_grid)); "
            f"got {coarse_correct.shape}, len(eps_grid)={len(eps_grid)}."
        )

    eps_max = float(eps_grid[-1])
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

    device = _module_device(eval_model)
    start = 0
    for batch_idx, (batch_x, batch_y) in enumerate(data_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        end = start + batch_x.shape[0]

        local_low = low[start:end].copy()
        local_high = high[start:end].copy()

        for _ in range(int(binary_steps)):
            active = local_high > local_low
            if tol is not None:
                active &= (local_high - local_low) > float(tol)
            active_idx = np.flatnonzero(active)
            if active_idx.size == 0:
                break

            mid_np = (local_low[active_idx] + local_high[active_idx]) / 2.0
            mid = torch.as_tensor(mid_np, device=device, dtype=batch_x.dtype)

            x_adv = _pgd_linf_attack_per_sample_eps(
                attack_model=attack_model,
                batch_x=batch_x[active_idx],
                batch_y=batch_y[active_idx],
                eps=mid,
                iters=pgd_iters,
                alpha=pgd_alpha,
                random_start=pgd_random_start,
            )
            correct_mid = _predict_correct(eval_model, x_adv, batch_y[active_idx])

            local_low[active_idx[correct_mid]] = mid_np[correct_mid]
            local_high[active_idx[~correct_mid]] = mid_np[~correct_mid]

            if tol is not None and np.all((local_high - local_low) <= float(tol)):
                break

        low[start:end] = local_low
        high[start:end] = local_high
        start = end

        if verbose:
            print(f"    binary r-PGD batch {batch_idx + 1}/{len(data_loader)}")

    if was_training_attack:
        attack_model.train()
    if was_training_eval:
        eval_model.train()

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


# =============================================================================
# Main function
# =============================================================================

def adversarial_metrics_eps_collect(
    models,
    model_names,
    test_loader,
    attacks,
    attack_names,
    loss,
    max_eps,
    step,
    foolbox_uses,
    out_root,
    dataset,
    threat,
    seed,
    run_id,
    attack_params_map=None,
    attack_models=None,
    shield_pgd_mode: str = "white-box",
    bootstrap_B: int   = 2000,
    bootstrap_alpha: float = 0.05,
    # --- r-PGD binary-search refinement options ------------------------------
    rpgd_binary_steps: int = 10,
    rpgd_binary_tol: float | None = None,
) -> None:
    """
    Evaluate models under adversarial attacks across an epsilon grid, collecting
    both accuracy and all internal interpretability metrics defined in the TFG.

    This function is a superset of adversarial_attacks_eps_plot_test(): it
    produces identical legacy outputs (CSV, plot, metrics.json) and adds
    per_example_metrics.npz with (N, E) matrices for every internal metric.

    Parameters
    ----------
    models : list[nn.Module]
        Loaded, eval-mode model instances (wrapped or raw).
    model_names : list[str]
        Names matching `models` 1-to-1.
    test_loader : DataLoader
        Test set. Batches yield (x, y) with x in [0, 1].
    attacks : list[callable]
        Attack functions, already partially applied with their hyperparameters
        (from run_metrics.py via functools.partial).
    attack_names : list[str]
        Names matching `attacks` 1-to-1.
    loss : callable
        Loss function used for non-foolbox attacks (e.g. CELoss).
    max_eps : float | int
        Maximum epsilon value for the grid.
    step : float
        Step between epsilon values (for float grids).
    foolbox_uses : list[bool]
        Whether each attack is a foolbox/AutoAttack attack.
    out_root : str
        Root output directory (mirrors adversarial_attacks_eps_plot_test).
    dataset : str
        Dataset name for path construction.
    threat : str
        Threat model name for path construction.
    seed : int
        Global seed (used for bootstrap and path construction).
    run_id : str | None
        Run identifier. Auto-generated from timestamp if None.
    attack_params_map : dict | None
        Attack hyperparameters for logging in metrics.json.
    attack_models : list[nn.Module] | None
        Optional models used to generate PGD adversarial examples. When None,
        evaluation models are attacked directly. Shield grey-box passes the
        matching non-Shield B30 model here, e.g. B30-FT-E-M for
        B30-FT-E-M-Shield, while still evaluating the Shield wrapper.
    shield_pgd_mode : str
        Logged attack mode for Shield PGD/AutoAttack models ("white-box" or
        "grey-box").
    bootstrap_B : int
        Bootstrap resamples for accuracy confidence intervals.
    bootstrap_alpha : float
        Alpha level for confidence intervals (0.05 → 95% CI).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build epsilon grid — identical logic to adversarial_attacks_eps_plot_test.
    if isinstance(max_eps, float):
        x_axis = np.arange(0.0, max_eps + 1e-12, step, dtype=np.float64)
    else:
        x_axis = np.arange(0, max_eps + 1, 1, dtype=np.int64)
    dim = len(x_axis)   # number of epsilon points including ε=0

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if attack_models is None:
        attack_models = models
    if len(attack_models) != len(models):
        raise ValueError(
            "attack_models must have the same length as models, "
            f"got {len(attack_models)} and {len(models)}."
        )

    # -------------------------------------------------------------------------
    # Pre-compute architecture flags and prototype labels (once per model).
    # These never change across attacks or batches.
    # -------------------------------------------------------------------------
    arch_flags_per_model  = [_arch_flags(m)            for m in models]
    shield_flags_per_model = [_is_shield_model(m)      for m in models]
    proto_labels_per_model = []
    for model in models:
        _, _, is_senn = _arch_flags(model)
        if is_senn:
            proto_labels_per_model.append(None)   # SENN has no prototypes
        else:
            proto_labels_per_model.append(get_proto_labels(model))

    # =========================================================================
    # Outer loop: one full sweep per attack
    # =========================================================================
    for attack, attack_name, foolbox_use in zip(attacks, attack_names, foolbox_uses):

        # -----------------------------------------------------------------
        # Accumulators — reset for each attack.
        #
        # per_example_correct[idm][ide] : list of (B,) uint8 arrays
        #     Correct classification flag per sample per epsilon.
        #
        # metric_store[idm] : dict { metric_name → [[] for _ in range(dim)] }
        #     Each inner list collects (B,) float32 arrays per batch.
        # -----------------------------------------------------------------
        per_example_correct = [
            [[] for _ in range(dim)] for _ in range(len(models))
        ]
        metric_store = [
            _init_metric_store(
                dim,
                *arch_flags_per_model[idm],
                is_shield=shield_flags_per_model[idm],
            )
            for idm in range(len(models))
        ]

        t0 = time.time()

        # =================================================================
        # Batch loop
        # =================================================================
        for batch_idx, batch in enumerate(test_loader):
            batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # =============================================================
            # Model loop
            # =============================================================
            for idm, model in enumerate(models):
                is_b30, is_protovae, is_senn = arch_flags_per_model[idm]
                is_shield                    = shield_flags_per_model[idm]
                proto_labels                 = proto_labels_per_model[idm]
                store                        = metric_store[idm]
                B                            = batch_x.shape[0]

                # ----------------------------------------------------------
                # CLEAN internals — computed ONCE per (batch, model).
                # These are reused as the reference for all drift metrics
                # (R_enc, R_dec, R_mu, R_sigma, R_concept, R_param).
                # ----------------------------------------------------------
                clean_int = _extract_internals_for_metrics(model, batch_x)

                # Accuracy at ε = 0 (clean).
                _, max_indices = torch.max(clean_int["logits"], 1)
                per_example_correct[idm][0].append(
                    (max_indices == batch_y).detach().cpu().numpy().astype(np.uint8)
                )

                # Metrics at ε = 0.
                # Drift metrics are identically zero — no perturbation applied.
                zeros_B = np.zeros(B, dtype=np.float32)

                store["m_pred"][0].append(
                    calc_m_pred(clean_int["logits"], batch_y)
                )
                if is_b30 or is_protovae:
                    store["m_proto"][0].append(
                        calc_m_proto(clean_int["d_proto"], batch_y, proto_labels)
                    )
                    store["R_enc"][0].append(zeros_B.copy())
                    store["R_dec"][0].append(zeros_B.copy())
                    if is_shield:
                        store["m_proto_latent"][0].append(
                            calc_m_proto(clean_int["d_proto_latent"], batch_y, proto_labels)
                        )
                        _append_shield_match_metrics(
                            store,
                            0,
                            clean_int,
                            batch_y,
                            proto_labels,
                        )
                if is_protovae:
                    store["R_mu"][0].append(zeros_B.copy())
                    store["R_sigma"][0].append(zeros_B.copy())
                if is_senn:
                    store["R_concept"][0].append(zeros_B.copy())
                    store["R_param"][0].append(zeros_B.copy())

                # Prepare loss function for non-foolbox attacks (same as original).
                if not foolbox_use:
                    attack_model = attack_models[idm] if attack_name == "PGDLInf_attack" else model
                    loss_f = partial(loss, model=attack_model, batch_y=batch_y)

                # ==========================================================
                # Epsilon loop  (ide=0 is ε=0, already handled above)
                # ==========================================================
                for ide in range(1, dim):
                    eps = float(x_axis[ide])

                    # Generate adversarial example — identical branching to original.
                    if foolbox_use:
                        attack_model = attack_models[idm] if attack_name == "AutoAttack_adv" else model
                        x_adv = attack(batch_x, batch_y, model=attack_model, epsilon=eps)
                    else:
                        if attack_name == "PatchPGD_attack":
                            adv_attack = partial(attack, loss_f=loss_f, eps=eps, batch_idx=batch_idx)
                        else:
                            adv_attack = partial(attack, loss_f=loss_f, eps=eps)
                        x_adv = adv_attack(batch_x)

                    # ADV internals — computed ONCE per (batch, model, eps).
                    adv_int = _extract_internals_for_metrics(model, x_adv)

                    # ---- Accuracy ----------------------------------------
                    _, max_indices_adv = torch.max(adv_int["logits"], 1)
                    per_example_correct[idm][ide].append(
                        (max_indices_adv == batch_y).detach().cpu().numpy().astype(np.uint8)
                    )

                    # ---- m_pred (all architectures) ----------------------
                    store["m_pred"][ide].append(
                        calc_m_pred(adv_int["logits"], batch_y)
                    )

                    # ---- Prototype-based metrics (B30 + ProtoVAE) --------
                    if is_b30 or is_protovae:
                        store["m_proto"][ide].append(
                            calc_m_proto(adv_int["d_proto"], batch_y, proto_labels)
                        )
                        store["R_enc"][ide].append(
                            calc_R_enc(clean_int["z"], adv_int["z"])
                        )
                        store["R_dec"][ide].append(
                            calc_R_dec(clean_int["x_hat"], adv_int["x_hat"])
                        )
                        if is_shield:
                            store["m_proto_latent"][ide].append(
                                calc_m_proto(adv_int["d_proto_latent"], batch_y, proto_labels)
                            )
                            _append_shield_match_metrics(
                                store,
                                ide,
                                adv_int,
                                batch_y,
                                proto_labels,
                            )

                    # ---- ProtoVAE-specific --------------------------------
                    if is_protovae:
                        store["R_mu"][ide].append(
                            calc_R_mu(clean_int["mu"], adv_int["mu"])
                        )
                        store["R_sigma"][ide].append(
                            calc_R_sigma(clean_int["sigma"], adv_int["sigma"])
                        )

                    # ---- SENN-specific -----------------------------------
                    if is_senn:
                        store["R_concept"][ide].append(
                            calc_R_concept(clean_int["h_concept"], adv_int["h_concept"])
                        )
                        store["R_param"][ide].append(
                            calc_R_param(clean_int["theta_param"], adv_int["theta_param"])
                        )

        runtime_s = time.time() - t0

        # =================================================================
        # Post-processing — once per (attack, model) after all batches.
        # =================================================================
        for idm, model_name in enumerate(model_names):
            is_b30, is_protovae, is_senn = arch_flags_per_model[idm]
            is_shield = shield_flags_per_model[idm]
            store = metric_store[idm]

            # ---- Accuracy matrix ----------------------------------------
            # X_correct : (N_test, dim)  bool/uint8
            cols     = [np.concatenate(per_example_correct[idm][j], axis=0) for j in range(dim)]
            X_correct = np.stack(cols, axis=1)          # (N, E)
            acc_curve = X_correct.mean(axis=0)          # (E,)
            clean_correct_mask = X_correct[:, 0].astype(bool)

            ci_low, ci_high = bootstrap_ci_mean_matrix(
                X_correct, B=bootstrap_B, alpha=bootstrap_alpha,
                seed=seed + idm, chunk=200
            )

            # ---- Internal metric matrices --------------------------------
            # M[metric_name] : (N_test, dim)  float32
            M: dict[str, np.ndarray] = {}
            for metric_name, batches_per_eps in store.items():
                cols_m = [np.concatenate(batches_per_eps[j], axis=0) for j in range(dim)]
                M[metric_name] = np.stack(cols_m, axis=1).astype(np.float32)   # (N, E)

            shield_summary = _shield_metric_summary(M)

            # ---- Global metrics -----------------------------------------

            # Empirical robustness interval — uses the monotone-from-zero definition.
            rob_interval = compute_empirical_robustness_interval(
                X_correct.astype(bool), x_axis.astype(np.float64)
            )

            # ICR — only meaningful for prototype-based architectures.
            icr: float | None = None
            icr_all: float | None = None
            if is_b30 or is_protovae:
                icr_all = compute_icr(M["m_proto"], M["m_pred"])
                icr = compute_icr(
                    M["m_proto"],
                    M["m_pred"],
                    clean_correct_mask=clean_correct_mask,
                )
                
            # ICR_SENN and tau_param — SENN only.
            # tau_param is computed from the R_param distribution at the first
            # non-zero epsilon (column 1), which defines the empirical noise floor
            # of the Parameterizer under minimal adversarial pressure.
            tau_param:       float | None = None
            tau_param_all:   float | None = None
            icr_senn: float | None = None
            icr_senn_all: float | None = None
            if is_senn:
                tau_param_all = compute_tau_param(M["R_param"], eps_idx=1)
                icr_senn_all = compute_icr_senn(
                    M["R_param"], M["m_pred"], tau_param_all
                )
                tau_param = compute_tau_param(
                    M["R_param"],
                    eps_idx=1,
                    clean_correct_mask=clean_correct_mask,
                )
                icr_senn = compute_icr_senn(
                    M["R_param"],
                    M["m_pred"],
                    tau_param,
                    clean_correct_mask=clean_correct_mask,
                )

            # ---- Binary-search r-PGD (only for PGDLInf_attack) ------
            rpgd_bs_result: dict | None = None
            if attack_name == "PGDLInf_attack":
                pgd_params = attack_params_map.get("PGDLInf_attack", {}) if attack_params_map else {}
                print(f"\n  Refining r-PGD with binary search for {model_name}...")
                rpgd_bs_result = compute_r_pgd_binary_search_for_eval_model(
                    attack_model=attack_models[idm],
                    eval_model=models[idm],
                    data_loader=test_loader,
                    eps_grid=x_axis.astype(np.float64),
                    coarse_correct=X_correct.astype(bool),
                    pgd_iters=pgd_params.get("iters", 80),
                    pgd_alpha=pgd_params.get("alpha", 0.01),
                    pgd_random_start=pgd_params.get("random_start", True),
                    binary_steps=rpgd_binary_steps,
                    tol=rpgd_binary_tol,
                    seed=seed + idm,
                    verbose=True,
                )

            # ---- Directories --------------------------------------------
            acc_dir  = f"resultsTFG/Accuracy/{attack_name}/{model_name}"
            plot_dir = f"resultsTFG/Plot/{attack_name}/{model_name}"
            run_dir  = os.path.join(
                out_root, dataset, threat, attack_name, model_name,
                f"seed={seed}", f"run_id={run_id}"
            )
            for d in (acc_dir, plot_dir, run_dir):
                _ensure_dir(d)

            # ---- Legacy CSV (mirrors original) --------------------------
            csv_path = f"{acc_dir}/maxeps({max_eps})_step({step})_seed{seed}_run{run_id}_results.csv"
            df = pd.DataFrame({model_name: acc_curve}, index=x_axis)
            df.index.name = "Epsilon"
            df.to_csv(csv_path)
            df.to_csv(os.path.join(run_dir, "curve.csv"))
            print(f"Accuracy results saved to {csv_path}")

            # ---- Legacy plot (mirrors original) -------------------------
            jpg_path = f"{plot_dir}/maxeps({max_eps})_step({step})_seed{seed}_run{run_id}_plot.jpg"
            plt.figure(figsize=(8, 6))
            plt.plot(x_axis, acc_curve, label=model_name)
            plt.scatter(x_axis, acc_curve)
            plt.xlabel("Epsilon")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.title(
                f"Adversarial Attack Accuracy vs. Epsilon\n"
                f"Model: {model_name} | Attack: {attack_name}"
            )
            plt.savefig(jpg_path, dpi=300)
            plt.close()
            print(f"Plot saved to {jpg_path}")

            # ---- per_example_correct.npz (mirrors original) -------------
            np.savez_compressed(
                os.path.join(run_dir, "per_example_correct.npz"),
                correct=X_correct.astype(np.uint8),
                eps=x_axis.astype(np.float32),
            )

            # ---- per_example_metrics.npz (NEW) --------------------------
            # Stores (N, E) float32 matrices for every computed metric.
            # Keys are metric names; eps array is included for convenience.
            npz_payload: dict[str, np.ndarray] = {"eps": x_axis.astype(np.float32)}
            npz_payload.update(M)
            np.savez_compressed(
                os.path.join(run_dir, "per_example_metrics.npz"),
                **npz_payload,
            )
            print(f"Internal metrics saved to {os.path.join(run_dir, 'per_example_metrics.npz')}")

            # ---- r_pgd_binary_search.npz (per-sample r-PGD) --------
            if rpgd_bs_result is not None:
                np.savez_compressed(
                    os.path.join(run_dir, "r_pgd_binary_search.npz"),
                    r_pgd=rpgd_bs_result["r_pgd"],
                    r_pgd_upper=rpgd_bs_result["r_pgd_upper"],
                    failing_mask=rpgd_bs_result["failing_mask"],
                    clean_correct_mask=rpgd_bs_result["clean_correct_mask"],
                    eps_grid=rpgd_bs_result["eps_grid"],
                )
                print(f"Binary-search r-PGD saved to {os.path.join(run_dir, 'r_pgd_binary_search.npz')}")

            # ---- Extended metrics.json ----------------------------------
            clean_acc = float(acc_curve[0]) if float(acc_curve[0]) > 0 else 1e-12
            rel_deg   = (clean_acc - acc_curve) / clean_acc * 100.0

            # Monotonicity violations in accuracy (from original).
            viol = []
            if isinstance(max_eps, float) and dim > 2:
                tol = 0.01
                for j in range(dim - 1):
                    if acc_curve[j + 1] > acc_curve[j] + tol:
                        viol.append({
                            "from_eps": float(x_axis[j]),
                            "to_eps":   float(x_axis[j + 1]),
                            "delta":    float(acc_curve[j + 1] - acc_curve[j]),
                        })

            params = {}
            if attack_params_map is not None:
                params = attack_params_map.get(attack_name, {}).copy()
            if is_shield and attack_name in {"PGDLInf_attack", "AutoAttack_adv"}:
                params["shield_pgd_mode"] = shield_pgd_mode
                params["shield_attack_mode"] = shield_pgd_mode

            shield_config = None
            if is_shield:
                shield_config = {
                    "enabled": True,
                    "gamma": float(models[idm].gamma),
                    "power": int(models[idm].power),
                    "pgd_mode": shield_pgd_mode,
                    "attack_mode": shield_pgd_mode,
                    "distance_formula": (
                        "d_final(x,p_j) = d_latent(x,p_j) + gamma * "
                        "(1 - SSIM(x, Dec(p_j))) ** power"
                    ),
                    "m_proto_for_icr": "d_final",
                    "m_proto_latent_metric": "m_proto_latent",
                }

            # Mean metric curves per epsilon (for quick inspection in JSON).
            mean_metric_curves: dict[str, list] = {
                name: mat.mean(axis=0).tolist() for name, mat in M.items()
            }

            robustness_json = {
                # Consistent-population means (same N - only failing samples).
                # mean_r_minus_failing <= mean_r_plus always holds.
                "mean_r_minus_failing": rob_interval["mean_r_minus_failing"],
                "mean_r_plus":          rob_interval["mean_r_plus"],
                # All-sample mean (includes never-failing samples at max_eps).
                # Higher than mean_r_minus_failing; NOT comparable to mean_r_plus.
                "mean_r_minus_all":     rob_interval["mean_r_minus_all"],
                "frac_never_fail":      rob_interval["frac_never_fail"],
                "hist_r_minus":         rob_interval["hist_r_minus"],
                "binary_search_refined": False,
                "definition":           (
                    "r_minus = max{eps in E | correct at all eps' <= eps}; "
                    "r_plus  = first eps where model fails; "
                    "mean_r_minus_failing and mean_r_plus share the same population "
                    "(samples that fail at least once); "
                    "mean_r_minus_all averages over all N samples."
                ),
            }

            rpgd_bs_json = None
            if rpgd_bs_result is not None:
                robustness_json.update({
                    "grid_mean_r_minus_failing": rob_interval["mean_r_minus_failing"],
                    "grid_mean_r_minus_all":     rob_interval["mean_r_minus_all"],
                    "grid_hist_r_minus":         rob_interval["hist_r_minus"],
                    "mean_r_minus_failing":      rpgd_bs_result["mean_r_pgd_failing"],
                    "mean_r_minus_all":          rpgd_bs_result["mean_r_pgd"],
                    "min_r_minus_all":           rpgd_bs_result["min_r_pgd"],
                    "frac_never_fail":           rpgd_bs_result["frac_never_fail"],
                    "n_clean_correct":           rpgd_bs_result["n_clean_correct"],
                    "n_excluded_clean_incorrect": rpgd_bs_result["n_excluded_clean_incorrect"],
                    "frac_clean_correct":        rpgd_bs_result["frac_clean_correct"],
                    "hist_r_minus":              rpgd_bs_result["hist_r_pgd"],
                    "binary_search_refined":     True,
                    "definition": (
                        "r_minus is refined by binary search inside the coarse epsilon-grid "
                        "interval [last correct eps, first failing eps]. r_plus remains the "
                        "first failing epsilon from the coarse grid. Samples misclassified at "
                        "eps=0 are excluded because their attack robustness radius is undefined."
                    ),
                })
                rpgd_bs_json = {
                    "mean_r_pgd":          rpgd_bs_result["mean_r_pgd"],
                    "mean_r_pgd_failing":  rpgd_bs_result["mean_r_pgd_failing"],
                    "min_r_pgd":           rpgd_bs_result["min_r_pgd"],
                    "frac_never_fail":     rpgd_bs_result["frac_never_fail"],
                    "n_total":             rpgd_bs_result["n_total"],
                    "n_clean_correct":     rpgd_bs_result["n_clean_correct"],
                    "n_excluded_clean_incorrect": rpgd_bs_result["n_excluded_clean_incorrect"],
                    "frac_clean_correct":  rpgd_bs_result["frac_clean_correct"],
                    "binary_steps":        rpgd_bs_result["binary_steps"],
                    "tol":                 rpgd_bs_result["tol"],
                    "pgd_iters":           rpgd_bs_result["pgd_iters"],
                    "pgd_alpha":           rpgd_bs_result["pgd_alpha"],
                    "pgd_random_start":    rpgd_bs_result["pgd_random_start"],
                    "npz_file":            "r_pgd_binary_search.npz",
                }

            metrics_dict = {
                # ---- provenance -----------------------------------------
                "dataset":      dataset,
                "threat_model": threat,
                "model":        model_name,
                "attack":       attack_name,
                "seed":         int(seed),
                "run_id":       run_id,
                "device":       str(device),
                "gpu_name":     gpu_name,
                "runtime_s":    float(runtime_s),
                "n_test":       int(X_correct.shape[0]),
                # ---- accuracy -------------------------------------------
                "eps":           [float(e) for e in x_axis.tolist()],
                "acc":           [float(a) for a in acc_curve.tolist()],
                "relative_degradation_pct": [float(r) for r in rel_deg.tolist()],
                "ci95_bootstrap": {
                    "low":   ci_low,
                    "high":  ci_high,
                    "B":     int(bootstrap_B),
                    "alpha": float(bootstrap_alpha),
                    "seed":  int(seed + idm),
                },
                # ---- empirical robustness interval ----------------------
                "empirical_robustness_interval": robustness_json,
                "r_pgd_binary_search": rpgd_bs_json,
                "grid_empirical_robustness_interval": {
                    # Consistent-population means (same N — only failing samples).
                    # mean_r_minus_failing <= mean_r_plus always holds.
                    "mean_r_minus_failing": rob_interval["mean_r_minus_failing"],
                    "mean_r_plus":          rob_interval["mean_r_plus"],
                    # All-sample mean (includes never-failing samples at max_eps).
                    # Higher than mean_r_minus_failing; NOT comparable to mean_r_plus.
                    "mean_r_minus_all":     rob_interval["mean_r_minus_all"],
                    "frac_never_fail":      rob_interval["frac_never_fail"],
                    "hist_r_minus":         rob_interval["hist_r_minus"],
                    "definition":           (
                        "r_minus = max{eps in E | correct at all eps' <= eps}; "
                        "r_plus  = first eps where model fails; "
                        "mean_r_minus_failing and mean_r_plus share the same population "
                        "(samples that fail at least once); "
                        "mean_r_minus_all averages over all N samples."
                    ),
                },
                # ---- ICR (prototype architectures only) -----------------
                "icr": icr,   # None for SENN
                "icr_all": icr_all,
                "icr_population": {
                    "n_total": int(X_correct.shape[0]),
                    "n_clean_correct": int(clean_correct_mask.sum()),
                    "n_excluded_clean_incorrect": int((~clean_correct_mask).sum()),
                    "frac_clean_correct": float(clean_correct_mask.mean()),
                },
                "icr_definition": (
                    "(1/N_clean_correct) * sum_i max_{eps} "
                    "I[m_proto(x_i,eps)<0 & m_pred(x_i,eps)>0], "
                    "restricted to samples correctly classified at eps=0. "
                    "Legacy all-sample value is stored as icr_all."
                    if icr is not None else "N/A — SENN has no prototypes"
                ),
                "icr_shield_definition": (
                    "For Shield models, ICR uses m_proto computed from "
                    "d_final(x,p_j)=d_latent(x,p_j)+gamma*(1-SSIM(x,Dec(p_j)))^power. "
                    "For non-Shield models this field is N/A."
                    if is_shield else "N/A"
                ),
                # ---- Shield diagnostics ---------------------------------
                "shield": shield_config,
                "suppressed_matches": (
                    None if shield_summary is None else shield_summary["suppressed_matches"]
                ),
                "harm_rate": (
                    None if shield_summary is None else shield_summary["harm_rate"]
                ),
                "shield_metrics": shield_summary,
                # ---- ICR_SENN (SENN only) -------------------------------
                "tau_param": tau_param,         # None for B30/ProtoVAE
                "tau_param_all": tau_param_all,
                "icr_senn": icr_senn,  # None for B30/ProtoVAE
                "icr_senn_all": icr_senn_all,
                "icr_senn_definition": (
                    f"(1/N_clean_correct) * sum_i max_{{eps}} "
                    f"I[R_param(x_i,eps)>tau({tau_param:.6f}) & m_pred(x_i,eps)>0]; "
                    "restricted to samples correctly classified at eps=0. "
                    "tau = p95 of R_param at eps_min over the same clean-correct population. "
                    "Legacy all-sample values are stored as tau_param_all and icr_senn_all."
                    if icr_senn is not None
                    else "N/A — only computed for SENN models"
                ),
                # ---- mean internal metric curves ------------------------
                "mean_internal_metrics": mean_metric_curves,
                # ---- metadata -------------------------------------------
                "attack_params":          params,
                "monotonicity_violations": viol,
                "architecture":           (
                    "B30-Shield" if is_shield else (
                        "B30" if is_b30 else ("ProtoVAE" if is_protovae else "SENN")
                    )
                ),
                "notes": (
                    "per_example_metrics.npz contains (N,E) float32 matrices "
                    "for all internal metrics; per_example_correct.npz contains "
                    "the (N,E) uint8 accuracy matrix."
                ),
            }

            _save_json(os.path.join(run_dir, "metrics.json"), metrics_dict)
            print(f"Extended metrics.json saved to {os.path.join(run_dir, 'metrics.json')}")

            if icr is not None:
                print(f"  ICR (proto) = {icr:.4f}")
            if icr_senn is not None:
                print(f"  tau_param = {tau_param:.6f}")
                print(f"  ICR_SENN    = {icr_senn:.4f}")
            print(
                f"  Empirical robustness (failing samples only): "
                f"mean r- = {robustness_json['mean_r_minus_failing']:.4f}, "
                f"mean r+ = {rob_interval['mean_r_plus']:.4f}  "
                f"| never-fail fraction = {robustness_json['frac_never_fail']:.4f}"
            )
            if rpgd_bs_result is not None:
                print(f"  Binary-search r-PGD worst case = {rpgd_bs_result['min_r_pgd']:.4f}")
