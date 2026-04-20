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
    metrics.json              — extended with EarlyRate, empirical robustness
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
from datetime import datetime
from functools import partial
from xml.parsers.expat import model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

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
    compute_empirical_robustness_interval,
    compute_early_rate,
    compute_tau_param,
    compute_early_rate_senn,
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


def _init_metric_store(dim: int, is_b30: bool, is_protovae: bool, is_senn: bool) -> dict:
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

    if is_protovae:
        store["R_mu"]    = [[] for _ in range(dim)]
        store["R_sigma"] = [[] for _ in range(dim)]

    if is_senn:
        store["R_concept"] = [[] for _ in range(dim)]
        store["R_param"]   = [[] for _ in range(dim)]

    return store


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
    bootstrap_B: int   = 2000,
    bootstrap_alpha: float = 0.05,
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

    # -------------------------------------------------------------------------
    # Pre-compute architecture flags and prototype labels (once per model).
    # These never change across attacks or batches.
    # -------------------------------------------------------------------------
    arch_flags_per_model  = [_arch_flags(m)            for m in models]
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
            _init_metric_store(dim, *arch_flags_per_model[idm])
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
                proto_labels                 = proto_labels_per_model[idm]
                store                        = metric_store[idm]
                B                            = batch_x.shape[0]

                # ----------------------------------------------------------
                # CLEAN internals — computed ONCE per (batch, model).
                # These are reused as the reference for all drift metrics
                # (R_enc, R_dec, R_mu, R_sigma, R_concept, R_param).
                # ----------------------------------------------------------
                clean_int = extract_internals(model, batch_x)

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
                if is_protovae:
                    store["R_mu"][0].append(zeros_B.copy())
                    store["R_sigma"][0].append(zeros_B.copy())
                if is_senn:
                    store["R_concept"][0].append(zeros_B.copy())
                    store["R_param"][0].append(zeros_B.copy())

                # Prepare loss function for non-foolbox attacks (same as original).
                if not foolbox_use:
                    loss_f = partial(loss, model=model, batch_y=batch_y)

                # ==========================================================
                # Epsilon loop  (ide=0 is ε=0, already handled above)
                # ==========================================================
                for ide in range(1, dim):
                    eps = float(x_axis[ide])

                    # Generate adversarial example — identical branching to original.
                    if foolbox_use:
                        x_adv = attack(batch_x, batch_y, model=model, epsilon=eps)
                    else:
                        if attack_name == "PatchPGD_attack":
                            adv_attack = partial(attack, loss_f=loss_f, eps=eps, batch_idx=batch_idx)
                        else:
                            adv_attack = partial(attack, loss_f=loss_f, eps=eps)
                        x_adv = adv_attack(batch_x)

                    # ADV internals — computed ONCE per (batch, model, eps).
                    adv_int = extract_internals(model, x_adv)

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
            store = metric_store[idm]

            # ---- Accuracy matrix ----------------------------------------
            # X_correct : (N_test, dim)  bool/uint8
            cols     = [np.concatenate(per_example_correct[idm][j], axis=0) for j in range(dim)]
            X_correct = np.stack(cols, axis=1)          # (N, E)
            acc_curve = X_correct.mean(axis=0)          # (E,)

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

            # ---- Global metrics -----------------------------------------

            # Empirical robustness interval — uses the monotone-from-zero definition.
            rob_interval = compute_empirical_robustness_interval(
                X_correct.astype(bool), x_axis.astype(np.float64)
            )

            # EarlyRate — only meaningful for prototype-based architectures.
            early_rate: float | None = None
            if is_b30 or is_protovae:
                early_rate = compute_early_rate(M["m_proto"], M["m_pred"])
                
            # EarlyRate_SENN and tau_param — SENN only.
            # tau_param is computed from the R_param distribution at the first
            # non-zero epsilon (column 1), which defines the empirical noise floor
            # of the Parameterizer under minimal adversarial pressure.
            tau_param:       float | None = None
            early_rate_senn: float | None = None
            if is_senn:
                tau_param       = compute_tau_param(M["R_param"], eps_idx=1)
                early_rate_senn = compute_early_rate_senn(
                    M["R_param"], M["m_pred"], tau_param
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
                params = attack_params_map.get(attack_name, {})

            # Mean metric curves per epsilon (for quick inspection in JSON).
            mean_metric_curves: dict[str, list] = {
                name: mat.mean(axis=0).tolist() for name, mat in M.items()
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
                "empirical_robustness_interval": {
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
                # ---- EarlyRate (prototype architectures only) -----------
                "early_rate": early_rate,   # None for SENN
                "early_rate_definition": (
                    "(1/N) * sum_i max_{eps} I[m_proto(x_i,eps)<0 & m_pred(x_i,eps)>0]"
                    if early_rate is not None else "N/A — SENN has no prototypes"
                ),
                # ---- EarlyRate_SENN (SENN only) -------------------------
                "tau_param": tau_param,         # None for B30/ProtoVAE
                "early_rate_senn": early_rate_senn,  # None for B30/ProtoVAE
                "early_rate_senn_definition": (
                    f"(1/N) * sum_i max_{{eps}} I[R_param(x_i,eps)>tau({tau_param:.6f}) & m_pred(x_i,eps)>0]; "
                    "tau = p95 of R_param at eps_min (first non-zero epsilon)"
                    if early_rate_senn is not None
                    else "N/A — only computed for SENN models"
                ),
                # ---- mean internal metric curves ------------------------
                "mean_internal_metrics": mean_metric_curves,
                # ---- metadata -------------------------------------------
                "attack_params":          params,
                "monotonicity_violations": viol,
                "architecture":           (
                    "B30" if is_b30 else ("ProtoVAE" if is_protovae else "SENN")
                ),
                "notes": (
                    "per_example_metrics.npz contains (N,E) float32 matrices "
                    "for all internal metrics; per_example_correct.npz contains "
                    "the (N,E) uint8 accuracy matrix."
                ),
            }

            _save_json(os.path.join(run_dir, "metrics.json"), metrics_dict)
            print(f"Extended metrics.json saved to {os.path.join(run_dir, 'metrics.json')}")

            if early_rate is not None:
                print(f"  EarlyRate (proto) = {early_rate:.4f}")
            if early_rate_senn is not None:
                print(f"  tau_param = {tau_param:.6f}")
                print(f"  EarlyRate_SENN    = {early_rate_senn:.4f}")
            print(
                f"  Empirical robustness (failing samples only): "
                f"mean r- = {rob_interval['mean_r_minus_failing']:.4f}, "
                f"mean r+ = {rob_interval['mean_r_plus']:.4f}  "
                f"| never-fail fraction = {rob_interval['frac_never_fail']:.4f}"
            )
