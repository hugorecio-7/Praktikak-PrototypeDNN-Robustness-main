"""
ch4_diagnosis.py
================
Chapter 4 — RQ1: Diagnosing fragility in the clean B30 model.

What this script produces
--------------------------
Figure 1  (ch4_degradation_panels.pdf)
    Three-panel degradation plot for the clean B30 model under PGD:
      Panel A — Semantic:  m_proto and m_pred vs epsilon  (y ∈ [-1, 1])
      Panel B — Latent:    R_enc vs epsilon               (y ≥ 0, ratio scale)
      Panel C — Visual:    R_dec vs epsilon               (y ≥ 0, L2 error)

Figure 2  (ch4_rPGD_histogram.pdf)
    Histogram of the per-sample lower robustness bound r_PGD- across the
    test set, with the global mean E[r_PGD-] marked.

Console output
--------------
    EarlyRate, frac_never_fail, E[r_PGD-], E[r_PGD+] for the clean model.

Usage
-----
    python ch4_diagnosis.py
    python ch4_diagnosis.py --model B30 --attack PGDLInf_attack --seed 1
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    RC_PARAMS, METRIC_LABELS, FIGURES_ROOT, EPS_GRID, EPS_REF,
    DEFAULT_SEED, DEFAULT_RUN_ID,
)
from loaders import (
    load_json, load_npz, load_correct,
    get_acc_curve, get_mean_metric_curve, get_robustness_interval,
)


# =============================================================================
# Helpers
# =============================================================================

def _apply_style() -> None:
    plt.rcParams.update(RC_PARAMS)


def _save(fig: plt.Figure, name: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# =============================================================================
# Figure 1 — Degradation panels
# =============================================================================

def plot_degradation_panels(
    data_json: dict,
    data_npz:  dict,
    model_name: str,
    out_dir: str,
) -> None:
    """
    Three-panel figure isolating the three degradation scales:
      A) Semantic margins  (m_proto, m_pred)  — y ∈ [-1, 1]
      B) Latent drift      (R_enc)            — y ≥ 0, ratio
      C) Visual drift      (R_dec)            — y ≥ 0, L2
    """
    eps = np.array(data_json["eps"])

    # Mean curves from json (avoids loading large npz columns just for means).
    m_proto = get_mean_metric_curve(data_json, "m_proto")
    m_pred  = get_mean_metric_curve(data_json, "m_pred")
    R_enc   = get_mean_metric_curve(data_json, "R_enc")
    R_dec   = get_mean_metric_curve(data_json, "R_dec")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # --- Panel A: Semantic ---
    ax = axes[0]
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.plot(eps, m_proto, color="#d62728", linewidth=2.0, label=r"$\tilde{m}_{proto}$")
    ax.plot(eps, m_pred,  color="#1f77b4", linewidth=2.0, label=r"$\tilde{m}_{pred}$")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Perturbation $\varepsilon$")
    ax.set_ylabel("Margin")
    ax.set_title("A — Semantic Margins")
    ax.legend()
    ax.axvline(EPS_REF, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)

    # --- Panel B: Latent drift ---
    ax = axes[1]
    ax.plot(eps, R_enc, color="#ff7f0e", linewidth=2.0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Perturbation $\varepsilon$")
    ax.set_ylabel(r"$R_{enc}$ (relative drift)")
    ax.set_title("B — Latent Drift")
    ax.axvline(EPS_REF, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)

    # --- Panel C: Visual drift ---
    ax = axes[2]
    ax.plot(eps, R_dec, color="#2ca02c", linewidth=2.0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Perturbation $\varepsilon$")
    ax.set_ylabel(r"$R_{dec}$ (L$_2$ error, $[0,1]$ space)")
    ax.set_title("C — Visual Drift")
    ax.axvline(EPS_REF, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)

    fig.suptitle(
        f"Degradation profile — {model_name} (clean model, PGD $L_\\infty$)",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    _save(fig, "ch4_degradation_panels.pdf", out_dir)


# =============================================================================
# Figure 2 — r_PGD- histogram
# =============================================================================

def plot_rpgd_histogram(
    data_json: dict,
    data_npz:  dict,
    data_correct: dict,
    model_name: str,
    out_dir: str,
) -> None:
    """
    Histogram of per-sample r_PGD- (lower robustness bound).

    r_PGD-(x) = max { eps in E | model correct at all eps' <= eps }
    Samples that never fail are placed at eps_max.
    """
    rob = get_robustness_interval(data_json)
    hist_data = rob["hist_r_minus"]
    counts    = np.array(hist_data["counts"])
    edges     = np.array(hist_data["bin_edges"])
    mean_all  = rob["mean_r_minus_all"]
    frac_never = rob["frac_never_fail"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        edges[:-1], counts,
        width=np.diff(edges),
        align="edge",
        color="#1f77b4", alpha=0.75, edgecolor="white",
    )
    ax.axvline(
        mean_all, color="#d62728", linewidth=2.0, linestyle="--",
        label=rf"$E[r_{{PGD}}^-]$ = {mean_all:.3f}",
    )
    ax.set_xlabel(r"$r_{PGD}^-$ (lower robustness bound per sample)")
    ax.set_ylabel("Sample count")
    ax.set_title(
        rf"Distribution of $r_{{PGD}}^-$ — {model_name}"
        f"\n(never-fail fraction = {frac_never:.1%})"
    )
    ax.legend()
    plt.tight_layout()
    _save(fig, "ch4_rPGD_histogram.pdf", out_dir)


# =============================================================================
# Console summary
# =============================================================================

def print_summary(data_json: dict) -> None:
    rob  = get_robustness_interval(data_json)
    er   = data_json.get("early_rate")
    arch = data_json.get("architecture", "?")

    print("\n" + "=" * 55)
    print(f"  Chapter 4 — Diagnostic Summary")
    print(f"  Model      : {data_json['model']}  [{arch}]")
    print(f"  Attack     : {data_json['attack']}")
    print("=" * 55)
    print(f"  EarlyRate              : {er:.4f}" if er is not None else "  EarlyRate              : N/A")
    print(f"  frac_never_fail        : {rob['frac_never_fail']:.4f}")
    print(f"  E[r_PGD-] (all)        : {rob['mean_r_minus_all']:.4f}")
    print(f"  E[r_PGD-] (failing)    : {rob['mean_r_minus_failing']:.4f}")
    print(f"  E[r_PGD+] (failing)    : {rob['mean_r_plus']:.4f}")
    print("=" * 55 + "\n")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 4 — B30 clean model diagnosis.")
    parser.add_argument("--model",  type=str, default="B30",             help="Model name (default: B30)")
    parser.add_argument("--attack", type=str, default="PGDLInf_attack",  help="Attack name")
    parser.add_argument("--seed",   type=int, default=DEFAULT_SEED)
    parser.add_argument("--run_id", type=str, default=DEFAULT_RUN_ID)
    args = parser.parse_args()

    out_dir = os.path.join(FIGURES_ROOT, "ch4")
    _apply_style()

    print(f"\nLoading data for {args.model} / {args.attack} / seed={args.seed} …")
    data_json    = load_json(args.model,    args.attack, args.seed, args.run_id)
    data_npz     = load_npz(args.model,     args.attack, args.seed, args.run_id)
    data_correct = load_correct(args.model, args.attack, args.seed, args.run_id)

    print_summary(data_json)

    print("Generating Figure 1 — degradation panels …")
    plot_degradation_panels(data_json, data_npz, args.model, out_dir)

    print("Generating Figure 2 — r_PGD- histogram …")
    plot_rpgd_histogram(data_json, data_npz, data_correct, args.model, out_dir)

    print("\nDone. All figures saved to:", out_dir)


if __name__ == "__main__":
    main()
