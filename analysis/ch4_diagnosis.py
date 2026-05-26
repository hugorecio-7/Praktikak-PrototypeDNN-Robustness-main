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
    ICR, E[r_PGD-], E[r_PGD+] for the clean model.

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
from matplotlib.ticker import FormatStrFormatter

from config import (
    RC_PARAMS, METRIC_LABELS, FIGURES_ROOT, EPS_REF,
    DEFAULT_SEED, DEFAULT_RUN_ID,
)
from loaders import (
    load_json, load_npz, load_correct, load_rpgd_binary_search,
    get_acc_curve, get_mean_metric_curve, get_robustness_interval,
    get_rpgd_min,
)


def _fmt_eps(value: float) -> str:
    """Format epsilon values without hiding very small non-zero radii."""
    if np.isnan(value):
        return "N/A"
    if abs(value) < 1e-3:
        return f"{value:.6f}"
    return f"{value:.4f}"


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
    
def _format_epsilon_axis(ax: plt.Axes, eps: np.ndarray) -> None:
    """Force epsilon ticks to be 0.0, 0.1, ..., eps_max."""
    eps_min = float(np.nanmin(eps))
    eps_max = float(np.nanmax(eps))

    ticks = np.round(np.arange(eps_min, eps_max + 1e-9, 0.1), 1)

    ax.set_xlim(eps_min, eps_max)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))


# =============================================================================
# Figure 1 — Degradation panels
# =============================================================================

def plot_degradation_panels(
    data_json: dict,
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
    
def plot_degradation_panels_separate(
    data_json: dict,
    model_name: str,
    out_dir: str,
) -> None:
    """
    Same content as Figure 1, but saved as three independent figures.
    Useful for LaTeX subfigure/subcaption layouts.
    """
    eps = np.array(data_json["eps"])

    m_proto = get_mean_metric_curve(data_json, "m_proto")
    m_pred  = get_mean_metric_curve(data_json, "m_pred")
    R_enc   = get_mean_metric_curve(data_json, "R_enc")
    R_dec   = get_mean_metric_curve(data_json, "R_dec")

    # --- Panel A: Semantic ---
    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.plot(eps, m_proto, color="#d62728", linewidth=2.0, label=r"$\tilde{m}_{proto}$")
    ax.plot(eps, m_pred,  color="#1f77b4", linewidth=2.0, label=r"$\tilde{m}_{pred}$")
    _format_epsilon_axis(ax, eps)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel(r"Perturbation $\varepsilon$")
    ax.set_ylabel("Margin")
    ax.set_title("A — Semantic Margins")
    ax.legend()
    ax.axvline(EPS_REF, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    plt.tight_layout()
    _save(fig, "ch4_degradation_panel_A_semantic.pdf", out_dir)

    # --- Panel B: Latent drift ---
    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    ax.plot(eps, R_enc, color="#ff7f0e", linewidth=2.0)
    _format_epsilon_axis(ax, eps)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Perturbation $\varepsilon$")
    ax.set_ylabel(r"$R_{enc}$ (relative drift)")
    ax.set_title("B — Latent Drift")
    ax.axvline(EPS_REF, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    plt.tight_layout()
    _save(fig, "ch4_degradation_panel_B_latent.pdf", out_dir)

    # --- Panel C: Visual drift ---
    fig, ax = plt.subplots(figsize=(4.3, 3.6))
    ax.plot(eps, R_dec, color="#2ca02c", linewidth=2.0)
    _format_epsilon_axis(ax, eps)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(r"Perturbation $\varepsilon$")
    ax.set_ylabel(r"$R_{dec}$ (L$_2$ error, $[0,1]$ space)")
    ax.set_title("C — Visual Drift")
    ax.axvline(EPS_REF, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)
    plt.tight_layout()
    _save(fig, "ch4_degradation_panel_C_visual.pdf", out_dir)


# =============================================================================
# Figure 2 — r_PGD- histogram
# =============================================================================

def plot_rpgd_histogram(
    data_json: dict,
    model_name: str,
    out_dir: str,
    rpgd_data: dict[str, np.ndarray] | None = None,
) -> None:
    """
    Histogram of per-sample r_PGD- (lower robustness bound).

    r_PGD-(x) = max { eps in E | model correct at all eps' <= eps }
    Samples that never fail are placed at eps_max.
    """
    rob = get_robustness_interval(data_json)
    mean_all  = rob["mean_r_minus_all"]
    min_all   = get_rpgd_min(data_json)

    fig, ax = plt.subplots(figsize=(7, 4))

    values = None
    if rpgd_data is not None and "r_pgd" in rpgd_data:
        values = np.asarray(rpgd_data["r_pgd"], dtype=np.float64)
        values = values[np.isfinite(values)]

    if values is not None and values.size > 0:
        data_min = float(values.min())
        data_max = float(values.max())
        if data_min == data_max:
            pad = max(data_min * 0.05, 1e-4)
            hist_range = (max(0.0, data_min - pad), data_max + pad)
        else:
            pad = 0.03 * (data_max - data_min)
            hist_range = (max(0.0, data_min - pad), data_max + pad)

        n_bins = int(np.clip(np.sqrt(values.size), 12, 45))
        ax.hist(
            values,
            bins=n_bins,
            range=hist_range,
            color="#1f77b4",
            alpha=0.75,
            edgecolor="white",
        )
        ax.set_xlim(hist_range)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    else:
        hist_data = rob["hist_r_minus"]
        counts    = np.array(hist_data["counts"])
        edges     = np.array(hist_data["bin_edges"])
        ax.bar(
            edges[:-1], counts,
            width=np.diff(edges),
            align="edge",
            color="#1f77b4", alpha=0.75, edgecolor="white",
        )

    ax.axvline(
        mean_all, color="#d62728", linewidth=2.0, linestyle="--",
        label=rf"$\overline{{r}}_{{\mathrm{{PGD}}}}$ = {_fmt_eps(mean_all)}",
    )
    if not np.isnan(min_all):
        ax.axvline(
            min_all, color="#222222", linewidth=1.6, linestyle=":",
            label=rf"$\min(r_{{\mathrm{{PGD}}}})$ = {_fmt_eps(min_all)}",
        )
    ax.set_xlabel(r"$r_{\mathrm{PGD}}$ (lower robustness bound per sample)")
    ax.set_ylabel("Sample count")
    ax.set_title(
        rf"Distribution of $r_{{\mathrm{{PGD}}}}$ — {model_name}"
    )
    ax.legend()
    plt.tight_layout()
    _save(fig, "ch4_rPGD_histogram.pdf", out_dir)


# =============================================================================
# Console summary
# =============================================================================

def print_summary(data_json: dict) -> None:
    rob  = get_robustness_interval(data_json)
    icr  = data_json.get("icr")
    arch = data_json.get("architecture", "?")

    print("\n" + "=" * 55)
    print(f"  Chapter 4 — Diagnostic Summary")
    print(f"  Model      : {data_json['model']}  [{arch}]")
    print(f"  Attack     : {data_json['attack']}")
    print("=" * 55)
    print(f"  ICR                    : {icr:.4f}" if icr is not None else "  ICR                    : N/A")
    print(f"  E[r_PGD-] (all)        : {rob['mean_r_minus_all']:.4f}")
    print(f"  E[r_PGD-] (failing)    : {rob['mean_r_minus_failing']:.4f}")
    min_rpgd = get_rpgd_min(data_json)
    if not np.isnan(min_rpgd):
        print(f"  min(r_PGD-)            : {_fmt_eps(min_rpgd)}")
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
    try:
        data_rpgd = load_rpgd_binary_search(args.model, args.attack, args.seed, args.run_id)
    except FileNotFoundError:
        data_rpgd = None

    print_summary(data_json)

    print("Generating Figure 1 — degradation panels …")
    plot_degradation_panels(data_json, args.model, out_dir)
    
    print("Generating Figure 1b — separate degradation panels …")
    plot_degradation_panels_separate(data_json, args.model, out_dir)

    print("Generating Figure 2 — r_PGD- histogram …")
    plot_rpgd_histogram(data_json, args.model, out_dir, data_rpgd)

    print("\nDone. All figures saved to:", out_dir)


if __name__ == "__main__":
    main()
