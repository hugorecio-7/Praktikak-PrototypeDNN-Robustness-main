"""
ch5_ch7_ablation.py
===================
Chapters 5 & 7 — Ablation analysis for B30, ProtoVAE, SENN (and ALL).

What this script produces
--------------------------
For --arch B30 | ProtoVAE | SENN
  Figure 1  ({arch}_accuracy_curves.pdf)
      Accuracy vs epsilon curves, one line per variant, superimposed.
  Figure 2  ({arch}_metric_curves.pdf)
      Mean metric curves vs epsilon for the key metrics of the architecture,
      one subplot per metric, one line per variant.
      If --metrics is passed, also saves one separate PDF per metric.
  Table     ({arch}_ablation_table.csv  +  {arch}_ablation_table.tex)
      Master ablation table with, per variant:
        - Frozen component
        - Clean accuracy  (acc at eps=0)
        - Adv accuracy    (acc at EPS_REF)
        - Δm_proto @ EPS_REF   (vs base model; N/A for SENN)
        - ΔR_enc   @ EPS_REF   (or ΔR_param for SENN)
        - ICR                  (or ICR_SENN for SENN)
        - frac_never_fail
        - E[r_PGD-]  (all-sample mean)

For --arch ALL
  Figure 3  (crossarch_r_minus_barplot.pdf)
      Grouped bar chart of E[r_PGD-] per variant and architecture.
  Table     (crossarch_ablation_table.csv  +  .tex)
      Unified table across all three architectures.

Usage
-----
    python ch5_ch7_ablation.py --arch B30
    python ch5_ch7_ablation.py --arch ProtoVAE
    python ch5_ch7_ablation.py --arch SENN
    python ch5_ch7_ablation.py --arch ALL
    python ch5_ch7_ablation.py --arch B30 --attack AutoAttack_adv
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    RC_PARAMS, FIGURES_ROOT, VARIANTS, VARIANT_LABELS, VARIANT_COLORS, METRIC_LABELS,
    ARCH_COLORS, FREEZE_DESCRIPTION, METRICS_BY_ARCH,
    EPS_GRID, EPS_REF, DEFAULT_SEED, DEFAULT_RUN_ID,
)
from loaders import (
    load_all_variants, get_acc_curve, get_mean_metric_curve,
    get_scalar_at_eps,
    get_rpgd_mean, get_rpgd_min,
)


# =============================================================================
# Internal helpers
# =============================================================================

def _apply_style() -> None:
    plt.rcParams.update(RC_PARAMS)


def _save(fig: plt.Figure, name: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


def _save_table(df: pd.DataFrame, name_stem: str, out_dir: str) -> None:
    """Save table as CSV and LaTeX."""
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{name_stem}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path}")

    tex_path = os.path.join(out_dir, f"{name_stem}.tex")
    latex_str = df.to_latex(
        index=False,
        escape=False,          # allow LaTeX math in column headers
        float_format="%.4f",
        na_rep="—",
        caption=f"Ablation results — {name_stem.replace('_', ' ')}",
        label=f"tab:{name_stem}",
    )
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"  Saved → {tex_path}")


def _arch_of(model_name: str) -> str:
    """Infer architecture string from model name."""
    if model_name.startswith("ProtoVAE"):
        return "ProtoVAE"
    if model_name.startswith("SENN"):
        return "SENN"
    return "B30"


def _plot_data_with_optional_ft_em(all_data: dict[str, dict], include_ft_em: bool) -> dict[str, dict]:
    """Return the subset of models that should appear in plots."""
    if include_ft_em:
        return all_data
    return {name: data for name, data in all_data.items() if name != "B30-FT-E-M"}


# =============================================================================
# Figure 1 — Accuracy curves
# =============================================================================

def plot_accuracy_curves(
    all_data: dict[str, dict],
    arch: str,
    out_dir: str,
) -> None:
    """
    Accuracy vs epsilon, one line per variant.
    Base model drawn with dashed line; FT variants with solid lines.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, data in all_data.items():
        eps, acc = get_acc_curve(data["json"])
        label     = VARIANT_LABELS.get(model_name, model_name)
        color     = VARIANT_COLORS.get(model_name, "#888888")
        is_base   = label == "Base"
        ax.plot(
            eps, acc,
            color=color, linewidth=2.2 if not is_base else 1.5,
            linestyle="--" if is_base else "-",
            label=label,
        )

    ax.axvline(EPS_REF, color="gray", linewidth=0.8, linestyle=":", alpha=0.7,
               label=rf"$\varepsilon_{{ref}}={EPS_REF}$")
    ax.set_xlabel(r"Perturbation $\varepsilon$")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(f"Adversarial accuracy — {arch} ablation (PGD $L_\\infty$)")
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    _save(fig, f"{arch}_accuracy_curves.pdf", out_dir)


# =============================================================================
# Figure 2 — Metric curves per variant
# =============================================================================

def plot_metric_curves(
    all_data: dict[str, dict],
    arch: str,
    out_dir: str,
) -> None:
    """
    One subplot per metric (key metrics for the architecture),
    one line per variant.
    """
    metrics = METRICS_BY_ARCH[arch]
    n_metrics = len(metrics)
    ncols = 2
    nrows = (n_metrics + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for ax_idx, metric in enumerate(metrics):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]

        for model_name, data in all_data.items():
            try:
                curve = get_mean_metric_curve(data["json"], metric)
            except KeyError:
                continue
            eps     = np.array(data["json"]["eps"])
            label   = VARIANT_LABELS.get(model_name, model_name)
            color   = VARIANT_COLORS.get(model_name, "#888888")
            is_base = label == "Base"
            ax.plot(
                eps, curve,
                color=color, linewidth=2.0 if not is_base else 1.4,
                linestyle="--" if is_base else "-",
                label=label,
            )

        # Horizontal zero line for margin metrics
        if metric in ("m_proto", "m_pred"):
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
            ax.set_ylim(-1.05, 1.05)

        ax.set_ylim(bottom=0) if metric.startswith("R_") else None
        ax.axvline(EPS_REF, color="gray", linewidth=0.7, linestyle=":", alpha=0.6)
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_title(_metric_title(metric))
        ax.legend(fontsize=7, framealpha=0.85)

    # Hide unused subplots
    for ax_idx in range(n_metrics, nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(f"Internal metric curves — {arch} ablation", fontsize=13)
    plt.tight_layout()
    _save(fig, f"{arch}_metric_curves.pdf", out_dir)


def plot_metric_curves_separate(
    all_data: dict[str, dict],
    arch: str,
    out_dir: str,
) -> None:
    """
    Save one figure per metric, keeping the same styling as the combined plot.
    """
    for metric in METRICS_BY_ARCH[arch]:
        fig, ax = plt.subplots(figsize=(8, 5))

        plotted_any = False
        for model_name, data in all_data.items():
            try:
                curve = get_mean_metric_curve(data["json"], metric)
            except KeyError:
                continue

            eps = np.array(data["json"]["eps"])
            label = VARIANT_LABELS.get(model_name, model_name)
            color = VARIANT_COLORS.get(model_name, "#888888")
            is_base = label == "Base"

            ax.plot(
                eps,
                curve,
                color=color,
                linewidth=2.0 if not is_base else 1.4,
                linestyle="--" if is_base else "-",
                label=label,
            )
            plotted_any = True

        if metric in ("m_proto", "m_pred"):
            ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
            ax.set_ylim(-1.05, 1.05)
        elif metric.startswith("R_"):
            ax.set_ylim(bottom=0)

        ax.axvline(EPS_REF, color="gray", linewidth=0.7, linestyle=":", alpha=0.6)
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(f"{METRIC_LABELS.get(metric, metric)} — {arch} ablation")

        if plotted_any:
            ax.legend(fontsize=7, framealpha=0.85)
        else:
            ax.text(
                0.5,
                0.5,
                "Metric not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        plt.tight_layout()
        _save(fig, f"{arch}_{metric}_curve.pdf", out_dir)


def _metric_title(metric: str) -> str:
    titles = {
        "m_proto":   r"Prototype Margin $\tilde{m}_{proto}$",
        "m_pred":    r"Prediction Margin $\tilde{m}_{pred}$",
        "R_enc":     r"Encoder Drift $R_{enc}$",
        "R_dec":     r"Decoder Drift $R_{dec}$",
        "R_mu":      r"Mean Drift $R_{\mu}$",
        "R_sigma":   r"Uncertainty Drift $R_{\sigma}$",
        "R_concept": r"Concept Drift $R_{concept}$",
        "R_param":   r"Parameterizer Drift $R_{param}$",
    }
    return titles.get(metric, metric)


# =============================================================================
# Table builder — single architecture
# =============================================================================

def build_ablation_table(
    all_data: dict[str, dict],
    arch: str,
    base_model_name: str,
) -> pd.DataFrame:
    """
    Build master ablation table for one architecture.

    Columns depend on architecture:
    All     : Variant, Frozen, Clean Acc, Adv Acc, ICR*, E[r-]
      B30/VAE : + Δm_proto@EPS_REF,  ΔR_enc@EPS_REF
      SENN    : + ΔR_param@EPS_REF   (replaces Δm_proto + ΔR_enc)

    Δ columns = mean(metric @ EPS_REF, variant) − mean(metric @ EPS_REF, base).
    """
    eps_array = np.array(
        next(iter(all_data.values()))["json"]["eps"], dtype=np.float32
    )

    # Pre-compute base model metric values at EPS_REF for delta calculation.
    base_data = all_data.get(base_model_name, {})
    base_npz  = base_data.get("npz", {})

    def _base_mean(metric: str) -> float:
        if metric not in base_npz:
            return float("nan")
        return float(get_scalar_at_eps(base_npz, metric, eps_array, EPS_REF).mean())

    if arch in ("B30", "ProtoVAE"):
        base_m_proto = _base_mean("m_proto")
        base_R_enc   = _base_mean("R_enc")
    else:  # SENN
        base_R_param = _base_mean("R_param")

    rows = []
    for model_name, data in all_data.items():
        j   = data["json"]
        npz = data["npz"]
        eps, acc = get_acc_curve(j)
        eps_idx_0   = 0
        eps_idx_ref = int(np.argmin(np.abs(eps - EPS_REF)))

        clean_acc = float(acc[eps_idx_0])
        adv_acc   = float(acc[eps_idx_ref])

        # ICR — architecture-aware
        if arch == "SENN":
            icr = j.get("icr_senn")
        else:
            icr = j.get("icr")

        row: dict = {
            "Variant":          VARIANT_LABELS.get(model_name, model_name),
            "Frozen":           FREEZE_DESCRIPTION.get(model_name, "—"),
            "Clean Acc":        round(clean_acc, 4),
            f"Adv Acc ({EPS_REF})": round(adv_acc, 4),
        }

        if arch in ("B30", "ProtoVAE"):
            m_proto_mean = float(get_scalar_at_eps(npz, "m_proto", eps_array, EPS_REF).mean()) \
                           if "m_proto" in npz else float("nan")
            R_enc_mean   = float(get_scalar_at_eps(npz, "R_enc",   eps_array, EPS_REF).mean()) \
                           if "R_enc"   in npz else float("nan")
            row[r"$\Delta\tilde{m}_{proto}$"] = round(m_proto_mean - base_m_proto, 4)
            row[r"$\Delta R_{enc}$"]           = round(R_enc_mean   - base_R_enc,   4)

        elif arch == "SENN":
            R_param_mean = float(get_scalar_at_eps(npz, "R_param", eps_array, EPS_REF).mean()) \
                           if "R_param" in npz else float("nan")
            row[r"$\Delta R_{param}$"] = round(R_param_mean - base_R_param, 4)

        row["ICR"]                 = round(icr, 4) if icr is not None else float("nan")
        row[r"$E[r_{PGD}^-]$"]     = round(get_rpgd_mean(j), 4)
        #row[r"$\min(r_{PGD}^-)$"]  = round(get_rpgd_min(j), 6)

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Figure 3 + Table — cross-architecture (--arch ALL)
# =============================================================================

def plot_crossarch_barplot(
    all_arch_data: dict[str, dict[str, dict]],
    out_dir: str,
) -> None:
    """
    Grouped bar chart of E[r_PGD-] per variant, grouped by architecture.
    Only FT variants are shown (base model excluded to keep it readable).
    """
    archs = [a for a in ("B30", "ProtoVAE", "SENN") if a in all_arch_data]

    # Collect variant short labels common across architectures
    # (show only FT-0..FT-E, not Base)
    ft_labels_per_arch: dict[str, list[str]] = {}
    r_minus_per_arch:   dict[str, list[float]] = {}

    for arch in archs:
        labels, values = [], []
        for model_name, data in all_arch_data[arch].items():
            lbl = VARIANT_LABELS.get(model_name, model_name)
            if lbl == "Base":
                continue
            labels.append(lbl)
            values.append(get_rpgd_mean(data["json"]))
        ft_labels_per_arch[arch] = labels
        r_minus_per_arch[arch]   = values

    # Align on the union of all FT labels
    all_labels = []
    for labels in ft_labels_per_arch.values():
        for lbl in labels:
            if lbl not in all_labels:
                all_labels.append(lbl)

    n_groups  = len(all_labels)
    n_archs   = len(archs)
    bar_width = 0.22
    x         = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.6), 5))

    for i, arch in enumerate(archs):
        label_map = dict(zip(ft_labels_per_arch[arch], r_minus_per_arch[arch]))
        heights   = [label_map.get(lbl, float("nan")) for lbl in all_labels]
        offset    = (i - (n_archs - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset, heights,
            width=bar_width,
            color=ARCH_COLORS.get(arch, "#888888"),
            alpha=0.85,
            label=arch,
            edgecolor="white",
        )
        # Value annotations on bars
        for bar, h in zip(bars, heights):
            if not np.isnan(h):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, h + 0.002,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels)
    ax.set_xlabel("Fine-tuning variant")
    ax.set_ylabel(r"$E[r_{PGD}^-]$ (global safe radius)")
    ax.set_title(
        r"Cross-architecture comparison of $E[r_{PGD}^-]$"
        "\n(higher = more robust)"
    )
    ax.legend(title="Architecture")
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    _save(fig, "crossarch_r_minus_barplot.pdf", out_dir)


def build_crossarch_table(
    all_arch_data: dict[str, dict[str, dict]],
    eps_ref: float = EPS_REF,
) -> pd.DataFrame:
    """
    Unified table across all architectures.
    ICR column adapts: uses ICR_SENN for SENN rows.
    """
    rows = []
    for arch, data_dict in all_arch_data.items():
        eps_array = np.array(
            next(iter(data_dict.values()))["json"]["eps"], dtype=np.float32
        )
        eps_idx_ref = int(np.argmin(np.abs(eps_array - eps_ref)))

        for model_name, data in data_dict.items():
            j   = data["json"]
            eps, acc = get_acc_curve(j)

            icr = j.get("icr_senn") if arch == "SENN" else j.get("icr")

            rows.append({
                "Arch":              arch,
                "Variant":           VARIANT_LABELS.get(model_name, model_name),
                "Frozen":            FREEZE_DESCRIPTION.get(model_name, "—"),
                "Clean Acc":         round(float(acc[0]),              4),
                f"Adv Acc ({eps_ref})": round(float(acc[eps_idx_ref]), 4),
                "ICR":                  round(icr, 4) if icr is not None else float("nan"),
                r"$E[r_{PGD}^-]$":      round(get_rpgd_mean(j), 4),
                #r"$\min(r_{PGD}^-)$":   round(get_rpgd_min(j), 6),
            })

    return pd.DataFrame(rows)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapters 5 & 7 — ablation tables and plots."
    )
    parser.add_argument(
        "--arch", type=str, required=True,
        choices=["B30", "ProtoVAE", "SENN", "ALL"],
        help="Architecture to analyse, or ALL for cross-architecture comparison.",
    )
    parser.add_argument("--attack", type=str, default="PGDLInf_attack")
    parser.add_argument("--seed",   type=int, default=DEFAULT_SEED)
    parser.add_argument("--run_id", type=str, default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="Also save one separate PDF per internal metric, in addition to the combined metric figure.",
    )
    parser.add_argument(
        "--ft-em",
        action="store_true",
        help="Include FT-E-M in plots. Tables always include it when available.",
    )
    args = parser.parse_args()

    _apply_style()

    archs_to_run = ["B30", "ProtoVAE", "SENN"] if args.arch == "ALL" else [args.arch]

    # Base model name per architecture (first entry in VARIANTS)
    base_names = {arch: VARIANTS[arch][0] for arch in ("B30", "ProtoVAE", "SENN")}

    all_arch_data: dict[str, dict[str, dict]] = {}

    for arch in archs_to_run:
        out_dir = os.path.join(FIGURES_ROOT, "ch5" if arch == "B30" else "ch7")
        print(f"\n{'='*55}")
        print(f"  Architecture: {arch}")
        print(f"{'='*55}")

        print("  Loading variants …")
        arch_data = load_all_variants(arch, args.attack, args.seed, args.run_id)
        all_arch_data[arch] = arch_data

        plot_data = _plot_data_with_optional_ft_em(arch_data, args.ft_em)

        if args.arch != "ALL":
            # Per-architecture figures and table
            print("  Generating accuracy curves …")
            plot_accuracy_curves(plot_data, arch, out_dir)

            print("  Generating metric curves …")
            plot_metric_curves(plot_data, arch, out_dir)

            if args.metrics:
                print("  Generating separate metric curves …")
                plot_metric_curves_separate(plot_data, arch, out_dir)

            print("  Building ablation table …")
            df = build_ablation_table(arch_data, arch, base_names[arch])
            print(df.to_string(index=False))
            _save_table(df, f"{arch}_ablation_table", out_dir)

    if args.arch == "ALL":
        out_dir = os.path.join(FIGURES_ROOT, "ch7")
        print(f"\n{'='*55}")
        print("  Cross-architecture analysis (--arch ALL)")
        print(f"{'='*55}")

        print("  Generating grouped bar chart …")
        plot_crossarch_barplot(all_arch_data, out_dir)

        print("  Building unified table …")
        df_all = build_crossarch_table(all_arch_data)
        print(df_all.to_string(index=False))
        _save_table(df_all, "crossarch_ablation_table", out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
