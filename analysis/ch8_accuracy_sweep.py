"""
ch8_accuracy_sweep.py
=====================
Chapter 8 helper to plot adversarial accuracy or internal metrics vs epsilon
for selected B30 or ProtoVAE models.

The styling mirrors the accuracy plot in ch5_ch7_ablation.py, but this script
lets you choose exactly which models to include.

Examples
--------
    python analysis/ch8_accuracy_sweep.py --arch B30 --attack PGD
    python analysis/ch8_accuracy_sweep.py --arch B30 --attack PGD --internal_metrics
    python analysis/ch8_accuracy_sweep.py --arch ProtoVAE --attack AutoAttack --models ProtoVAE ProtoVAE-FT-E
    python analysis/ch8_accuracy_sweep.py --arch B30 --attack PGD --models B30 B30-FT-E B30-FT-E-M
    python analysis/ch8_accuracy_sweep.py --arch B30 --attack PGDLInf_attack --run_id ablation_B30_v1
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

from foolbox.utils import accuracy

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    DEFAULT_RUN_ID,
    DEFAULT_SEED,
    EPS_REF,
    FIGURES_ROOT,
    METRIC_LABELS,
    METRICS_BY_ARCH,
    RC_PARAMS,
    VARIANTS,
    VARIANT_COLORS,
    VARIANT_LABELS,
)
from loaders import get_acc_curve, get_mean_metric_curve, load_json


OUT_DIR = Path(FIGURES_ROOT) / "ch8"

ATTACK_ALIASES = {
    "pgd": "PGDLInf_attack",
    "pgdlinf": "PGDLInf_attack",
    "pgdlinf_attack": "PGDLInf_attack",
    "pgd_linf": "PGDLInf_attack",
    "autoattack": "AutoAttack_adv",
    "auto_attack": "AutoAttack_adv",
    "autoattack_adv": "AutoAttack_adv",
}


def _apply_style() -> None:
    plt.rcParams.update(RC_PARAMS)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def _resolve_attack(attack: str) -> str:
    """Accept friendly attack aliases while keeping raw run-folder names valid."""
    key = attack.strip().lower().replace("-", "_")
    return ATTACK_ALIASES.get(key, attack)


def _default_output_stem(
    arch: str,
    models: list[str],
    attack: str,
    plot_kind: str,
) -> str:
    if models == VARIANTS[arch]:
        model_part = "all"
    else:
        model_part = "_vs_".join(_slug(model) for model in models)
    return f"ch8_{arch}_{attack}_{model_part}_{plot_kind}"


def _resolve_models(arch: str, requested_models: list[str] | None) -> list[str]:
    known_models = VARIANTS[arch]
    if not requested_models:
        return known_models

    unknown = [model for model in requested_models if model not in known_models]
    if unknown:
        raise ValueError(
            f"Models {unknown} do not belong to arch '{arch}'. "
            f"Valid options: {known_models}"
        )

    return requested_models


def _load_accuracy_data(
    models: list[str],
    attack: str,
    seed: int,
    run_id: str | None,
) -> dict[str, dict]:
    data = {}
    for model_name in models:
        try:
            data[model_name] = load_json(model_name, attack=attack, seed=seed, run_id=run_id)
            print(f"  [OK] {model_name}")
        except FileNotFoundError as exc:
            print(f"  [SKIP] {model_name} - {exc}")

    if not data:
        raise RuntimeError(
            "No metrics.json files were loaded. Check --arch, --models, --attack, "
            "--seed and --run_id."
        )
    return data


def plot_accuracy_sweep(
    data_by_model: dict[str, dict],
    arch: str,
    attack: str,
    output_stem: str,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8, 5))

    for model_name, data_json in data_by_model.items():
        eps, acc = get_acc_curve(data_json)
        label = VARIANT_LABELS.get(model_name, model_name)
        color = VARIANT_COLORS.get(model_name, "#888888")
        is_base = label == "Base"

        ax.plot(
            eps,
            acc,
            color=color,
            linewidth=2.2 if not is_base else 1.5,
            linestyle="--" if is_base else "-",
            label=label,
        )

    ax.axvline(
        EPS_REF,
        color="gray",
        linewidth=0.8,
        linestyle=":",
        alpha=0.7,
        label=rf"$\varepsilon_{{ref}}={EPS_REF}$",
    )
    ax.set_xlabel(r"Perturbation $\varepsilon$")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.02, 1.02)
    if attack in ("PGDLInf_attack", "PGD"):
        ax.set_title(f"Adversarial accuracy - {arch} models (PGD $L_\\infty$)")
    else:
        ax.set_title(f"Adversarial accuracy - {arch} models ({attack})")
    ax.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{output_stem}.pdf"
    fig.savefig(path, bbox_inches="tight")
    saved_paths = [path]
    print(f"  Saved -> {path}")

    plt.close(fig)
    return saved_paths


def plot_internal_metric_sweep(
    data_by_model: dict[str, dict],
    arch: str,
    attack: str,
    output_stem: str,
) -> list[Path]:
    metrics = METRICS_BY_ARCH[arch]
    n_metrics = len(metrics)
    ncols = 2
    nrows = (n_metrics + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
    )

    for ax_idx, metric in enumerate(metrics):
        row, col = divmod(ax_idx, ncols)
        ax = axes[row][col]
        plotted_any = False

        for model_name, data_json in data_by_model.items():
            try:
                curve = get_mean_metric_curve(data_json, metric)
            except KeyError:
                continue

            eps = data_json["eps"]
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
        ax.set_title(METRIC_LABELS.get(metric, metric))

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

    for ax_idx in range(n_metrics, nrows * ncols):
        row, col = divmod(ax_idx, ncols)
        axes[row][col].set_visible(False)

    if attack in ("PGDLInf_attack", "PGD"):
        fig.suptitle(f"Internal metric curves - {arch} models (PGD $L_\\infty$)", fontsize=13)
    else:
        fig.suptitle(f"Internal metric curves - {arch} models ({attack})", fontsize=13)
    plt.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{output_stem}.pdf"
    fig.savefig(path, bbox_inches="tight")
    saved_paths = [path]
    print(f"  Saved -> {path}")

    plt.close(fig)
    return saved_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Plot accuracy or internal metrics vs epsilon for selected B30 or "
            "ProtoVAE models."
        )
    )
    parser.add_argument(
        "--arch",
        required=True,
        choices=["B30", "ProtoVAE"],
        help="Architecture family to plot.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model names from the selected architecture. Defaults to all variants.",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="PGD",
        help=(
            "Attack to plot. Friendly aliases: PGD -> PGDLInf_attack, "
            "AutoAttack -> AutoAttack_adv. Raw attack directory names also work."
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--run_id", type=str, default=DEFAULT_RUN_ID)
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename stem without extension. Defaults to an automatic descriptive name.",
    )
    parser.add_argument(
        "--internal_metrics",
        action="store_true",
        help="Generate only the internal metric curves PDF instead of the accuracy PDF.",
    )
    args = parser.parse_args()

    _apply_style()

    models = _resolve_models(args.arch, args.models)
    attack = _resolve_attack(args.attack)
    plot_kind = "internal_metric_curves" if args.internal_metrics else "accuracy_epsilon_sweep"
    output_stem = _slug(args.output) if args.output else _default_output_stem(
        args.arch, models, attack, plot_kind
    )

    print(f"\nArchitecture: {args.arch}")
    print(f"Attack:       {args.attack} -> {attack}" if args.attack != attack else f"Attack:       {attack}")
    print(f"Seed:         {args.seed}")
    print(f"Run id:       {args.run_id}")
    print(f"Models:       {', '.join(models)}")
    print(f"Plot:         {'internal metrics' if args.internal_metrics else 'accuracy'}")

    data_by_model = _load_accuracy_data(models, attack, args.seed, args.run_id)
    if args.internal_metrics:
        plot_internal_metric_sweep(data_by_model, args.arch, attack, output_stem)
    else:
        plot_accuracy_sweep(data_by_model, args.arch, attack, output_stem)

    print("\nDone.")


if __name__ == "__main__":
    main()
