"""
ch6_pca_training.py
===================
Chapter 6.2 — Latent space evolution during fine-tuning (training time).

What this script produces
--------------------------
CLEAN MODE (default):
    Figure  (ch6_pca_training_{variant}.pdf)
    Frames arranged in a grid (max 3 columns), one subplot per epoch.
    Each subplot shows the PCA projection of z(x) for a fixed test subset,
    coloured by ground-truth class, with prototype positions overlaid as
    larger star markers with black edges.
    PCA is fitted on the FINAL epoch's latent codes and reused for all frames.

ADVERSARIAL MODE (--show_adv):
    One figure per epsilon in --eps_list:
        ch6_pca_training_{variant}_adv_eps{eps}.pdf
    Same grid layout and same PCA as the clean figure, but the test samples
    are first attacked with PGD before extracting z. Prototypes are overlaid
    at their fixed positions (parameters — they do not move under attack).
    Direct visual evidence of cluster drift supporting H3.

Design decisions
----------------
- Epoch 0  = base model before fine-tuning (loaded from CKPT_PATHS[base]).
- Epochs 1…N = pca_frame_epoch_*.pth files produced by --save_n_frames N.
- Test subset: N_SAMPLES_VIZ (default 1000) drawn with fixed seed.
- The PCA used for adversarial figures is identical to the clean one (fitted
  on clean z of the final epoch). Both projections are therefore directly
  comparable: cluster drift is readable as displacement in the same space.
- B30 and ProtoVAE are supported (SENN has no prototype latent z here).

Usage
-----
    # Clean figure (original behaviour):
    python ch6_pca_training.py --variant B30-FT-0

    # Compare variants with one global clean PCA, saving one PDF per variant:
    python ch6_pca_training.py --arch B30 --variants B30-FT-0 B30-FT-E

    # Clean ProtoVAE figure:
    python ch6_pca_training.py --variant ProtoVAE-FT-0

    # Both clean + adversarial at eps 0.1, 0.3, 0.5:
    python ch6_pca_training.py --variant B30-FT-0 --show_adv --eps_list 0.1 0.3 0.5

    # Adversarial only (skip clean figure):
    python ch6_pca_training.py --variant B30-FT-0 --show_adv --no_clean
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from config import (
    RC_PARAMS, FIGURES_ROOT, CKPT_PATHS,
    EPOCH_CKPT_DIR_TEMPLATE, PCA_FRAME_PREFIX,
    DEFAULT_SEED, VARIANT_LABELS,
)
from loaders import load_pca_frame_paths
import sys
path = str(Path(__file__).parent.parent)
sys.path.append(path)
from metric_extractor import extract_internals, get_proto_labels


# =============================================================================
# Constants
# =============================================================================

SUPPORTED_ARCHS   = ["B30", "ProtoVAE"]
N_SAMPLES_VIZ     = 1000
CMAP              = "tab10"

PGD_ITERS_DEFAULT = 80
PGD_ALPHA_DEFAULT = 0.01
EPS_LIST_DEFAULT  = [0.3]


# =============================================================================
# Model loading
# =============================================================================

def _load_b30(ckpt_path: str | Path, device: torch.device) -> nn.Module:
    """
    Load B30 from a checkpoint.

    Base checkpoint: full model object (torch.load gives nn.Module directly).
    PCA frame:       dict with "model_state" key — applied on top of base arch.
    """
    base_path = CKPT_PATHS["B30"]

    try:
        model = torch.load(base_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(base_path, map_location=device)

    ckpt_path = Path(ckpt_path)
    if ckpt_path != Path(base_path):
        state = torch.load(ckpt_path, map_location=device)
        model_state = state.get("model_state", state)
        model.load_state_dict(model_state)

    return model.to(device).eval()


class ProtoVAEWrapper(nn.Module):
    """Evaluation wrapper matching run_metrics.py: accepts x in [0,1]."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        self.input_bounds = (0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2 - 1
        logits, _ = self.base.pred_class(x)
        return logits


def _load_protovae(ckpt_path: str | Path, device: torch.device) -> nn.Module:
    """
    Load ProtoVAE from the base checkpoint or a fine-tuning PCA frame.

    Base checkpoint: plain state_dict.
    PCA frame:       dict with "model_state" key.
    """
    from ProtoVAE import model as model_protovae

    model = model_protovae.ProtoVAE().to(device)
    state = torch.load(ckpt_path, map_location=device)
    model_state = state.get("model_state", state) if isinstance(state, dict) else state
    model.load_state_dict(model_state)
    return ProtoVAEWrapper(model).to(device).eval()


def load_model_for_frame(arch: str, ckpt_path: str | Path, device: torch.device) -> nn.Module:
    if arch == "B30":
        return _load_b30(ckpt_path, device)
    if arch == "ProtoVAE":
        return _load_protovae(ckpt_path, device)
    raise ValueError(f"[pca_training] Unsupported arch '{arch}'.")


def infer_arch_from_variant(variant: str) -> str:
    """Infer architecture from the variant name when --arch is omitted."""
    if variant == "B30" or variant.startswith("B30-"):
        return "B30"
    if variant == "ProtoVAE" or variant.startswith("ProtoVAE-"):
        return "ProtoVAE"
    raise ValueError(
        f"Cannot infer architecture from variant '{variant}'. "
        f"Please pass --arch explicitly. Choices: {SUPPORTED_ARCHS}"
    )


def validate_variants_for_arch(arch: str, variants: list[str]) -> None:
    """Fail fast if a multi-variant PCA would mix architectures."""
    for variant in variants:
        inferred = infer_arch_from_variant(variant)
        if inferred != arch:
            raise ValueError(
                f"Variant '{variant}' belongs to arch '{inferred}', "
                f"but --arch is '{arch}'. Do not mix architectures in one PCA."
            )


# =============================================================================
# Data helpers
# =============================================================================

def get_test_subset(n_samples: int, seed: int, batch_size: int = 250):
    """
    Return (x_subset, y_subset) tensors drawn deterministically from MNIST test.
    """
    from data_loader import get_test_loader as _gtl
    loader = _gtl("data", batch_size=batch_size, shuffle=False,
                  num_workers=0, pin_memory=False)

    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)

    x_all = torch.cat(xs, dim=0)   # (10000, 1, 28, 28)
    y_all = torch.cat(ys, dim=0)   # (10000,)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_all), size=min(n_samples, len(x_all)), replace=False)
    idx = np.sort(idx)
    return x_all[idx], y_all[idx]


# =============================================================================
# Latent code extractors
# =============================================================================

@torch.no_grad()
def extract_z(model: nn.Module, x: torch.Tensor, device: torch.device,
              batch_size: int = 250) -> np.ndarray:
    """Extract z for clean inputs. Returns (N, D) float32 numpy array."""
    model.eval()
    zs = []
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size].to(device)
        internals = extract_internals(model, xb)
        zs.append(internals["z"].cpu().numpy())
    return np.concatenate(zs, axis=0)


def extract_z_adv(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    device: torch.device,
    pgd_iters: int = PGD_ITERS_DEFAULT,
    pgd_alpha: float = PGD_ALPHA_DEFAULT,
    batch_size: int = 128,
) -> np.ndarray:
    """
    Run PGD-Linf on x, then extract z from the adversarial examples.
    Returns (N, D) float32 numpy array.

    batch_size is smaller than for clean extraction because PGD keeps the
    full gradient graph in memory during the attack iterations.
    """
    from adversarial_attacks import PGDLInf_attack

    model.eval()
    zs = []

    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size].to(device)
        yb = y[start:start + batch_size].to(device)

        def loss_f(*, batch_x: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(model(batch_x), yb)

        x_adv = PGDLInf_attack(
            xb, loss_f,
            iters=pgd_iters,
            eps=eps,
            alpha=pgd_alpha,
            random_start=True,
        )

        with torch.no_grad():
            internals = extract_internals(model, x_adv)
            zs.append(internals["z"].cpu().numpy())

    return np.concatenate(zs, axis=0)


@torch.no_grad()
def extract_prototypes(arch: str, model: nn.Module) -> np.ndarray:
    """Return prototype vectors as (n_proto, D) numpy array."""
    if arch == "B30":
        return model.prototype_layer.prototype_distances.detach().cpu().numpy()
    if arch == "ProtoVAE":
        base = model.base if hasattr(model, "base") else model
        return base.prototype_vectors.detach().cpu().numpy()
    raise ValueError(f"[pca_training] No prototype extractor for arch '{arch}'.")


# =============================================================================
# PCA helpers
# =============================================================================

def fit_pca(z: np.ndarray) -> PCA:
    """Fit and return a 2-component PCA on z (N, D)."""
    pca = PCA(n_components=2, random_state=0)
    pca.fit(z)
    return pca


def save_pca(pca: PCA, fpath: str) -> None:
    """Persist a fitted PCA object for reproducibility/reuse."""
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    with open(fpath, "wb") as f:
        pickle.dump(pca, f)
    print(f"  Saved PCA -> {fpath}")


def load_pca(pca_path: str | Path) -> PCA:
    """Load a fitted PCA object saved by ch6_pca_training.py."""
    with open(pca_path, "rb") as f:
        return pickle.load(f)


def project(pca: PCA, z: np.ndarray) -> np.ndarray:
    """Project (N, D) → (N, 2)."""
    return pca.transform(z)


# =============================================================================
# Plotting — shared helpers
# =============================================================================

def _draw_pca_ax(
    ax: plt.Axes,
    z_2d: np.ndarray,
    proto_2d: np.ndarray,
    proto_lbl: np.ndarray,
    y: np.ndarray,
    title: str,
    var_exp: np.ndarray,
    cmap,
    n_classes: int = 10,
) -> None:
    """Draw one PCA scatter panel. Shared by clean and adversarial plots."""
    for cls in range(n_classes):
        mask = y == cls
        if not mask.any():
            continue
        ax.scatter(
            z_2d[mask, 0], z_2d[mask, 1],
            s=6, alpha=0.45,
            color=cmap(cls / n_classes),
            linewidths=0,
        )

    for p_idx in range(len(proto_2d)):
        cls = int(proto_lbl[p_idx])
        ax.scatter(
            proto_2d[p_idx, 0], proto_2d[p_idx, 1],
            s=120, marker="*",
            color=cmap(cls / n_classes),
            edgecolors="black", linewidths=0.7, zorder=5,
        )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"PC1 ({var_exp[0]:.1%})", fontsize=8)
    ax.set_ylabel(f"PC2 ({var_exp[1]:.1%})", fontsize=8)
    ax.tick_params(labelsize=7)


def _make_figure(n_frames: int) -> tuple[plt.Figure, np.ndarray]:
    """Create figure + flattened axes array. Hidden axes for spare slots."""
    ncols = min(n_frames, 3)
    nrows = (n_frames + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.2 * nrows))
    axes_flat = np.array(axes).flatten()
    for idx in range(n_frames, nrows * ncols):
        axes_flat[idx].set_visible(False)
    return fig, axes_flat


def _add_legend(fig: plt.Figure, n_classes: int, cmap) -> None:
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=cmap(c / n_classes), markersize=7,
                   label=str(c))
        for c in range(n_classes)
    ]
    fig.legend(
        handles=handles, title="Class",
        loc="lower center", ncol=n_classes,
        fontsize=7, bbox_to_anchor=(0.5, -0.04),
    )


def _save_fig(fig: plt.Figure, fpath: str) -> None:
    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {fpath}")


# =============================================================================
# Plotting — CLEAN figure (original behaviour, unchanged interface)
# =============================================================================

def plot_pca_frames(
    frames: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    pca: PCA,
    y: np.ndarray,
    arch: str,
    variant: str,
    out_dir: str,
    *,
    global_pca: bool = False,
) -> None:
    """
    Plot all clean frames in a grid figure.

    Parameters
    ----------
    frames  : list of (epoch, z_2d, proto_2d, proto_labels)
    pca     : fitted PCA (for explained variance annotation)
    y       : (N,) ground-truth class labels
    variant : model variant name
    out_dir : output directory
    """
    n_frames = len(frames)
    cmap     = plt.get_cmap(CMAP)
    var_exp  = pca.explained_variance_ratio_

    fig, axes = _make_figure(n_frames)

    for ax, (epoch, z_2d, proto_2d, proto_lbl) in zip(axes, frames):
        title = "Pre-FT (Base)" if epoch == 0 else f"Epoch {epoch}"
        _draw_pca_ax(ax, z_2d, proto_2d, proto_lbl, y, title, var_exp, cmap)

    _add_legend(fig, 10, cmap)
    label = VARIANT_LABELS.get(variant, variant)
    pca_desc = "PCA fit on global clean latent codes" if global_pca else "PCA fit on final epoch"
    fig.suptitle(
        f"Latent space evolution during fine-tuning — {label} ({arch})\n"
        f"★ = prototypes  |  dots = test samples  |  {pca_desc}",
        fontsize=11, y=1.03,
    )
    plt.tight_layout()
    prefix = "ch6_pca_training_global" if global_pca else "ch6_pca_training"
    _save_fig(fig, os.path.join(out_dir, f"{prefix}_{variant}.pdf"))


# =============================================================================
# Plotting — ADVERSARIAL figure (new, one per epsilon)
# =============================================================================

def plot_pca_frames_adv(
    frames: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    pca: PCA,
    y: np.ndarray,
    arch: str,
    variant: str,
    eps: float,
    out_dir: str,
    *,
    global_pca: bool = False,
) -> None:
    """
    Same layout as plot_pca_frames but z_2d contains adversarial latent codes.
    Prototypes stay at their fixed positions.

    Parameters
    ----------
    frames  : list of (epoch, z_adv_2d, proto_2d, proto_labels)
    eps     : epsilon used for PGD (shown in title and filename)
    """
    n_frames = len(frames)
    cmap     = plt.get_cmap(CMAP)
    var_exp  = pca.explained_variance_ratio_

    fig, axes = _make_figure(n_frames)

    for ax, (epoch, z_2d, proto_2d, proto_lbl) in zip(axes, frames):
        title = "Pre-FT (Base)" if epoch == 0 else f"Epoch {epoch}"
        _draw_pca_ax(ax, z_2d, proto_2d, proto_lbl, y, title, var_exp, cmap)

    _add_legend(fig, 10, cmap)
    label = VARIANT_LABELS.get(variant, variant)
    pca_desc = (
        "PCA fit on global clean latent codes"
        if global_pca
        else "PCA fit on clean final epoch"
    )
    fig.suptitle(
        rf"Latent space under PGD attack ($\varepsilon = {eps}$) — {label} ({arch})"
        f"\n★ = prototypes  |  dots = adversarial samples  |  {pca_desc}",
        fontsize=11, y=1.03,
    )
    plt.tight_layout()

    eps_tag = f"{eps:.3f}".replace(".", "p")
    prefix  = "ch6_pca_training_global" if global_pca else "ch6_pca_training"
    fname   = f"{prefix}_{variant}_adv_eps{eps_tag}.pdf"
    _save_fig(fig, os.path.join(out_dir, fname))


# =============================================================================
# Variant processing helpers
# =============================================================================

def load_variant_frames(
    arch: str,
    variant: str,
    seed: int,
) -> list[tuple[int, Path]] | None:
    """Discover saved PCA frames for one variant, including epoch 0 base."""
    print(f"Discovering PCA frame checkpoints for {variant} ...")
    try:
        all_frames = load_pca_frame_paths(arch, variant, seed)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print(
            "  Make sure the fine-tuning was run with --save_n_frames N.\n"
            "  See config.py -> EPOCH_CKPT_DIR_TEMPLATE."
        )
        return None

    if len(all_frames) < 2:
        print(
            f"[ERROR] Only {len(all_frames)} frame(s) found for {variant} "
            "(including epoch 0). Need at least 2.\n"
            "  Run fine-tuning with --save_n_frames N (N >= 1)."
        )
        return None

    print(f"  Frames found: {[ep for ep, _ in all_frames]}")
    return all_frames


def extract_variant_frame_data(
    arch: str,
    variant: str,
    selected: list[tuple[int, Path]],
    x_sub: torch.Tensor,
    device: torch.device,
) -> tuple[
    list[nn.Module],
    list[tuple[int, np.ndarray]],
    list[tuple[np.ndarray, np.ndarray]],
]:
    """Load models and extract clean latent codes/prototypes for one variant."""
    models_per_frame:  list[nn.Module]                     = []
    z_clean_per_frame: list[tuple[int, np.ndarray]]        = []
    proto_per_frame:   list[tuple[np.ndarray, np.ndarray]] = []

    for epoch, ckpt_path in selected:
        print(f"  Loading {variant} epoch {epoch} from {ckpt_path} ...")
        model      = load_model_for_frame(arch, ckpt_path, device)
        z          = extract_z(model, x_sub, device)
        prototypes = extract_prototypes(arch, model)
        proto_lbl  = get_proto_labels(model).numpy()

        models_per_frame.append(model)
        z_clean_per_frame.append((epoch, z))
        proto_per_frame.append((prototypes, proto_lbl))

    return models_per_frame, z_clean_per_frame, proto_per_frame


def make_clean_frames(
    z_clean_per_frame: list[tuple[int, np.ndarray]],
    proto_per_frame: list[tuple[np.ndarray, np.ndarray]],
    pca: PCA,
) -> list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]:
    """Project clean latent codes and prototypes through the chosen PCA."""
    return [
        (epoch, project(pca, z), project(pca, protos), proto_lbl)
        for (epoch, z), (protos, proto_lbl) in zip(z_clean_per_frame, proto_per_frame)
    ]


def generate_adversarial_figures(
    models_per_frame: list[nn.Module],
    z_clean_per_frame: list[tuple[int, np.ndarray]],
    proto_per_frame: list[tuple[np.ndarray, np.ndarray]],
    pca: PCA,
    y_np: np.ndarray,
    x_sub: torch.Tensor,
    y_sub: torch.Tensor,
    arch: str,
    variant: str,
    out_dir: str,
    eps_list: list[float],
    pgd_iters: int,
    pgd_alpha: float,
    device: torch.device,
    *,
    global_pca: bool = False,
) -> None:
    """Generate one adversarial PCA PDF per epsilon for one variant."""
    for eps in eps_list:
        print(f"\nGenerating adversarial PCA figure for {variant} -- eps={eps} ...")
        frames_adv = []
        for model, (epoch, _), (protos, proto_lbl) in zip(
            models_per_frame, z_clean_per_frame, proto_per_frame
        ):
            print(f"  Attacking {variant} epoch {epoch} ...")
            z_adv = extract_z_adv(
                model, x_sub, y_sub, eps, device,
                pgd_iters=pgd_iters,
                pgd_alpha=pgd_alpha,
            )
            frames_adv.append(
                (epoch, project(pca, z_adv), project(pca, protos), proto_lbl)
            )
        plot_pca_frames_adv(
            frames_adv, pca, y_np, arch, variant, eps, out_dir,
            global_pca=global_pca,
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 6.2 — PCA of latent space evolution during fine-tuning."
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        help="FT variant name, e.g. B30-FT-0 or ProtoVAE-FT-0.",
    )
    parser.add_argument(
        "--variants", type=str, nargs="+", default=None,
        help="FT variant names to compare with one global PCA.",
    )
    parser.add_argument(
        "--arch", type=str, default=None,
        choices=SUPPORTED_ARCHS,
        help="Architecture to visualise. If omitted, inferred from --variant.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=N_SAMPLES_VIZ,
        help="Test samples to visualise (default 1000).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    # --- Adversarial mode ---
    parser.add_argument(
        "--show_adv", action="store_true",
        help="Also generate adversarial PCA figures (one PDF per epsilon).",
    )
    parser.add_argument(
        "--eps_list", type=float, nargs="+", default=EPS_LIST_DEFAULT,
        help="Epsilon values for adversarial mode (default: 0.1 0.3 0.5).",
    )
    parser.add_argument(
        "--pgd_iters", type=int, default=PGD_ITERS_DEFAULT,
        help="PGD iterations (default: 40).",
    )
    parser.add_argument(
        "--pgd_alpha", type=float, default=PGD_ALPHA_DEFAULT,
        help="PGD step size (default: 0.01).",
    )
    parser.add_argument(
        "--no_clean", action="store_true",
        help="Skip the clean figure (useful when only regenerating adversarial).",
    )
    parser.add_argument(
        "--pca_path", type=str, default=None,
        help="Optional PCA .pkl to load instead of fitting a new PCA.",
    )

    args = parser.parse_args()
    if args.variants is not None:
        if args.variant is not None:
            parser.error("Use either --variant or --variants, not both.")
        if args.arch is None:
            parser.error("--arch is required when --variants is provided.")
        try:
            validate_variants_for_arch(args.arch, args.variants)
        except ValueError as e:
            parser.error(str(e))
    elif args.variant is None:
        parser.error("Either --variant or --variants is required.")
    elif args.arch is None:
        args.arch = infer_arch_from_variant(args.variant)

    plt.rcParams.update(RC_PARAMS)
    out_dir = os.path.join(FIGURES_ROOT, "ch6")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load fixed test subset
    # ------------------------------------------------------------------
    print(f"\nLoading {args.n_samples} test samples ...")
    x_sub, y_sub = get_test_subset(args.n_samples, args.seed)
    y_np = y_sub.numpy()

    if args.variants is not None:
        variants = args.variants
        variant_data = {}
        all_clean_z = []

        for variant in variants:
            selected = load_variant_frames(args.arch, variant, args.seed)
            if selected is None:
                return

            models_per_frame, z_clean_per_frame, proto_per_frame = extract_variant_frame_data(
                args.arch, variant, selected, x_sub, device
            )
            variant_data[variant] = {
                "models_per_frame": models_per_frame,
                "z_clean_per_frame": z_clean_per_frame,
                "proto_per_frame": proto_per_frame,
            }
            all_clean_z.extend(z for _, z in z_clean_per_frame)

        if args.pca_path is not None:
            print(f"\nLoading PCA from {args.pca_path} ...")
            pca = load_pca(args.pca_path)
        else:
            print("\nFitting global PCA on clean latent codes from all variants/epochs ...")
            z_global = np.concatenate(all_clean_z, axis=0)
            pca = fit_pca(z_global)
        print(
            f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, "
            f"PC2={pca.explained_variance_ratio_[1]:.1%}"
        )
        if args.pca_path is None:
            pca_out_path = os.path.join(out_dir, f"ch6_pca_training_global_{args.arch}_pca.pkl")
            save_pca(pca, pca_out_path)

        for variant in variants:
            data = variant_data[variant]
            frames_clean = make_clean_frames(
                data["z_clean_per_frame"], data["proto_per_frame"], pca
            )

            if not args.no_clean:
                print(f"\nGenerating clean PCA figure for {variant} ...")
                plot_pca_frames(
                    frames_clean, pca, y_np, args.arch, variant, out_dir,
                    global_pca=True,
                )

            if args.show_adv:
                generate_adversarial_figures(
                    data["models_per_frame"],
                    data["z_clean_per_frame"],
                    data["proto_per_frame"],
                    pca,
                    y_np,
                    x_sub,
                    y_sub,
                    args.arch,
                    variant,
                    out_dir,
                    args.eps_list,
                    args.pgd_iters,
                    args.pgd_alpha,
                    device,
                    global_pca=True,
                )

        print("\nDone.")
        return

    # ------------------------------------------------------------------
    # 2. Discover epoch frame checkpoints
    # ------------------------------------------------------------------
    print(f"Discovering PCA frame checkpoints for {args.variant} ...")
    try:
        all_frames = load_pca_frame_paths(args.arch, args.variant, args.seed)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print(
            "  Make sure the fine-tuning was run with --save_n_frames N.\n"
            "  See config.py -> EPOCH_CKPT_DIR_TEMPLATE."
        )
        return

    if len(all_frames) < 2:
        print(
            f"[ERROR] Only {len(all_frames)} frame(s) found "
            "(including epoch 0). Need at least 2.\n"
            "  Run fine-tuning with --save_n_frames N (N >= 1)."
        )
        return

    selected = all_frames
    print(f"  Frames found: {[ep for ep, _ in selected]}")

    # ------------------------------------------------------------------
    # 3. Load models + extract clean z and prototypes for every frame.
    #    Models are kept alive for the adversarial pass (step 6).
    # ------------------------------------------------------------------
    models_per_frame:  list[nn.Module]                     = []
    z_clean_per_frame: list[tuple[int, np.ndarray]]        = []
    proto_per_frame:   list[tuple[np.ndarray, np.ndarray]] = []

    for epoch, ckpt_path in selected:
        print(f"  Loading epoch {epoch} from {ckpt_path} ...")
        model      = load_model_for_frame(args.arch, ckpt_path, device)
        z          = extract_z(model, x_sub, device)
        prototypes = extract_prototypes(args.arch, model)
        proto_lbl  = get_proto_labels(model).numpy()

        models_per_frame.append(model)
        z_clean_per_frame.append((epoch, z))
        proto_per_frame.append((prototypes, proto_lbl))

    # ------------------------------------------------------------------
    # 4. Fit PCA on the FINAL frame's clean latent codes, or load one.
    # ------------------------------------------------------------------
    if args.pca_path is not None:
        print(f"\nLoading PCA from {args.pca_path} ...")
        pca = load_pca(args.pca_path)
    else:
        print("\nFitting PCA on final-epoch latent codes ...")
        _, z_final = z_clean_per_frame[-1]
        pca = fit_pca(z_final)
    print(
        f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, "
        f"PC2={pca.explained_variance_ratio_[1]:.1%}"
    )

    # ------------------------------------------------------------------
    # 5. Clean figure
    # ------------------------------------------------------------------
    frames_clean = [
        (epoch, project(pca, z), project(pca, protos), proto_lbl)
        for (epoch, z), (protos, proto_lbl) in zip(z_clean_per_frame, proto_per_frame)
    ]

    if not args.no_clean:
        print("\nGenerating clean PCA figure ...")
        plot_pca_frames(frames_clean, pca, y_np, args.arch, args.variant, out_dir)

    # ------------------------------------------------------------------
    # 6. Adversarial figures — one per epsilon
    # ------------------------------------------------------------------
    if args.show_adv:
        for eps in args.eps_list:
            print(f"\nGenerating adversarial PCA figure -- eps={eps} ...")
            frames_adv = []
            for model, (epoch, _), (protos, proto_lbl) in zip(
                models_per_frame, z_clean_per_frame, proto_per_frame
            ):
                print(f"  Attacking epoch {epoch} ...")
                z_adv = extract_z_adv(
                    model, x_sub, y_sub, eps, device,
                    pgd_iters=args.pgd_iters,
                    pgd_alpha=args.pgd_alpha,
                )
                frames_adv.append(
                    (epoch, project(pca, z_adv), project(pca, protos), proto_lbl)
                )
            plot_pca_frames_adv(frames_adv, pca, y_np, args.arch, args.variant, eps, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
