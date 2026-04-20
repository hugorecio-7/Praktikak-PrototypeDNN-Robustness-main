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
- Only B30 is supported (SENN has no z).

Usage
-----
    # Clean figure (original behaviour):
    python ch6_pca_training.py --variant B30-FT-0

    # Both clean + adversarial at eps 0.1, 0.3, 0.5:
    python ch6_pca_training.py --variant B30-FT-0 --show_adv --eps_list 0.1 0.3 0.5

    # Adversarial only (skip clean figure):
    python ch6_pca_training.py --variant B30-FT-0 --show_adv --no_clean
"""

from __future__ import annotations

import argparse
import os
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

SUPPORTED_ARCHS   = ["B30"]
N_SAMPLES_VIZ     = 1000
CMAP              = "tab10"

PGD_ITERS_DEFAULT = 40
PGD_ALPHA_DEFAULT = 0.01
EPS_LIST_DEFAULT  = [0.1, 0.3, 0.5]


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

        def loss_f(x_in: torch.Tensor) -> torch.Tensor:
            return F.cross_entropy(model(x_in), yb)

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
    raise ValueError(f"[pca_training] No prototype extractor for arch '{arch}'.")


# =============================================================================
# PCA helpers
# =============================================================================

def fit_pca(z: np.ndarray) -> PCA:
    """Fit and return a 2-component PCA on z (N, D)."""
    pca = PCA(n_components=2, random_state=0)
    pca.fit(z)
    return pca


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
    print(f"  Saved → {fpath}")


# =============================================================================
# Plotting — CLEAN figure (original behaviour, unchanged interface)
# =============================================================================

def plot_pca_frames(
    frames: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    pca: PCA,
    y: np.ndarray,
    variant: str,
    out_dir: str,
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
    fig.suptitle(
        f"Latent space evolution during fine-tuning — {label} (B30)\n"
        f"★ = prototypes  |  dots = test samples  |  PCA fit on final epoch",
        fontsize=11, y=1.03,
    )
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, f"ch6_pca_training_{variant}.pdf"))


# =============================================================================
# Plotting — ADVERSARIAL figure (new, one per epsilon)
# =============================================================================

def plot_pca_frames_adv(
    frames: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    pca: PCA,
    y: np.ndarray,
    variant: str,
    eps: float,
    out_dir: str,
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
    fig.suptitle(
        rf"Latent space under PGD attack ($\varepsilon = {eps}$) — {label} (B30)"
        "\n★ = prototypes  |  dots = adversarial samples  |  PCA fit on clean final epoch",
        fontsize=11, y=1.03,
    )
    plt.tight_layout()

    eps_tag = f"{eps:.3f}".replace(".", "p")
    fname   = f"ch6_pca_training_{variant}_adv_eps{eps_tag}.pdf"
    _save_fig(fig, os.path.join(out_dir, fname))


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 6.2 — PCA of latent space evolution during fine-tuning."
    )
    parser.add_argument(
        "--variant", type=str, required=True,
        help="FT variant name, e.g. B30-FT-0.",
    )
    parser.add_argument(
        "--arch", type=str, default="B30",
        choices=SUPPORTED_ARCHS,
        help="Architecture (only B30 supported).",
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

    args = parser.parse_args()

    plt.rcParams.update(RC_PARAMS)
    out_dir = os.path.join(FIGURES_ROOT, "ch6")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load fixed test subset
    # ------------------------------------------------------------------
    print(f"\nLoading {args.n_samples} test samples …")
    x_sub, y_sub = get_test_subset(args.n_samples, args.seed)
    y_np = y_sub.numpy()

    # ------------------------------------------------------------------
    # 2. Discover epoch frame checkpoints
    # ------------------------------------------------------------------
    print(f"Discovering PCA frame checkpoints for {args.variant} …")
    try:
        all_frames = load_pca_frame_paths(args.arch, args.variant, args.seed)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print(
            "  Make sure the fine-tuning was run with --save_n_frames N.\n"
            "  See config.py → EPOCH_CKPT_DIR_TEMPLATE."
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
        print(f"  Loading epoch {epoch} from {ckpt_path} …")
        model      = _load_b30(ckpt_path, device)
        z          = extract_z(model, x_sub, device)
        prototypes = extract_prototypes(args.arch, model)
        proto_lbl  = get_proto_labels(model).numpy()

        models_per_frame.append(model)
        z_clean_per_frame.append((epoch, z))
        proto_per_frame.append((prototypes, proto_lbl))

    # ------------------------------------------------------------------
    # 4. Fit PCA on the FINAL frame's clean latent codes
    # ------------------------------------------------------------------
    print("\nFitting PCA on final-epoch latent codes …")
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
        print("\nGenerating clean PCA figure …")
        plot_pca_frames(frames_clean, pca, y_np, args.variant, out_dir)

    # ------------------------------------------------------------------
    # 6. Adversarial figures — one per epsilon
    # ------------------------------------------------------------------
    if args.show_adv:
        for eps in args.eps_list:
            print(f"\nGenerating adversarial PCA figure — ε={eps} …")
            frames_adv = []
            for model, (epoch, _), (protos, proto_lbl) in zip(
                models_per_frame, z_clean_per_frame, proto_per_frame
            ):
                print(f"  Attacking epoch {epoch} …")
                z_adv = extract_z_adv(
                    model, x_sub, y_sub, eps, device,
                    pgd_iters=args.pgd_iters,
                    pgd_alpha=args.pgd_alpha,
                )
                frames_adv.append(
                    (epoch, project(pca, z_adv), project(pca, protos), proto_lbl)
                )
            plot_pca_frames_adv(frames_adv, pca, y_np, args.variant, eps, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
