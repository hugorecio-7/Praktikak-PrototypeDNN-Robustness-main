"""
ch6_pca_training.py
===================
Chapter 6.2 — Latent space evolution during fine-tuning (training time).

What this script produces
--------------------------
Figure  (ch6_pca_training_{variant}.pdf)
    N_FRAMES scatter plots arranged in a single row, one per selected epoch.
    Each plot shows the PCA projection of z(x) for a fixed test subset,
    coloured by ground-truth class, with prototype positions overlaid as
    larger markers with black edges.

    The PCA fit is done on the FINAL epoch's latent codes and then reused
    for all frames (same projection, comparable positions across epochs).

Design decisions
----------------
- Epoch 0  = base model before fine-tuning (loaded from CKPT_PATHS[base]).
- Epochs 1…N = pca_frame_epoch_*.pth files produced by --save_n_frames 5.
- Subset of test samples: N_SAMPLES_VIZ (default 1000) drawn with fixed seed
  for reproducibility without loading the full 10k test set N_FRAMES times.
- Only B30 is supported (SENN has no z, ProtoVAE PCA is future work and
  checkpoints are too large for routine use — see config.py notes).

Usage
-----
    python ch6_pca_training.py --variant B30-FT-0
    python ch6_pca_training.py --variant B30-FT-0 --n_samples 2000 --seed 42
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
from metric_extractor import extract_internals, get_proto_labels


# =============================================================================
# Constants
# =============================================================================

SUPPORTED_ARCHS  = ["B30"]          # ProtoVAE/SENN: see module docstring
N_SAMPLES_VIZ    = 1000             # test samples used for PCA scatter
N_FRAMES_DEFAULT = 5                # number of epoch frames to show
CMAP             = "tab10"          # one colour per MNIST class (10 classes)


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

    # Always start from the base architecture object.
    try:
        model = torch.load(base_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(base_path, map_location=device)

    ckpt_path = Path(ckpt_path)
    if ckpt_path != Path(base_path):
        state = torch.load(ckpt_path, map_location=device)
        # pca_frame checkpoints: {"epoch": int, "model_state": dict}
        model_state = state.get("model_state", state)
        model.load_state_dict(model_state)

    return model.to(device).eval()


# =============================================================================
# Data helpers
# =============================================================================

def get_test_subset(n_samples: int, seed: int, batch_size: int = 250):
    """
    Return (x_subset, y_subset) tensors of shape (n_samples, C, H, W) and
    (n_samples,) drawn deterministically from the MNIST test set.
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

    rng  = np.random.default_rng(seed)
    idx  = rng.choice(len(x_all), size=min(n_samples, len(x_all)), replace=False)
    idx  = np.sort(idx)
    return x_all[idx], y_all[idx]


# =============================================================================
# Latent code extractor
# =============================================================================

@torch.no_grad()
def extract_z(model: nn.Module, x: torch.Tensor, device: torch.device,
              batch_size: int = 250) -> np.ndarray:
    """
    Extract z for all samples in x, returning (N, D) float32 numpy array.
    Processes in mini-batches to stay within GPU memory.
    """
    model.eval()
    zs = []
    for start in range(0, len(x), batch_size):
        xb = x[start:start + batch_size].to(device)
        internals = extract_internals(model, xb)
        zs.append(internals["z"].cpu().numpy())
    return np.concatenate(zs, axis=0)   # (N, D)


@torch.no_grad()
def extract_prototypes(arch: str, model: nn.Module) -> np.ndarray:
    """
    Return prototype vectors as (n_proto, D) numpy array.
    Architecture-aware.
    """
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
# Plotting
# =============================================================================

def plot_pca_frames(
    frames: list[tuple[int, np.ndarray, np.ndarray, np.ndarray]],
    pca: PCA,
    y: np.ndarray,
    variant: str,
    out_dir: str,
) -> None:
    """
    Plot all frames in a single-row figure.

    Parameters
    ----------
    frames      : list of (epoch, z_2d, proto_2d, proto_labels)
    pca         : fitted PCA (for explained variance annotation)
    y           : (N,) ground-truth class labels for the test subset
    variant     : model variant name (used in title and filename)
    out_dir     : output directory
    """
    n_frames = len(frames)
    cmap     = plt.get_cmap(CMAP)
    n_classes = 10

    fig, axes = plt.subplots(1, n_frames, figsize=(4.5 * n_frames, 4.5))
    if n_frames == 1:
        axes = [axes]

    var_exp = pca.explained_variance_ratio_

    for ax, (epoch, z_2d, proto_2d, proto_lbl) in zip(axes, frames):
        # Scatter test samples coloured by class
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

        # Overlay prototypes
        for p_idx in range(len(proto_2d)):
            cls = int(proto_lbl[p_idx])
            ax.scatter(
                proto_2d[p_idx, 0], proto_2d[p_idx, 1],
                s=120, marker="*",
                color=cmap(cls / n_classes),
                edgecolors="black", linewidths=0.7, zorder=5,
            )

        label = VARIANT_LABELS.get(variant, variant)
        title = "Pre-FT (Base)" if epoch == 0 else f"Epoch {epoch}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1%})", fontsize=8)
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1%})", fontsize=8)
        ax.tick_params(labelsize=7)

    # Shared legend for classes
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=cmap(c / n_classes), markersize=7,
                   label=str(c))
        for c in range(n_classes)
    ]
    fig.legend(
        handles=legend_handles,
        title="Class", loc="lower center",
        ncol=n_classes, fontsize=7,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        f"Latent space evolution during fine-tuning — {label} (B30)\n"
        f"★ = prototypes  |  dots = test samples  |  PCA fit on final epoch",
        fontsize=11, y=1.03,
    )
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fname = f"ch6_pca_training_{variant}.pdf"
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


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
            "  Make sure the fine-tuning was run with --save_n_frames 5.\n"
            "  See config.py → EPOCH_CKPT_DIR_TEMPLATE."
        )
        return

    if len(all_frames) < 2:
        print(
            f"[ERROR] Only {len(all_frames)} frame(s) found "
            f"(including epoch 0). Need at least 2.\n"
            f"  Run fine-tuning with --save_n_frames {args.n_frames}."
        )
        return

    selected = all_frames
    print(f"  Frames found: {[ep for ep, _ in selected]}")

    # ------------------------------------------------------------------
    # 3. Extract z for each frame
    # ------------------------------------------------------------------
    z_per_frame: list[tuple[int, np.ndarray]] = []
    proto_per_frame: list[tuple[np.ndarray, np.ndarray]] = []

    for epoch, ckpt_path in selected:
        print(f"  Loading epoch {epoch} from {ckpt_path} …")
        model      = _load_b30(ckpt_path, device)
        z          = extract_z(model, x_sub, device)           # (N, D)
        prototypes = extract_prototypes(args.arch, model)       # (n_proto, D)
        proto_lbl  = get_proto_labels(model).numpy()            # (n_proto,)

        z_per_frame.append((epoch, z))
        proto_per_frame.append((prototypes, proto_lbl))

    # ------------------------------------------------------------------
    # 4. Fit PCA on the FINAL frame's latent codes
    # ------------------------------------------------------------------
    print("\nFitting PCA on final-epoch latent codes …")
    _, z_final = z_per_frame[-1]
    pca        = fit_pca(z_final)
    print(
        f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, "
        f"PC2={pca.explained_variance_ratio_[1]:.1%}"
    )

    # ------------------------------------------------------------------
    # 5. Project all frames and prototypes with the same PCA
    # ------------------------------------------------------------------
    frames_for_plot = []
    for (epoch, z), (protos, proto_lbl) in zip(z_per_frame, proto_per_frame):
        z_2d     = project(pca, z)       # (N, 2)
        proto_2d = project(pca, protos)  # (n_proto, 2)
        frames_for_plot.append((epoch, z_2d, proto_2d, proto_lbl))

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------
    print("Generating PCA figure …")
    plot_pca_frames(frames_for_plot, pca, y_np, args.variant, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
