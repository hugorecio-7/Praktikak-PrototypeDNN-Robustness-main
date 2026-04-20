"""
ch6_pca_trajectories.py
=======================
Chapter 6.3 — Adversarial trajectories in latent space (attack time).

What this script produces
--------------------------
Figure  (ch6_pca_trajectories_{variant_a}_vs_{variant_b}.pdf)
    Two-panel PCA plot contrasting two models (typically Base vs FT-0):
      Left panel  — base model: PGD trajectory crosses the prototype boundary.
      Right panel — FT model:   trajectory stays contained near correct protos.

    For N_TRAJ_SAMPLES selected test samples, each trajectory is drawn as a
    connected line from the clean point (circle) through all PGD iterations
    to the final adversarial point (cross). Prototype positions are overlaid.

    The PCA is fit on the clean latent codes of BOTH models jointly so the
    two panels share the same projection axes and are directly comparable.

Design decisions
----------------
- Uses a local PGD-with-trajectory function that mirrors PGDLInf_attack
  exactly (same math, same clipping) but captures z at every iteration.
  The original function is not modified.
- N_TRAJ_SAMPLES is kept small (default 10) so trajectories remain readable.
  Samples are selected to include a mix of classes.
- PCA fit: on clean z of both models combined (after centering each model's
  codes independently to avoid scale dominating the projection).
- Only B30 is supported (same reasoning as ch6_pca_training.py).

Usage
-----
    # Default: Base vs FT-0
    python ch6_pca_trajectories.py

    # Custom pair
    python ch6_pca_trajectories.py --model_a B30 --model_b B30-FT-E

    # More trajectory samples, higher PGD iterations
    python ch6_pca_trajectories.py --n_samples 20 --pgd_iters 50
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
import matplotlib.patheffects as pe
from sklearn.decomposition import PCA

from config import (
    RC_PARAMS, FIGURES_ROOT, CKPT_PATHS,
    DEFAULT_SEED, VARIANT_LABELS, VARIANT_COLORS,
)
import sys
path = str(Path(__file__).parent.parent)
sys.path.append(path)
from metric_extractor import extract_internals, get_proto_labels


# =============================================================================
# Constants
# =============================================================================

N_TRAJ_SAMPLES   = 10      # number of trajectories to draw per panel
PGD_EPS_DEFAULT  = 0.3     # epsilon for trajectory visualisation
PGD_ALPHA        = 0.01
PGD_ITERS        = 40
CMAP             = "tab10"
N_CLASSES        = 10


# =============================================================================
# Model loading  (mirrors run_metrics.py exactly)
# =============================================================================

def _load_b30(model_name: str, device: torch.device) -> nn.Module:
    base_path = CKPT_PATHS["B30"]
    try:
        model = torch.load(base_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(base_path, map_location=device)

    if model_name != "B30":
        ckpt  = torch.load(CKPT_PATHS[model_name], map_location=device)
        model.load_state_dict(ckpt["model_state"])

    return model.to(device).eval()


# =============================================================================
# Data helpers
# =============================================================================

def get_test_subset(n_samples: int, seed: int):
    """
    Return (x, y) tensors with n_samples samples, one per class where possible,
    drawn deterministically.
    """
    from data_loader import get_test_loader as _gtl
    loader = _gtl("data", batch_size=500, shuffle=False,
                  num_workers=0, pin_memory=False)

    xs, ys = [], []
    for x, y in loader:
        xs.append(x); ys.append(y)

    x_all = torch.cat(xs, dim=0)
    y_all = torch.cat(ys, dim=0)

    rng = np.random.default_rng(seed)

    # Try to pick samples evenly across classes
    indices = []
    per_class = max(1, n_samples // N_CLASSES)
    for cls in range(N_CLASSES):
        cls_idx = np.where(y_all.numpy() == cls)[0]
        chosen  = rng.choice(cls_idx, size=min(per_class, len(cls_idx)), replace=False)
        indices.extend(chosen.tolist())

    # Fill up to n_samples if needed
    remaining = n_samples - len(indices)
    if remaining > 0:
        pool = list(set(range(len(x_all))) - set(indices))
        extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
        indices.extend(extra.tolist())

    indices = sorted(indices[:n_samples])
    return x_all[indices], y_all[indices]


# =============================================================================
# Local PGD with trajectory capture
# ─────────────────────────────────
# Mirrors PGDLInf_attack from adversarial_attacks.py mathematically.
# The only additions are:
#   - extract_fn called at every iteration to capture z
#   - returns (x_adv, trajectory) instead of just x_adv
# =============================================================================

def pgd_with_trajectory(
    model:       nn.Module,
    x:           torch.Tensor,
    loss_f,
    iters:       int,
    eps:         float,
    alpha:       float,
    device:      torch.device,
    extract_fn,
) -> tuple[torch.Tensor, list[np.ndarray]]:
    """
    PGD L-inf attack that captures the latent code z at every iteration.

    Parameters
    ----------
    model      : eval-mode model
    x          : (B, C, H, W) clean inputs in [0, 1]
    loss_f     : callable(batch_x) → scalar loss
    iters      : number of PGD steps
    eps        : L-inf budget
    alpha      : step size
    device     : torch device
    extract_fn : function(model, x_batch) → dict with "z" key

    Returns
    -------
    x_adv      : (B, C, H, W) final adversarial examples
    trajectory : list of (iters+1) arrays, each (B, D)
                 trajectory[0] = clean z, trajectory[k] = z after step k
    """
    x_orig = x.clone().detach().to(device)
    x_adv  = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv  = torch.clamp(x_adv, 0.0, 1.0).detach()

    trajectory: list[np.ndarray] = []

    # Step 0: clean latent code
    with torch.no_grad():
        z0 = extract_fn(model, x_orig)["z"].cpu().numpy()
    trajectory.append(z0)

    for _ in range(iters):
        x_adv.requires_grad = True
        loss = loss_f(batch_x=x_adv)
        grad = torch.autograd.grad(loss, x_adv)[0]

        x_adv = x_adv.detach() + alpha * torch.sign(grad)
        delta  = torch.clamp(x_adv - x_orig, min=-eps, max=eps)
        x_adv  = torch.clamp(x_orig + delta, 0.0, 1.0).detach()

        with torch.no_grad():
            z_step = extract_fn(model, x_adv)["z"].cpu().numpy()
        trajectory.append(z_step)

    return x_adv, trajectory     # trajectory: list[(iters+1), (B, D)]


# =============================================================================
# Loss function builders  (one per architecture, mirrors run_metrics.py)
# =============================================================================

def _b30_loss_f(model: nn.Module, y: torch.Tensor):
    """Return a loss callable for B30 (cross-entropy on raw logits)."""
    import torch.nn.functional as F
    def loss_f(*, batch_x):
        return F.cross_entropy(model(batch_x), y)
    return loss_f


# =============================================================================
# PCA helpers
# =============================================================================

def fit_joint_pca(z_a: np.ndarray, z_b: np.ndarray) -> PCA:
    """
    Fit PCA on the union of clean latent codes from two models.

    Each model's codes are mean-centred independently before stacking so that
    scale differences between models do not dominate the projection.
    """
    z_a_c = z_a - z_a.mean(axis=0, keepdims=True)
    z_b_c = z_b - z_b.mean(axis=0, keepdims=True)
    joint  = np.concatenate([z_a_c, z_b_c], axis=0)
    pca    = PCA(n_components=2, random_state=0)
    pca.fit(joint)
    return pca


def _centre_and_project(pca: PCA, z: np.ndarray, z_ref: np.ndarray) -> np.ndarray:
    """
    Centre z by the mean of z_ref, then project with pca.
    Used so each model's trajectories start from a consistently centred origin.
    """
    return pca.transform(z - z_ref.mean(axis=0, keepdims=True))


# =============================================================================
# Plotting
# =============================================================================

def plot_trajectory_panel(
    ax:            plt.Axes,
    traj_2d:       list[np.ndarray],   # list[(iters+1)] of (B, 2)
    proto_2d:      np.ndarray,         # (n_proto, 2)
    proto_lbl:     np.ndarray,         # (n_proto,)
    y:             np.ndarray,         # (B,) class labels
    title:         str,
) -> None:
    """
    Draw one panel: trajectory lines + start/end markers + prototypes.
    """
    cmap      = plt.get_cmap(CMAP)
    n_samples = traj_2d[0].shape[0]

    for i in range(n_samples):
        cls   = int(y[i])
        color = cmap(cls / N_CLASSES)

        # Full trajectory as a thin line
        traj_i = np.stack([step[i] for step in traj_2d], axis=0)  # (iters+1, 2)
        ax.plot(
            traj_i[:, 0], traj_i[:, 1],
            color=color, linewidth=0.9, alpha=0.65, zorder=2,
        )

        # Start: clean point (filled circle)
        ax.scatter(
            traj_i[0, 0], traj_i[0, 1],
            s=40, marker="o", color=color,
            edgecolors="black", linewidths=0.5, zorder=4,
        )
        # End: adversarial point (cross)
        ax.scatter(
            traj_i[-1, 0], traj_i[-1, 1],
            s=60, marker="X", color=color,
            edgecolors="black", linewidths=0.5, zorder=4,
        )

    # Prototypes
    for p_idx in range(len(proto_2d)):
        cls   = int(proto_lbl[p_idx])
        color = cmap(cls / N_CLASSES)
        ax.scatter(
            proto_2d[p_idx, 0], proto_2d[p_idx, 1],
            s=140, marker="*", color=color,
            edgecolors="black", linewidths=0.7, zorder=5,
        )

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("PC1", fontsize=8)
    ax.set_ylabel("PC2", fontsize=8)
    ax.tick_params(labelsize=7)


def plot_trajectories(
    model_a_name:  str,
    model_b_name:  str,
    traj_a:        list[np.ndarray],
    traj_b:        list[np.ndarray],
    z_clean_a:     np.ndarray,
    z_clean_b:     np.ndarray,
    proto_a:       np.ndarray,
    proto_b:       np.ndarray,
    proto_lbl_a:   np.ndarray,
    proto_lbl_b:   np.ndarray,
    pca:           PCA,
    y:             np.ndarray,
    eps:           float,
    iters:         int,
    out_dir:       str,
) -> None:
    """
    Build the full two-panel figure and save.
    """
    mean_a = z_clean_a.mean(axis=0, keepdims=True)
    mean_b = z_clean_b.mean(axis=0, keepdims=True)

    def _proj_traj(traj, mean):
        return [pca.transform(step - mean) for step in traj]

    traj_a_2d  = _proj_traj(traj_a, mean_a)
    traj_b_2d  = _proj_traj(traj_b, mean_b)
    proto_a_2d = pca.transform(proto_a - mean_a)
    proto_b_2d = pca.transform(proto_b - mean_b)

    label_a = VARIANT_LABELS.get(model_a_name, model_a_name)
    label_b = VARIANT_LABELS.get(model_b_name, model_b_name)

    var_exp = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    plot_trajectory_panel(
        axes[0], traj_a_2d, proto_a_2d, proto_lbl_a, y,
        title=f"{label_a}  (ε={eps}, {iters} steps)",
    )
    plot_trajectory_panel(
        axes[1], traj_b_2d, proto_b_2d, proto_lbl_b, y,
        title=f"{label_b}  (ε={eps}, {iters} steps)",
    )

    for ax in axes:
        ax.set_xlabel(f"PC1 ({var_exp[0]:.1%})", fontsize=8)
        ax.set_ylabel(f"PC2 ({var_exp[1]:.1%})", fontsize=8)

    # Shared legend
    cmap = plt.get_cmap(CMAP)
    class_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=cmap(c / N_CLASSES),
                   markersize=7, label=str(c))
        for c in range(N_CLASSES)
    ]
    marker_handles = [
        plt.Line2D([0], [0], marker="o",  color="gray", markersize=7,
                   linestyle="None", label="Clean start (●)"),
        plt.Line2D([0], [0], marker="X",  color="gray", markersize=7,
                   linestyle="None", label="Adversarial end (✗)"),
        plt.Line2D([0], [0], marker="*",  color="gray", markersize=9,
                   linestyle="None", label="Prototype (★)"),
    ]
    fig.legend(
        handles=class_handles + marker_handles,
        loc="lower center", ncol=N_CLASSES + 3,
        fontsize=7, bbox_to_anchor=(0.5, -0.06),
        title="Class (colour)  |  Marker legend",
    )

    fig.suptitle(
        f"PGD trajectories in latent space — {label_a} vs {label_b}\n"
        "PCA fit on joint clean codes (mean-centred per model)",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fname = f"ch6_pca_trajectories_{model_a_name}_vs_{model_b_name}.pdf"
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 6.3 — PGD adversarial trajectories in latent space."
    )
    parser.add_argument("--model_a",   type=str, default="B30",
                        help="First model (default: B30 base).")
    parser.add_argument("--model_b",   type=str, default="B30-FT-0",
                        help="Second model (default: B30-FT-0).")
    parser.add_argument("--n_samples", type=int, default=N_TRAJ_SAMPLES,
                        help="Number of trajectory samples per panel.")
    parser.add_argument("--eps",       type=float, default=PGD_EPS_DEFAULT)
    parser.add_argument("--alpha",     type=float, default=PGD_ALPHA)
    parser.add_argument("--pgd_iters", type=int,   default=PGD_ITERS)
    parser.add_argument("--seed",      type=int,   default=DEFAULT_SEED)
    args = parser.parse_args()

    plt.rcParams.update(RC_PARAMS)
    out_dir = os.path.join(FIGURES_ROOT, "ch6")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load test subset
    # ------------------------------------------------------------------
    print(f"\nLoading {args.n_samples} test samples …")
    x_sub, y_sub = get_test_subset(args.n_samples, args.seed)
    x_sub = x_sub.to(device)
    y_sub = y_sub.to(device)
    y_np  = y_sub.cpu().numpy()

    # ------------------------------------------------------------------
    # 2. Load both models
    # ------------------------------------------------------------------
    print(f"Loading {args.model_a} …")
    model_a = _load_b30(args.model_a, device)
    print(f"Loading {args.model_b} …")
    model_b = _load_b30(args.model_b, device)

    # ------------------------------------------------------------------
    # 3. Extract clean latent codes and prototypes for both models
    # ------------------------------------------------------------------
    print("Extracting clean latent codes …")
    with torch.no_grad():
        z_clean_a = extract_internals(model_a, x_sub)["z"].cpu().numpy()
        z_clean_b = extract_internals(model_b, x_sub)["z"].cpu().numpy()

    proto_lbl_a = get_proto_labels(model_a).numpy()
    proto_lbl_b = get_proto_labels(model_b).numpy()
    proto_a     = model_a.prototype_layer.prototype_distances.detach().cpu().numpy()
    proto_b     = model_b.prototype_layer.prototype_distances.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # 4. Fit joint PCA on clean codes of both models
    # ------------------------------------------------------------------
    print("Fitting joint PCA …")
    pca = fit_joint_pca(z_clean_a, z_clean_b)
    print(
        f"  Explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, "
        f"PC2={pca.explained_variance_ratio_[1]:.1%}"
    )

    # ------------------------------------------------------------------
    # 5. Run PGD with trajectory capture for both models
    # ------------------------------------------------------------------
    print(f"\nRunning PGD (eps={args.eps}, iters={args.pgd_iters}) on {args.model_a} …")
    _, traj_a = pgd_with_trajectory(
        model_a, x_sub,
        loss_f    = _b30_loss_f(model_a, y_sub),
        iters     = args.pgd_iters,
        eps       = args.eps,
        alpha     = args.alpha,
        device    = device,
        extract_fn= extract_internals,
    )

    print(f"Running PGD (eps={args.eps}, iters={args.pgd_iters}) on {args.model_b} …")
    _, traj_b = pgd_with_trajectory(
        model_b, x_sub,
        loss_f    = _b30_loss_f(model_b, y_sub),
        iters     = args.pgd_iters,
        eps       = args.eps,
        alpha     = args.alpha,
        device    = device,
        extract_fn= extract_internals,
    )

    # ------------------------------------------------------------------
    # 6. Plot
    # ------------------------------------------------------------------
    print("\nGenerating trajectory figure …")
    plot_trajectories(
        model_a_name = args.model_a,
        model_b_name = args.model_b,
        traj_a       = traj_a,
        traj_b       = traj_b,
        z_clean_a    = z_clean_a,
        z_clean_b    = z_clean_b,
        proto_a      = proto_a,
        proto_b      = proto_b,
        proto_lbl_a  = proto_lbl_a,
        proto_lbl_b  = proto_lbl_b,
        pca          = pca,
        y            = y_np,
        eps          = args.eps,
        iters        = args.pgd_iters,
        out_dir      = out_dir,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
