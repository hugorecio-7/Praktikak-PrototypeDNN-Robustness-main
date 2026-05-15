"""
ch6_pca_trajectories_autoattack.py
==================================
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
    PGD is the trajectory-based probe and provides the main quantitative
    latent dynamics metrics.

    AutoAttack is used only as a successful-endpoint validator. Standard
    AutoAttack returns unchanged samples when it does not find a successful
    adversarial example, so AutoAttack latent displacement is not exported as
    a trajectory metric. Successful AutoAttack endpoints are plotted only when
    available.

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
    python ch6_pca_trajectories_autoattack.py

    # Custom pair
    python ch6_pca_trajectories_autoattack.py --model_a B30 --model_b B30-FT-E

    # More trajectory samples, higher PGD iterations
    python ch6_pca_trajectories_autoattack.py --n_samples 20 --pgd_iters 50

    # Use the global PCA saved by ch6_pca_training.py
    python ch6_pca_trajectories_autoattack.py --pca_path analysis/figures/ch6/ch6_pca_training_global_B30_pca.pkl
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
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
from adversarial_attacks import AutoAttack_adv
from metric_calculators import (
    compute_pgd_final_shift,
    compute_pgd_path_efficiency,
    compute_pgd_path_length,
)


# =============================================================================
# Constants
# =============================================================================

N_TRAJ_SAMPLES   = 10      # number of trajectories to draw per panel
PGD_EPS_DEFAULT  = 0.3     # epsilon for trajectory visualisation
PGD_ALPHA        = 0.01
PGD_ITERS        = 80
CMAP             = "tab10"
N_CLASSES        = 10


# =============================================================================
# Model loading  (mirrors run_metrics.py exactly)
# =============================================================================

def _infer_arch(model_name: str) -> str:
    if model_name == "B30" or model_name.startswith("B30-"):
        return "B30"
    if model_name == "ProtoVAE" or model_name.startswith("ProtoVAE-"):
        return "ProtoVAE"
    raise ValueError(
        f"Unsupported model '{model_name}'. Use a B30 or ProtoVAE variant."
    )


def _checkpoint_model_state(state):
    return state.get("model_state", state) if isinstance(state, dict) else state


def _load_b30(model_name: str, device: torch.device) -> nn.Module:
    base_path = CKPT_PATHS["B30"]
    try:
        model = torch.load(base_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(base_path, map_location=device)

    if model_name != "B30":
        ckpt  = torch.load(CKPT_PATHS[model_name], map_location=device)
        model.load_state_dict(_checkpoint_model_state(ckpt))

    return model.to(device).eval()


class ProtoVAEWrapper(nn.Module):
    """Evaluation wrapper matching run_metrics.py: accepts x in [0, 1]."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model
        self.input_bounds = (0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * 2 - 1
        logits, _ = self.base.pred_class(x)
        return logits


def _load_protovae(model_name: str, device: torch.device) -> nn.Module:
    from ProtoVAE import model as model_protovae

    model = model_protovae.ProtoVAE().to(device)
    state = torch.load(CKPT_PATHS[model_name], map_location=device)
    model.load_state_dict(_checkpoint_model_state(state))
    return ProtoVAEWrapper(model).to(device).eval()


def load_model(model_name: str, device: torch.device) -> nn.Module:
    arch = _infer_arch(model_name)
    if arch == "B30":
        return _load_b30(model_name, device)
    if arch == "ProtoVAE":
        return _load_protovae(model_name, device)
    raise AssertionError(f"Unhandled architecture '{arch}'.")


def get_prototype_vectors(model: nn.Module) -> np.ndarray:
    base = model.base if hasattr(model, "base") else model
    if hasattr(base, "prototype_vectors"):
        return base.prototype_vectors.detach().cpu().numpy()
    if hasattr(base, "prototype_layer"):
        return base.prototype_layer.prototype_distances.detach().cpu().numpy()
    raise ValueError(f"Model {type(base).__name__} has no prototype vectors.")


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


def autoattack_reference_points(
    model:   nn.Module,
    x:       torch.Tensor,
    y:       torch.Tensor,
    eps:     float,
    version: str,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """
    Run AutoAttack and return final examples, z, and success mask.

    AutoAttack does not expose a trajectory. In the standard implementation,
    unsuccessful samples are returned unchanged, so this script only treats
    AutoAttack as a successful-endpoint validator.
    """
    x_adv = AutoAttack_adv(
        batch_x=x,
        batch_y=y,
        model=model,
        epsilon=eps,
        version=version,
    )
    with torch.no_grad():
        z_auto = extract_internals(model, x_adv)["z"].cpu().numpy()
        pred_auto = model(x_adv).argmax(dim=1)
    aa_success = (pred_auto != y).detach().cpu().numpy().astype(bool)
    return x_adv, z_auto, aa_success


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


def load_pca(pca_path: str | Path) -> PCA:
    """Load a fitted PCA object saved by ch6_pca_training.py."""
    with open(pca_path, "rb") as f:
        return pickle.load(f)


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
    auto_2d:       np.ndarray,         # (B, 2), AutoAttack final points
    aa_success:    np.ndarray,         # (B,), successful AutoAttack endpoints
    proto_2d:      np.ndarray,         # (n_proto, 2)
    proto_lbl:     np.ndarray,         # (n_proto,)
    y:             np.ndarray,         # (B,) class labels
    title:         str,
) -> None:
    """
    Draw one panel: PGD trajectory markers, successful AutoAttack endpoints,
    and prototypes.
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
        # AutoAttack is plotted only when a successful endpoint exists.
        if aa_success[i]:
            ax.scatter(
                auto_2d[i, 0], auto_2d[i, 1],
                s=54, marker="D", facecolors="none",
                edgecolors=color, linewidths=1.4, zorder=6,
            )

    n_success = int(aa_success.sum())
    aa_text = (
        "AutoAttack: no successful endpoint"
        if n_success == 0
        else f"AutoAttack endpoints: {n_success}/{n_samples}"
    )
    ax.text(
        0.02, 0.98, aa_text,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=8,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=2),
        zorder=7,
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
    z_auto_a:      np.ndarray,
    z_auto_b:      np.ndarray,
    aa_success_a:  np.ndarray,
    aa_success_b:  np.ndarray,
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
    auto_version:  str,
    out_dir:       str,
    use_global_pca: bool = False,
) -> str:
    """
    Build the full two-panel figure and save.
    """
    mean_a = None if use_global_pca else z_clean_a.mean(axis=0, keepdims=True)
    mean_b = None if use_global_pca else z_clean_b.mean(axis=0, keepdims=True)

    def _proj_traj(traj, mean):
        return [pca.transform(step if mean is None else step - mean) for step in traj]

    def _project(z, mean):
        return pca.transform(z if mean is None else z - mean)

    traj_a_2d  = _proj_traj(traj_a, mean_a)
    traj_b_2d  = _proj_traj(traj_b, mean_b)
    auto_a_2d  = _project(z_auto_a, mean_a)
    auto_b_2d  = _project(z_auto_b, mean_b)
    proto_a_2d = _project(proto_a, mean_a)
    proto_b_2d = _project(proto_b, mean_b)

    label_a = VARIANT_LABELS.get(model_a_name, model_a_name)
    label_b = VARIANT_LABELS.get(model_b_name, model_b_name)

    var_exp = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    plot_trajectory_panel(
        axes[0], traj_a_2d, auto_a_2d, aa_success_a, proto_a_2d, proto_lbl_a, y,
        title=f"{label_a}  (ε={eps}, {iters} steps)",
    )
    plot_trajectory_panel(
        axes[1], traj_b_2d, auto_b_2d, aa_success_b, proto_b_2d, proto_lbl_b, y,
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
        plt.Line2D([0], [0], marker="D",  color="gray", markersize=7,
                   markerfacecolor="none", linestyle="None",
                   label="Successful AutoAttack endpoint (D)"),
        plt.Line2D([0], [0], marker="*",  color="gray", markersize=9,
                   linestyle="None", label="Prototype (★)"),
    ]
    fig.legend(
        handles=class_handles + marker_handles,
        loc="lower center", ncol=N_CLASSES + 4,
        fontsize=7, bbox_to_anchor=(0.5, -0.06),
        title="Class (colour)  |  Marker legend",
    )

    pca_title = (
        "PCA loaded from global clean latent space analysis"
        if use_global_pca
        else "PCA fit on joint clean codes"
    )
    fig.suptitle(
        f"PGD trajectories in latent space — {label_a} vs {label_b}\n"
        f"{pca_title}; AutoAttack={auto_version}",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fname = f"ch6_pca_trajectories_autoattack_{model_a_name}_vs_{model_b_name}.pdf"
    fpath = os.path.join(out_dir, fname)
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {fpath}")
    return fpath


# =============================================================================
# Metric export
# =============================================================================

def _mean_std(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std(ddof=1)) if values.size > 1 else 0.0
    return float(values.mean()), std


def _trajectory_metric_row(
    model_name: str,
    eps: float,
    pgd_iters: int,
    z_clean: np.ndarray,
    traj: list[np.ndarray],
    aa_success: np.ndarray,
) -> dict[str, float | int | str]:
    pgd_final_shift = compute_pgd_final_shift(z_clean, traj[-1])
    path_length = compute_pgd_path_length(traj)
    path_efficiency = compute_pgd_path_efficiency(z_clean, traj[-1], traj)
    aa_success = np.asarray(aa_success, dtype=bool)

    pgd_final_mean, pgd_final_std = _mean_std(pgd_final_shift)
    path_mean, path_std = _mean_std(path_length)
    eff_mean, eff_std = _mean_std(path_efficiency)

    return {
        "model": model_name,
        "eps": float(eps),
        "pgd_iters": int(pgd_iters),
        "pgd_final_shift_mean": pgd_final_mean,
        "pgd_final_shift_std": pgd_final_std,
        "pgd_path_length_mean": path_mean,
        "pgd_path_length_std": path_std,
        "pgd_path_efficiency_mean": eff_mean,
        "pgd_path_efficiency_std": eff_std,
        "aa_success_count": int(aa_success.sum()),
        "aa_success_rate": float(aa_success.mean()) if aa_success.size else 0.0,
    }


def save_trajectory_metrics_csv(
    pdf_path: str,
    model_a_name: str,
    model_b_name: str,
    eps: float,
    pgd_iters: int,
    z_clean_a: np.ndarray,
    z_clean_b: np.ndarray,
    traj_a: list[np.ndarray],
    traj_b: list[np.ndarray],
    aa_success_a: np.ndarray,
    aa_success_b: np.ndarray,
) -> str:
    """Save PGD latent trajectory metrics plus AutoAttack endpoint metadata."""
    rows = [
        _trajectory_metric_row(
            model_a_name, eps, pgd_iters, z_clean_a, traj_a, aa_success_a
        ),
        _trajectory_metric_row(
            model_b_name, eps, pgd_iters, z_clean_b, traj_b, aa_success_b
        ),
    ]

    fieldnames = [
        "model",
        "eps",
        "pgd_iters",
        "pgd_final_shift_mean",
        "pgd_final_shift_std",
        "pgd_path_length_mean",
        "pgd_path_length_std",
        "pgd_path_efficiency_mean",
        "pgd_path_efficiency_std",
        "aa_success_count",
        "aa_success_rate",
    ]

    metrics_dir = os.path.join(os.path.dirname(pdf_path), "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    stem = Path(pdf_path).stem
    csv_path = os.path.join(metrics_dir, f"{stem}_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Saved metrics -> {csv_path}")
    return csv_path


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
    parser.add_argument("--autoattack_version", type=str, default="standard",
                        help="AutoAttack version used for endpoint validation.")
    parser.add_argument("--pca_path",  type=str,   default=None,
                        help="Optional global PCA .pkl from ch6_pca_training.py.")
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
    model_a = load_model(args.model_a, device)
    print(f"Loading {args.model_b} …")
    model_b = load_model(args.model_b, device)

    # ------------------------------------------------------------------
    # 3. Extract clean latent codes and prototypes for both models
    # ------------------------------------------------------------------
    print("Extracting clean latent codes …")
    with torch.no_grad():
        z_clean_a = extract_internals(model_a, x_sub)["z"].cpu().numpy()
        z_clean_b = extract_internals(model_b, x_sub)["z"].cpu().numpy()

    proto_lbl_a = get_proto_labels(model_a).numpy()
    proto_lbl_b = get_proto_labels(model_b).numpy()
    proto_a     = get_prototype_vectors(model_a)
    proto_b     = get_prototype_vectors(model_b)

    # ------------------------------------------------------------------
    # 4. Fit local PCA or load global PCA
    # ------------------------------------------------------------------
    use_global_pca = args.pca_path is not None
    if use_global_pca:
        print(f"Loading global PCA from {args.pca_path} ...")
        pca = load_pca(args.pca_path)
    else:
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
    # 6. Run AutoAttack successful endpoint validation for the same samples
    # ------------------------------------------------------------------
    print(
        f"Running AutoAttack ({args.autoattack_version}, eps={args.eps}) "
        f"on {args.model_a} ..."
    )
    _, z_auto_a, aa_success_a = autoattack_reference_points(
        model=model_a,
        x=x_sub,
        y=y_sub,
        eps=args.eps,
        version=args.autoattack_version,
    )

    print(
        f"Running AutoAttack ({args.autoattack_version}, eps={args.eps}) "
        f"on {args.model_b} ..."
    )
    _, z_auto_b, aa_success_b = autoattack_reference_points(
        model=model_b,
        x=x_sub,
        y=y_sub,
        eps=args.eps,
        version=args.autoattack_version,
    )

    # ------------------------------------------------------------------
    # 7. Plot
    # ------------------------------------------------------------------
    print("\nGenerating trajectory figure …")
    pdf_path = plot_trajectories(
        model_a_name = args.model_a,
        model_b_name = args.model_b,
        traj_a       = traj_a,
        traj_b       = traj_b,
        z_auto_a     = z_auto_a,
        z_auto_b     = z_auto_b,
        aa_success_a = aa_success_a,
        aa_success_b = aa_success_b,
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
        auto_version = args.autoattack_version,
        out_dir      = out_dir,
        use_global_pca = use_global_pca,
    )

    # ------------------------------------------------------------------
    # 8. Save PGD original-latent-space metrics and AutoAttack endpoint metadata
    # ------------------------------------------------------------------
    save_trajectory_metrics_csv(
        pdf_path=pdf_path,
        model_a_name=args.model_a,
        model_b_name=args.model_b,
        eps=args.eps,
        pgd_iters=args.pgd_iters,
        z_clean_a=z_clean_a,
        z_clean_b=z_clean_b,
        traj_a=traj_a,
        traj_b=traj_b,
        aa_success_a=aa_success_a,
        aa_success_b=aa_success_b,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
