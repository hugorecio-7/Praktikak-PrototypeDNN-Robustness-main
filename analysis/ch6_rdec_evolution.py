"""
ch6_rdec_evolution.py
=====================
Chapter 6 — Decoder reconstruction drift under PGD attack.

What this script produces
--------------------------
Three PDFs per selected sample, designed to be composed as a single LaTeX
figure with a shared caption and label:

  ch6_rdec_{model}_s{idx}_clean.pdf   — clean reconstruction (ε = 0)
  ch6_rdec_{model}_s{idx}_low.pdf    — 1×4 row, ε = 0.1, 0.2, 0.3, 0.4
  ch6_rdec_{model}_s{idx}_high.pdf   — 1×4 row, ε = 0.5, 0.6, 0.7, 0.8

R_dec = ||x̂_adv − x̂_clean||_2 is annotated below each adversarial panel.
All three PDFs use the same panel size for easy alignment in LaTeX.

Supported architectures
------------------------
B30 / B30-FT-*   : encoder → z → decoder → x̂ ∈ [0,1]
ProtoVAE / ProtoVAE-FT-* : features → μ → decoder (tanh) → x̂ ∈ [0,1]
(SENN has no decoder — not supported.)

Both cases are handled by extract_internals() which already returns x_hat
in [0,1] for both architectures.

LaTeX usage (three PDFs as one figure)
---------------------------------------
    \\begin{figure}[h]
        \\centering
        \\subfloat{\\includegraphics[height=3cm]{ch6_rdec_B30_s0_clean.pdf}}%
        \\hspace{0.3cm}%
        \\subfloat{\\includegraphics[height=3cm]{ch6_rdec_B30_s0_low.pdf}}%
        \\hspace{0.3cm}%
        \\subfloat{\\includegraphics[height=3cm]{ch6_rdec_B30_s0_high.pdf}}
        \\caption{Decoder reconstruction drift under PGD-$L_\\infty$ attack.}
        \\label{fig:ch6_rdec_B30}
    \\end{figure}
    (Requires \\usepackage{subfig})

Usage
-----
    # B30 base model, sample 0:
    python ch6_rdec_evolution.py --model B30

    # ProtoVAE base model, samples 0 and 7:
    python ch6_rdec_evolution.py --model ProtoVAE --sample_idx 0 7

    # B30-FT-1 variant:
    python ch6_rdec_evolution.py --model B30-FT-1 --sample_idx 0
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

from config import RC_PARAMS, FIGURES_ROOT, CKPT_PATHS, DEFAULT_SEED
import sys
path = str(Path(__file__).parent.parent)
sys.path.append(path)
from metric_extractor import extract_internals


# =============================================================================
# Constants
# =============================================================================

EPS_LOW  = [round(e, 3) for e in np.arange(0.1, 0.5, 0.1)]   # 0.1 0.2 0.3 0.4
EPS_HIGH = [round(e, 3) for e in np.arange(0.5, 0.9, 0.1)]   # 0.5 0.6 0.7 0.8

PGD_ITERS = 40
PGD_ALPHA = 0.01

# Fixed panel size in inches — keep consistent across all three PDFs so they
# align when placed side-by-side in LaTeX with the same height= value.
PANEL_SIZE = 2.4


# =============================================================================
# Architecture detection helpers
# =============================================================================

def _detect_arch(model_name: str) -> str:
    """
    Return 'B30' or 'ProtoVAE' from the model name string.
    Raises ValueError for unsupported architectures (SENN).
    """
    if model_name.startswith("ProtoVAE"):
        return "ProtoVAE"
    if model_name.startswith("B30") or model_name == "B30":
        return "B30"
    raise ValueError(
        f"[rdec_evolution] Cannot detect arch from model name '{model_name}'.\n"
        f"  SENN is not supported (no decoder). Use a B30 or ProtoVAE variant."
    )


# =============================================================================
# Model loading
# =============================================================================

def _load_b30(model_name: str, device: torch.device) -> nn.Module:
    """
    Load B30 or any B30-FT variant.
    Base: full object checkpoint.
    FT:  base architecture + FT state_dict.
    """
    base_path = CKPT_PATHS["B30"]
    try:
        model = torch.load(base_path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(base_path, map_location=device)

    if model_name != "B30":
        state = torch.load(CKPT_PATHS[model_name], map_location=device)
        model.load_state_dict(state["model_state"])

    return model.to(device).eval()


def _load_protovae(model_name: str, device: torch.device) -> nn.Module:
    """
    Load ProtoVAE or any ProtoVAE-FT variant.
    """
    from ProtoVAE import model as model_module
    model = model_module.ProtoVAE().to(device)
    state = torch.load(CKPT_PATHS["ProtoVAE"], map_location=device)
    model.load_state_dict(state)

    if model_name != "ProtoVAE":
        state_ft = torch.load(CKPT_PATHS[model_name], map_location=device)
        model.load_state_dict(state_ft["model_state"])

    return model.eval()


def load_model(model_name: str, device: torch.device) -> nn.Module:
    """Load and return model in eval mode, architecture-aware."""
    arch = _detect_arch(model_name)
    if arch == "B30":
        return _load_b30(model_name, device)
    return _load_protovae(model_name, device)


# =============================================================================
# Forward pass for PGD (with gradients, architecture-aware)
# =============================================================================

def _forward_pgd(model: nn.Module, x: torch.Tensor, arch: str) -> torch.Tensor:
    """
    Compute logits with gradients enabled for PGD loss.

    B30      : model(x) works directly (input in [0,1]).
    ProtoVAE : must normalise to [-1,1] and replicate pred_class path.
               Cannot use extract_internals here because it wraps no_grad.
    """
    if arch == "B30":
        return model(x)

    # ProtoVAE prediction path (mirrors _extract_protovae in metric_extractor.py)
    x_norm = x * 2.0 - 1.0
    latent = model.prototype_vectors.shape[1]
    conv_features = model.features(x_norm)
    mu = conv_features[:, :latent]
    sim_scores = model.calc_sim_scores(mu)
    return model.last_layer(sim_scores)


# =============================================================================
# Data
# =============================================================================

def load_test_set():
    """Return (x_all, y_all) for the full MNIST test set."""
    from data_loader import get_test_loader as _gtl
    loader = _gtl("data", batch_size=500, shuffle=False,
                  num_workers=0, pin_memory=False)
    xs, ys = [], []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


# =============================================================================
# Reconstruction helpers
# =============================================================================

@torch.no_grad()
def get_clean_reconstruction(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Return x̂_clean from the decoder. x̂ ∈ [0,1] for all architectures."""
    internals = extract_internals(model, x.to(device))
    return internals["x_hat"].cpu()


def get_adv_reconstruction(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    arch: str,
    device: torch.device,
    pgd_iters: int = PGD_ITERS,
    pgd_alpha: float = PGD_ALPHA,
) -> tuple[torch.Tensor, float]:
    """
    Attack x with PGD, decode the adversarial z, and compute R_dec.

    Returns
    -------
    x_hat_adv : (B, C, H, W) in [0,1]
    r_dec     : mean R_dec over the batch
    """
    from adversarial_attacks import PGDLInf_attack

    model.eval()
    xb = x.to(device)
    yb = y.to(device)

    def loss_f(x_in: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(_forward_pgd(model, x_in, arch), yb)

    x_adv = PGDLInf_attack(
        xb, loss_f,
        iters=pgd_iters,
        eps=eps,
        alpha=pgd_alpha,
        random_start=True,
    )

    with torch.no_grad():
        x_hat_adv   = extract_internals(model, x_adv)["x_hat"].cpu()
        x_hat_clean = extract_internals(model, xb)["x_hat"].cpu()

    flat_adv   = x_hat_adv.view(len(x_hat_adv), -1).numpy().astype(np.float64)
    flat_clean = x_hat_clean.view(len(x_hat_clean), -1).numpy().astype(np.float64)
    r_dec = float(np.linalg.norm(flat_adv - flat_clean, axis=1).mean())

    return x_hat_adv, r_dec


# =============================================================================
# Plotting helpers
# =============================================================================

def _img_np(t: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) or (C, H, W) tensor → (H, W) numpy array clipped to [0,1]."""
    return np.clip(t.squeeze().numpy(), 0.0, 1.0)


def _draw_panel(ax: plt.Axes, img: np.ndarray,
                label_top: str, label_bot: str) -> None:
    ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(label_top, fontsize=9, pad=3)
    ax.set_xlabel(label_bot, fontsize=8, labelpad=3)
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_clean_pdf(
    x_hat_clean: torch.Tensor,
    model_name: str,
    sample_idx: int,
    out_dir: str,
) -> None:
    """Single-panel PDF: clean reconstruction."""
    fig, ax = plt.subplots(1, 1, figsize=(PANEL_SIZE, PANEL_SIZE + 0.5))
    _draw_panel(
        ax,
        _img_np(x_hat_clean),
        label_top=r"$\varepsilon = 0$  (clean)",
        label_bot=r"$R_{dec} = 0.000$",
    )
    plt.tight_layout(pad=0.3)
    fpath = os.path.join(out_dir, f"ch6_rdec_{model_name}_s{sample_idx}_clean.pdf")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


def save_row_pdf(
    recons: list[tuple[float, torch.Tensor, float]],
    model_name: str,
    sample_idx: int,
    suffix: str,
    out_dir: str,
) -> None:
    """
    1×N row of adversarial reconstructions.

    Parameters
    ----------
    recons : list of (eps, x_hat_adv, r_dec)
    suffix : "low" or "high" — used in filename
    """
    n = len(recons)
    fig, axes = plt.subplots(1, n, figsize=(PANEL_SIZE * n, PANEL_SIZE + 0.5))
    if n == 1:
        axes = [axes]

    for ax, (eps, x_hat_adv, r_dec) in zip(axes, recons):
        _draw_panel(
            ax,
            _img_np(x_hat_adv),
            label_top=rf"$\varepsilon = {eps:.1f}$",
            label_bot=rf"$R_{{dec}} = {r_dec:.3f}$",
        )

    plt.tight_layout(pad=0.3)
    fpath = os.path.join(out_dir, f"ch6_rdec_{model_name}_s{sample_idx}_{suffix}.pdf")
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(fpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {fpath}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 6 — Decoder reconstruction drift visualisation."
    )
    parser.add_argument(
        "--model", type=str, default="B30",
        help=(
            "Model name as in CKPT_PATHS. Examples:\n"
            "  B30, B30-FT-0, B30-FT-1\n"
            "  ProtoVAE, ProtoVAE-FT-0, ProtoVAE-FT-1\n"
            "SENN is not supported (no decoder)."
        ),
    )
    parser.add_argument(
        "--sample_idx", type=int, nargs="+", default=[0],
        help="Test-set indices to visualise. Multiple values produce one set of PDFs each.",
    )
    parser.add_argument("--pgd_iters", type=int, default=PGD_ITERS)
    parser.add_argument("--pgd_alpha", type=float, default=PGD_ALPHA)

    args = parser.parse_args()

    plt.rcParams.update(RC_PARAMS)
    out_dir = os.path.join(FIGURES_ROOT, "ch6")
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Detect architecture before loading anything.
    arch = _detect_arch(args.model)
    print(f"Architecture: {arch}")

    # Load model.
    print(f"Loading {args.model} …")
    model = load_model(args.model, device)

    # Load test set.
    print("Loading MNIST test set …")
    x_all, y_all = load_test_set()
    print(f"  {len(x_all)} samples loaded.")

    for sample_idx in args.sample_idx:
        if sample_idx >= len(x_all):
            print(f"  [SKIP] sample_idx={sample_idx} out of range ({len(x_all)} total).")
            continue

        x_single = x_all[[sample_idx]]   # (1, C, H, W)
        y_single = y_all[[sample_idx]]   # (1,)
        print(f"\nSample {sample_idx}  (class {int(y_single.item())})")

        # Clean reconstruction.
        x_hat_clean = get_clean_reconstruction(model, x_single, device)

        # Adversarial reconstructions.
        recons_low : list[tuple[float, torch.Tensor, float]] = []
        recons_high: list[tuple[float, torch.Tensor, float]] = []

        for eps in EPS_LOW:
            print(f"  PGD ε={eps:.1f} …", end=" ", flush=True)
            x_hat_adv, r_dec = get_adv_reconstruction(
                model, x_single, y_single, eps, arch, device,
                args.pgd_iters, args.pgd_alpha,
            )
            print(f"R_dec = {r_dec:.4f}")
            recons_low.append((eps, x_hat_adv, r_dec))

        for eps in EPS_HIGH:
            print(f"  PGD ε={eps:.1f} …", end=" ", flush=True)
            x_hat_adv, r_dec = get_adv_reconstruction(
                model, x_single, y_single, eps, arch, device,
                args.pgd_iters, args.pgd_alpha,
            )
            print(f"R_dec = {r_dec:.4f}")
            recons_high.append((eps, x_hat_adv, r_dec))

        # Save the three PDFs.
        save_clean_pdf(x_hat_clean, args.model, sample_idx, out_dir)
        save_row_pdf(recons_low,  args.model, sample_idx, "low",  out_dir)
        save_row_pdf(recons_high, args.model, sample_idx, "high", out_dir)

    print("\nDone.")
    print(
        "\nLaTeX hint — three PDFs as one figure with shared caption:\n"
        "  \\subfloat{\\includegraphics[height=Xcm]{..._clean.pdf}}%\n"
        "  \\hspace{0.3cm}%\n"
        "  \\subfloat{\\includegraphics[height=Xcm]{..._low.pdf}}%\n"
        "  \\hspace{0.3cm}%\n"
        "  \\subfloat{\\includegraphics[height=Xcm]{..._high.pdf}}\n"
        "Use the same Xcm in all three for perfect alignment."
    )


if __name__ == "__main__":
    main()
