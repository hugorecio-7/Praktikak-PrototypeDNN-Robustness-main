"""
config.py
=========
Central configuration for all analysis scripts (chapters 4–7).

The only file you need to edit when:
  - Paths change (OUT_ROOT, model checkpoints).
  - You add a new seed or run_id.
  - You want to change plot aesthetics globally.

All other scripts import from here and never hard-code paths or names.
"""

from __future__ import annotations
import numpy as np

# =============================================================================
# 1. Result tree — must match OUT_ROOT / DATASET_NAME / THREAT_NAME in run_metrics.py
# =============================================================================
OUT_ROOT      = "resultsTFG/runs_v1"
DATASET_NAME  = "mnist"
THREAT_NAME   = "Linf"

# Default seed and run_id used when loading results.
# Override per-call in loaders.py if you ran multiple seeds.
DEFAULT_SEED   = 1
DEFAULT_RUN_ID = None   # None → loaders will pick the only/latest run_id found

# =============================================================================
# 2. Model checkpoint paths  (mirrors paths dict in run_metrics.py)
# =============================================================================
CKPT_PATHS: dict[str, str] = {
    # --- B30 ---
    "B30":      "saved_model/mnist_model/mnist_cae_balanced_clstsep_1500_0.002_250_True_0.0_20_1_1_1_1.0_0.0_30_4_32_1/mnist_cae00750.pth",
    "B30-FT-0": "tfg_models/B30/B30-FT-0/seed=1/checkpoints/B30_B30-FT-0_seed1_best_val_adv_acc.pth",
    "B30-FT-1": "tfg_models/B30/B30-FT-1/seed=1/checkpoints/B30_B30-FT-1_seed1_best_val_adv_acc.pth",
    "B30-FT-2": "tfg_models/B30/B30-FT-2/seed=1/checkpoints/B30_B30-FT-2_seed1_best_val_adv_acc.pth",
    "B30-FT-3": "tfg_models/B30/B30-FT-3/seed=1/checkpoints/B30_B30-FT-3_seed1_best_val_adv_acc.pth",
    "B30-FT-E": "tfg_models/B30/B30-FT-E/seed=1/checkpoints/B30_B30-FT-E_seed1_best_val_adv_acc.pth",
    "B30-FT-E-M": "tfg_models/B30/B30-FT-E-M/seed=1_loweps005_pca/checkpoints/B30_B30-FT-E_seed1_best_val_adv_acc.pth",
    # --- ProtoVAE ---
    "ProtoVAE":      "ProtoVAE/saved_models/mnist/model.pth",
    "ProtoVAE-FT-0": "tfg_models/ProtoVAE/ProtoVAE-FT-0/seed=1/checkpoints/ProtoVAE_ProtoVAE-FT-0_seed1_best_val_adv_acc.pth",
    "ProtoVAE-FT-1": "tfg_models/ProtoVAE/ProtoVAE-FT-1/seed=1/checkpoints/ProtoVAE_ProtoVAE-FT-1_seed1_best_val_adv_acc.pth",
    "ProtoVAE-FT-2": "tfg_models/ProtoVAE/ProtoVAE-FT-2/seed=1/checkpoints/ProtoVAE_ProtoVAE-FT-2_seed1_best_val_adv_acc.pth",
    "ProtoVAE-FT-3": "tfg_models/ProtoVAE/ProtoVAE-FT-3/seed=1/checkpoints/ProtoVAE_ProtoVAE-FT-3_seed1_best_val_adv_acc.pth",
    "ProtoVAE-FT-E": "tfg_models/ProtoVAE/ProtoVAE-FT-E/seed=1/checkpoints/ProtoVAE_ProtoVAE-FT-E_seed1_best_val_adv_acc.pth",
    # --- SENN ---
    "SENN_0_01": "SENN/results/mnist_lambda1e-2_seed29/checkpoints/best_model.pt",
    "SENN-FT-0": "tfg_models/SENN_0_01/SENN-FT-0/seed=1/checkpoints/SENN_0_01_SENN-FT-0_seed1_best_val_adv_acc.pth",
    "SENN-FT-1": "tfg_models/SENN_0_01/SENN-FT-1/seed=1/checkpoints/SENN_0_01_SENN-FT-1_seed1_best_val_adv_acc.pth",
    "SENN-FT-2": "tfg_models/SENN_0_01/SENN-FT-2/seed=1/checkpoints/SENN_0_01_SENN-FT-2_seed1_best_val_adv_acc.pth",
    # --- SENN config (needed to instantiate the architecture) ---
    "SENN_0_01_config": "SENN/configs/mnist_lambda1e-2_seed29.json",
}

# Used only by ch6_pca_training.py.
EPOCH_CKPT_DIR_TEMPLATE = "tfg_models/{arch}/{variant}/seed={seed}/checkpoints/"
PCA_FRAME_PREFIX = "pca_frame_epoch_"

# =============================================================================
# 3. Architecture variant lists
# =============================================================================

# Ordered list of variants per architecture (base model first).
VARIANTS: dict[str, list[str]] = {
    "B30":      ["B30",      "B30-FT-0",      "B30-FT-1",      "B30-FT-2",      "B30-FT-3",      "B30-FT-E", "B30-FT-E-M"],
    "ProtoVAE": ["ProtoVAE", "ProtoVAE-FT-0", "ProtoVAE-FT-1", "ProtoVAE-FT-2", "ProtoVAE-FT-3", "ProtoVAE-FT-E"],
    "SENN":     ["SENN_0_01","SENN-FT-0",     "SENN-FT-1",     "SENN-FT-2"],
}

# Short display labels for tables and plot tick labels.
VARIANT_LABELS: dict[str, str] = {
    "B30":           "Base",
    "B30-FT-0":      "FT-0",
    "B30-FT-1":      "FT-1",
    "B30-FT-2":      "FT-2",
    "B30-FT-3":      "FT-3",
    "B30-FT-E":      "FT-E",
    "B30-FT-E-M":    "FT-E-M",
    "ProtoVAE":      "Base",
    "ProtoVAE-FT-0": "FT-0",
    "ProtoVAE-FT-1": "FT-1",
    "ProtoVAE-FT-2": "FT-2",
    "ProtoVAE-FT-3": "FT-3",
    "ProtoVAE-FT-E": "FT-E",
    "SENN_0_01":     "Base",
    "SENN-FT-0":     "FT-0",
    "SENN-FT-1":     "FT-1",
    "SENN-FT-2":     "FT-2",
}

# What component each FT variant freezes (for table annotations).
FREEZE_DESCRIPTION: dict[str, str] = {
    "B30":           "—",
    "B30-FT-0":      "Nothing (full)",
    "B30-FT-1":      "Autoencoder",
    "B30-FT-2":      "Prototypes",
    "B30-FT-3":      "Classifier",
    "B30-FT-E":      "Protos + Classifier",
    "B30-FT-E-M":    "Protos + Classifier (Mixed)",
    "ProtoVAE":      "—",
    "ProtoVAE-FT-0": "Nothing (full)",
    "ProtoVAE-FT-1": "Autoencoder",
    "ProtoVAE-FT-2": "Prototypes",
    "ProtoVAE-FT-3": "Classifier",
    "ProtoVAE-FT-E": "Protos + Classifier",
    "SENN_0_01":     "—",
    "SENN-FT-0":     "Nothing (full)",
    "SENN-FT-1":     "Conceptizer",
    "SENN-FT-2":     "Parameterizer",
}

# =============================================================================
# 4. Metrics available per architecture
#    Keys match the arrays saved in per_example_metrics.npz
# =============================================================================
METRICS_BY_ARCH: dict[str, list[str]] = {
    "B30":      ["m_proto", "m_pred", "R_enc", "R_dec"],
    "ProtoVAE": ["m_proto", "m_pred", "R_enc", "R_dec", "R_mu", "R_sigma"],
    "SENN":     ["m_pred", "R_concept", "R_param"],
}

# Human-readable axis labels for each metric.
METRIC_LABELS: dict[str, str] = {
    "m_proto":   r"Prototype Margin $\tilde{m}_{proto}$",
    "m_pred":    r"Prediction Margin $\tilde{m}_{pred}$",
    "R_enc":     r"Encoder Drift $R_{enc}$",
    "R_dec":     r"Decoder Drift $R_{dec}$",
    "R_mu":      r"Mean Drift $R_{\mu}$",
    "R_sigma":   r"Uncertainty Drift $R_{\sigma}$",
    "R_concept": r"Concept Drift $R_{concept}$",
    "R_param":   r"Parameterizer Drift $R_{param}$",
}

# =============================================================================
# 5. Epsilon grid (must match --max_eps and --step used in run_metrics.py)
# =============================================================================
EPS_MAX  = 0.8
EPS_STEP = 0.025
EPS_GRID = np.round(np.arange(0.0, EPS_MAX + EPS_STEP / 2, EPS_STEP), decimals=6)

# Reference epsilon for scalar comparisons in tables (Chapter 5 / 7).
EPS_REF = 0.3

# =============================================================================
# 6. Plot aesthetics
# =============================================================================

# Color per variant — consistent across all chapters.
VARIANT_COLORS: dict[str, str] = {
    "B30":           "#555555",   # dark gray  — baseline
    "B30-FT-0":      "#1f77b4",   # blue
    "B30-FT-1":      "#d62728",   # red
    "B30-FT-2":      "#ff7f0e",   # orange
    "B30-FT-3":      "#2ca02c",   # green
    "B30-FT-E":      "#9467bd",   # purple
    "B30-FT-E-M":    "#8c564b",   # brown — distinguishable from purple FT-E
    "ProtoVAE":      "#555555",
    "ProtoVAE-FT-0": "#1f77b4",
    "ProtoVAE-FT-1": "#d62728",
    "ProtoVAE-FT-2": "#ff7f0e",
    "ProtoVAE-FT-3": "#2ca02c",
    "ProtoVAE-FT-E": "#9467bd",
    "SENN_0_01":     "#555555",
    "SENN-FT-0":     "#1f77b4",
    "SENN-FT-1":     "#d62728",
    "SENN-FT-2":     "#ff7f0e",
}

# Color per architecture — used in ch5_ch7_ablation.py --arch ALL.
ARCH_COLORS: dict[str, str] = {
    "B30":      "#1f77b4",
    "ProtoVAE": "#d62728",
    "SENN":     "#2ca02c",
}

# Matplotlib rcParams applied at the top of every analysis script.
RC_PARAMS: dict = {
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "figure.facecolor":  "white",
    "axes.spines.top":   False,
    "axes.spines.right": False,
}

# Output directory for saved figures, organised by chapter.
FIGURES_ROOT = "analysis/figures"
