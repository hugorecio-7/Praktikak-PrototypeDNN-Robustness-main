"""
loaders.py
==========
All path-resolution and data-loading logic lives here.
Analysis scripts never construct paths manually — they call these functions.

Path structure (mirrors model_testing_metrics.py):
    OUT_ROOT / DATASET / THREAT / attack / model_name / seed=S / run_id=R /
        metrics.json
        per_example_metrics.npz
        per_example_correct.npz
        curve.csv
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np

from config import (
    OUT_ROOT, DATASET_NAME, THREAT_NAME,
    DEFAULT_SEED, DEFAULT_RUN_ID,
    VARIANTS,
)


# =============================================================================
# Internal path resolver
# =============================================================================

def _run_dir(
    model_name: str,
    attack: str    = "PGDLInf_attack",
    seed: int      = DEFAULT_SEED,
    run_id: Optional[str] = DEFAULT_RUN_ID,
) -> Path:
    """
    Return the run directory for a given model / attack / seed / run_id.

    If run_id is None, the function lists all run_id subdirectories and picks
    the only one (raises if there are multiple — pass run_id explicitly then).
    """
    base = Path(OUT_ROOT) / DATASET_NAME / THREAT_NAME / attack / model_name / f"seed={seed}"

    if not base.exists():
        raise FileNotFoundError(
            f"[loaders] Base directory not found:\n  {base}\n"
            f"  Check OUT_ROOT, attack name, model name, and seed."
        )

    if run_id is not None:
        return base / f"run_id={run_id}"

    # Auto-detect: list run_id= subdirs
    candidates = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_id=")])
    if len(candidates) == 0:
        raise FileNotFoundError(f"[loaders] No run_id subdirectory found under:\n  {base}")
    if len(candidates) > 1:
        names = [d.name for d in candidates]
        raise ValueError(
            f"[loaders] Multiple run_ids found under {base}:\n  {names}\n"
            f"  Pass run_id explicitly to disambiguate."
        )
    return candidates[0]


# =============================================================================
# Public loaders
# =============================================================================

def load_json(
    model_name: str,
    attack: str   = "PGDLInf_attack",
    seed: int     = DEFAULT_SEED,
    run_id: Optional[str] = DEFAULT_RUN_ID,
) -> dict:
    """
    Load metrics.json for one model run.

    Returns the full dict as saved by model_testing_metrics.py.
    Key fields of interest:
        "acc"                        : list[float] — accuracy per epsilon
        "eps"                        : list[float] — epsilon grid
        "icr"                        : float | None
        "icr_senn"                   : float | None
        "tau_param"                  : float | None
        "empirical_robustness_interval": dict
        "mean_internal_metrics"      : dict[metric_name → list[float]]
        "architecture"               : "B30" | "ProtoVAE" | "SENN"
    """
    path = _run_dir(model_name, attack, seed, run_id) / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"[loaders] metrics.json not found:\n  {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_npz(
    model_name: str,
    attack: str   = "PGDLInf_attack",
    seed: int     = DEFAULT_SEED,
    run_id: Optional[str] = DEFAULT_RUN_ID,
) -> dict[str, np.ndarray]:
    """
    Load per_example_metrics.npz for one model run.

    Returns a dict of (N, E) float32 arrays, one per metric, plus:
        "eps"     : (E,) float32 — epsilon grid
    Available keys depend on architecture (see config.METRICS_BY_ARCH).
    """
    path = _run_dir(model_name, attack, seed, run_id) / "per_example_metrics.npz"
    if not path.exists():
        raise FileNotFoundError(f"[loaders] per_example_metrics.npz not found:\n  {path}")
    data = np.load(path)
    return dict(data)


def load_correct(
    model_name: str,
    attack: str   = "PGDLInf_attack",
    seed: int     = DEFAULT_SEED,
    run_id: Optional[str] = DEFAULT_RUN_ID,
) -> dict[str, np.ndarray]:
    """
    Load per_example_correct.npz for one model run.

    Returns:
        "correct" : (N, E) uint8 — 1 if model is correct, 0 otherwise
        "eps"     : (E,) float32 — epsilon grid
    """
    path = _run_dir(model_name, attack, seed, run_id) / "per_example_correct.npz"
    if not path.exists():
        raise FileNotFoundError(f"[loaders] per_example_correct.npz not found:\n  {path}")
    data = np.load(path)
    return dict(data)


def load_rpgd_binary_search(
    model_name: str,
    attack: str   = "PGDLInf_attack",
    seed: int     = DEFAULT_SEED,
    run_id: Optional[str] = DEFAULT_RUN_ID,
) -> dict[str, np.ndarray]:
    """
    Load r_pgd_binary_search.npz for one model run.

    Returns per-sample refined r-PGD values when the binary-search updater has
    been run. Samples misclassified at eps=0 are stored as NaN in r_pgd.
    """
    path = _run_dir(model_name, attack, seed, run_id) / "r_pgd_binary_search.npz"
    if not path.exists():
        raise FileNotFoundError(f"[loaders] r_pgd_binary_search.npz not found:\n  {path}")
    data = np.load(path)
    return dict(data)


def load_all_variants(
    arch: str,
    attack: str   = "PGDLInf_attack",
    seed: int     = DEFAULT_SEED,
    run_id: Optional[str] = DEFAULT_RUN_ID,
    verbose: bool = True,
) -> dict[str, dict]:
    """
    Load metrics.json + per_example_metrics.npz for every variant of an
    architecture in one call.

    Parameters
    ----------
    arch : "B30" | "ProtoVAE" | "SENN"

    Returns
    -------
    dict[model_name → {"json": dict, "npz": dict, "correct": dict}]

    Missing variants are skipped with a warning (useful when some runs are
    still in progress).
    """
    if arch not in VARIANTS:
        raise ValueError(f"[loaders] Unknown arch '{arch}'. Choose from {list(VARIANTS.keys())}.")

    result = {}
    for name in VARIANTS[arch]:
        try:
            j = load_json(name, attack, seed, run_id)
            n = load_npz(name, attack, seed, run_id)
            c = load_correct(name, attack, seed, run_id)
            result[name] = {"json": j, "npz": n, "correct": c}
            if verbose:
                print(f"  [OK] {name}")
        except FileNotFoundError as e:
            print(f"  [SKIP] {name} — {e}")

    if not result:
        raise RuntimeError(
            f"[loaders] No data found for arch='{arch}'. "
            "Check OUT_ROOT and that run_metrics.py has been executed."
        )
    return result


# =============================================================================
# Convenience extractors
# =============================================================================

def get_acc_curve(data_json: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (eps, acc) as float32 arrays from a loaded metrics.json."""
    eps = np.array(data_json["eps"],  dtype=np.float32)
    acc = np.array(data_json["acc"],  dtype=np.float32)
    return eps, acc


def get_mean_metric_curve(data_json: dict, metric: str) -> np.ndarray:
    """
    Return the mean metric curve (E,) from metrics.json["mean_internal_metrics"].
    These are the per-epsilon averages over the test set — useful for quick plots
    without loading the full (N, E) npz.
    """
    curves = data_json.get("mean_internal_metrics", {})
    if metric not in curves:
        available = list(curves.keys())
        raise KeyError(
            f"[loaders] Metric '{metric}' not in mean_internal_metrics. "
            f"Available: {available}"
        )
    return np.array(curves[metric], dtype=np.float32)


def get_robustness_interval(data_json: dict) -> dict:
    """
    Return the empirical robustness interval dict from metrics.json.

    Keys: mean_r_minus_failing, mean_r_plus, mean_r_minus_all, frac_never_fail,
          hist_r_minus (counts + bin_edges). For PGDLInf_attack runs updated
          with binary search, mean_r_minus_all is the refined mean r-PGD and
          min_r_minus_all is the refined worst-case r-PGD.
    """
    return data_json["empirical_robustness_interval"]


def get_rpgd_mean(data_json: dict) -> float:
    """Return the current mean r-PGD used by the analysis tables."""
    rob = get_robustness_interval(data_json)
    return float(rob["mean_r_minus_all"])


def get_rpgd_min(data_json: dict) -> float:
    """
    Return the refined worst-case r-PGD when available.

    Older metrics.json files and non-PGD attacks may not have the binary-search
    field, so callers get NaN instead of a hard failure.
    """
    rob = get_robustness_interval(data_json)
    if "min_r_minus_all" in rob:
        return float(rob["min_r_minus_all"])

    rpgd = data_json.get("r_pgd_binary_search") or {}
    if "min_r_pgd" in rpgd:
        return float(rpgd["min_r_pgd"])

    return float("nan")


def get_scalar_at_eps(
    npz: dict[str, np.ndarray],
    metric: str,
    eps_grid: np.ndarray,
    eps_ref: float,
) -> np.ndarray:
    """
    Return the (N,) slice of an (N, E) metric matrix at the epsilon closest
    to eps_ref.  Used for delta-metric columns in ablation tables.
    """
    idx = int(np.argmin(np.abs(eps_grid - eps_ref)))
    mat = npz[metric]        # (N, E)
    return mat[:, idx]       # (N,)

def load_pca_frame_paths(
    arch: str,
    variant: str,
    seed: int = DEFAULT_SEED,
) -> list[tuple[int, Path]]:
    """
    Return [(epoch, path), ...] for all pca_frame_epoch_*.pth files found,
    sorted by epoch. The pre-fine-tuning base (epoch 0) is prepended using
    CKPT_PATHS[base_model] — it is always available.

    Raises FileNotFoundError if the checkpoint directory does not exist.
    """
    from config import CKPT_PATHS, PCA_FRAME_PREFIX, EPOCH_CKPT_DIR_TEMPLATE

    base_models = {"B30": "B30", "ProtoVAE": "ProtoVAE", "SENN": "SENN_0_01"}
    ckpt_dir = Path(EPOCH_CKPT_DIR_TEMPLATE.format(
        arch=arch, variant=variant, seed=seed
    ))
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"[loaders] Checkpoint dir not found: {ckpt_dir}")

    frames = []
    for p in sorted(ckpt_dir.glob(f"{PCA_FRAME_PREFIX}*.pth")):
        epoch = int(p.stem.replace(PCA_FRAME_PREFIX, ""))
        frames.append((epoch, p))

    # Prepend epoch 0 = base model before fine-tuning
    base_path = Path(CKPT_PATHS[base_models[arch]])
    return [(0, base_path)] + frames
