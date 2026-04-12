"""
ch6_cohesion_patch.py
=====================
Chapter 6.1 — Cohesion ratio patch.

What this script does
---------------------
Computes calc_cohesion_ratio() (already implemented in metric_calculators.py)
for one or all variants of an architecture, then patches the existing
metrics.json by adding a "cohesion_ratio" field.

No re-running of the attack pipeline is needed. The script loads the model
checkpoint, runs one clean forward pass over the test set, computes the ratio,
and writes the result back to the metrics.json that run_metrics.py already
produced.

Structure of the added field
-----------------------------
"cohesion_ratio": {
    "mean":        float,   # E[D_intra / D_inter] over test set
    "std":         float,
    "median":      float,
    "pct_below_1": float,   # fraction of samples with ratio < 1 (correct cohesion)
    "per_sample":  [...]    # list of N floats (omitted if --no_per_sample)
}

Why ratio < 1 means correct cohesion
--------------------------------------
D_intra = distance to nearest CORRECT prototype.
D_inter = distance to nearest WRONG prototype.
ratio   = D_intra / D_inter.
ratio < 1  →  sample is closer to a correct prototype than to any wrong one.
ratio > 1  →  cohesion has broken (sample drifted toward wrong prototypes).

Note on SENN
------------
SENN has no prototype layer. Running this script on SENN variants will exit
with a clear error message.

Usage
-----
    # Patch one variant:
    python ch6_cohesion_patch.py --arch B30 --variant B30-FT-0

    # Patch all variants of an architecture:
    python ch6_cohesion_patch.py --arch B30 --all_variants

    # Skip saving per-sample array (saves space in metrics.json):
    python ch6_cohesion_patch.py --arch B30 --all_variants --no_per_sample
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from config import (
    CKPT_PATHS, VARIANTS, DEFAULT_SEED, DEFAULT_RUN_ID,
    OUT_ROOT, DATASET_NAME, THREAT_NAME,
)
from loaders import load_json, _run_dir
from metric_extractor import extract_internals, get_proto_labels
from metric_calculators import calc_cohesion_ratio


# =============================================================================
# Model loading  (mirrors run_metrics.py load_models logic exactly)
# =============================================================================

def _load_b30_base(device: torch.device) -> nn.Module:
    path = CKPT_PATHS["B30"]
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(path, map_location=device)
    return model.to(device).eval()


def _load_protovae_base(device: torch.device) -> nn.Module:
    from ProtoVAE import model as model_protovae
    model = model_protovae.ProtoVAE().to(device)
    state = torch.load(CKPT_PATHS["ProtoVAE"], map_location=device)
    model.load_state_dict(state)
    return model.eval()


def _load_senn_base(device: torch.device) -> nn.Module:
    raise ValueError(
        "[cohesion_patch] SENN has no prototype layer. "
        "calc_cohesion_ratio cannot be applied to SENN variants."
    )


def load_model_for_arch(
    arch: str,
    model_name: str,
    device: torch.device,
) -> nn.Module:
    """
    Load and return the model in eval mode, unwrapped (no ProtoVAEWrapper /
    SENNWrapper) so that extract_internals and get_proto_labels work directly.

    For FT variants the architecture is instantiated from the base checkpoint
    and the FT state_dict is applied on top — identical to run_metrics.py.
    """
    if arch == "SENN":
        _load_senn_base(device)   # always raises

    # ---- B30 ----------------------------------------------------------------
    if arch == "B30":
        model = _load_b30_base(device)
        if model_name != "B30":                         # FT variant
            state = torch.load(CKPT_PATHS[model_name], map_location=device)
            model.load_state_dict(state["model_state"])
        return model

    # ---- ProtoVAE -----------------------------------------------------------
    if arch == "ProtoVAE":
        model = _load_protovae_base(device)
        if model_name != "ProtoVAE":                    # FT variant
            state = torch.load(CKPT_PATHS[model_name], map_location=device)
            model.load_state_dict(state["model_state"])
        return model

    raise ValueError(f"[cohesion_patch] Unsupported arch '{arch}'.")


# =============================================================================
# Prototype tensor extractor
#   Returns (n_proto, D) tensor — architecture-aware.
# =============================================================================

def get_prototype_tensor(arch: str, model: nn.Module) -> torch.Tensor:
    """
    Return the learnable prototype matrix as a (n_proto, D) CPU tensor.

    B30      : model.prototype_layer.prototype_distances  (n_proto, flat_dim)
    ProtoVAE : model.prototype_vectors                    (n_proto, latent_dim)
    """
    if arch == "B30":
        return model.prototype_layer.prototype_distances.detach().cpu()
    if arch == "ProtoVAE":
        return model.prototype_vectors.detach().cpu()
    raise ValueError(f"[cohesion_patch] No prototype tensor for arch '{arch}'.")


# =============================================================================
# Core computation
# =============================================================================

def compute_cohesion_for_model(
    arch: str,
    model: nn.Module,
    test_loader,
    device: torch.device,
    save_per_sample: bool = True,
) -> dict:
    """
    Run a clean forward pass over the full test set and compute cohesion ratios.

    Returns
    -------
    dict with keys: mean, std, median, pct_below_1, [per_sample]
    """
    proto_labels = get_proto_labels(model).to(device)     # (n_proto,)
    prototypes   = get_prototype_tensor(arch, model).to(device)  # (n_proto, D)

    all_ratios: list[np.ndarray] = []

    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            internals = extract_internals(model, x)
            z = internals["z"].to(device)               # (B, D)

            ratios = calc_cohesion_ratio(
                z,
                y,
                proto_labels.cpu(),
                prototypes.cpu(),
            )                                           # (B,) float32
            all_ratios.append(ratios)

    ratios_all = np.concatenate(all_ratios, axis=0)     # (N,)

    result = {
        "mean":        float(np.mean(ratios_all)),
        "std":         float(np.std(ratios_all)),
        "median":      float(np.median(ratios_all)),
        "pct_below_1": float(np.mean(ratios_all < 1.0)),
    }
    if save_per_sample:
        result["per_sample"] = ratios_all.tolist()

    return result


# =============================================================================
# JSON patcher
# =============================================================================

def patch_metrics_json(
    model_name: str,
    attack: str,
    seed: int,
    run_id: Optional[str],
    cohesion_result: dict,
) -> None:
    """
    Add "cohesion_ratio" field to the existing metrics.json and overwrite it.
    All previous fields are preserved.
    """
    run_path  = _run_dir(model_name, attack, seed, run_id)
    json_path = run_path / "metrics.json"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["cohesion_ratio"] = cohesion_result

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"  Patched → {json_path}")


# =============================================================================
# Data loader helper  (only test set needed, no attack)
# =============================================================================

def get_test_loader(batch_size: int = 250, num_workers: int = 0):
    from data_loader import get_test_loader as _gtl
    return _gtl(
        "data", batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=False,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute cohesion ratio and patch metrics.json."
    )
    parser.add_argument(
        "--arch", type=str, required=True,
        choices=["B30", "ProtoVAE"],
        help="Architecture (SENN not supported — no prototypes).",
    )
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Single variant to patch (e.g. B30-FT-0). "
             "Ignored if --all_variants is set.",
    )
    parser.add_argument(
        "--all_variants", action="store_true",
        help="Patch all variants of the architecture.",
    )
    parser.add_argument("--attack",         type=str, default="PGDLInf_attack")
    parser.add_argument("--seed",           type=int, default=DEFAULT_SEED)
    parser.add_argument("--run_id",         type=str, default=DEFAULT_RUN_ID)
    parser.add_argument("--batch_size",     type=int, default=250)
    parser.add_argument("--num_workers",    type=int, default=0)
    parser.add_argument(
        "--no_per_sample", action="store_true",
        help="Do not save the per-sample array in metrics.json (saves space).",
    )
    args = parser.parse_args()

    # Determine which variants to process
    if args.all_variants:
        model_names = VARIANTS[args.arch]
    elif args.variant is not None:
        model_names = [args.variant]
    else:
        parser.error("Provide --variant <name> or --all_variants.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    test_loader = get_test_loader(args.batch_size, args.num_workers)
    save_per_sample = not args.no_per_sample

    for model_name in model_names:
        print(f"\n[{model_name}]")
        print(f"  Loading model …")
        try:
            model = load_model_for_arch(args.arch, model_name, device)
        except (FileNotFoundError, KeyError) as e:
            print(f"  [SKIP] Could not load model: {e}")
            continue

        print(f"  Computing cohesion ratio over test set …")
        cohesion = compute_cohesion_for_model(
            args.arch, model, test_loader, device, save_per_sample,
        )

        print(
            f"  mean={cohesion['mean']:.4f}  "
            f"std={cohesion['std']:.4f}  "
            f"median={cohesion['median']:.4f}  "
            f"pct_below_1={cohesion['pct_below_1']:.4f}"
        )

        try:
            patch_metrics_json(model_name, args.attack, args.seed, args.run_id, cohesion)
        except FileNotFoundError as e:
            print(f"  [SKIP] metrics.json not found: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
