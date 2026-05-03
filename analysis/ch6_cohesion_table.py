"""
ch6_cohesion_table.py
=====================
Chapter 6.1 — Invariant Cohesion Ratio table.

What this script produces
--------------------------
Table  (ch6_cohesion_table.csv  +  ch6_cohesion_table.tex)
    One row per variant, columns:
        Variant | Frozen | mean ratio | std | median | pct_below_1 |
        E[r_PGD-] | min(r_PGD-)

    pct_below_1 is the narrative key column: the fraction of test samples
    that are geometrically closer to a correct prototype than to any wrong one.
    A high pct_below_1 across FT variants proves that fine-tuning preserves
    the semantic cluster structure even when R_enc is large (Cap 6.1 claim).

Prerequisite
------------
Run ch6_cohesion_patch.py --arch B30 --all_variants first.
The "cohesion_ratio" field must exist in each metrics.json.

Usage
-----
    python ch6_cohesion_table.py
    python ch6_cohesion_table.py --arch ProtoVAE
    python ch6_cohesion_table.py --arch B30 --attack PGDLInf_attack
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from config import (
    FIGURES_ROOT, VARIANTS, VARIANT_LABELS,
    FREEZE_DESCRIPTION, DEFAULT_SEED, DEFAULT_RUN_ID,
)
from loaders import load_json, get_rpgd_mean, get_rpgd_min


# =============================================================================
# Table builder
# =============================================================================

def build_cohesion_table(
    arch:    str,
    attack:  str,
    seed:    int,
    run_id:  str | None,
) -> pd.DataFrame:
    rows = []

    for model_name in VARIANTS[arch]:
        try:
            j = load_json(model_name, attack, seed, run_id)
        except FileNotFoundError:
            print(f"  [SKIP] {model_name} — metrics.json not found.")
            continue

        cohesion = j.get("cohesion_ratio")
        if cohesion is None:
            print(
                f"  [SKIP] {model_name} — 'cohesion_ratio' field missing. "
                f"Run ch6_cohesion_patch.py first."
            )
            continue

        rows.append({
            "Variant":              VARIANT_LABELS.get(model_name, model_name),
            "Frozen":               FREEZE_DESCRIPTION.get(model_name, "—"),
            "Mean ratio":           round(cohesion["mean"],        4),
            "Std":                  round(cohesion["std"],         4),
            "Median":               round(cohesion["median"],      4),
            r"$\%_{<1}$":           round(cohesion["pct_below_1"], 4),
            r"$E[r_{PGD}^-]$":      round(get_rpgd_mean(j), 4),
            #r"$\min(r_{PGD}^-)$":   round(get_rpgd_min(j), 6),
        })

    if not rows:
        raise RuntimeError(
            "[cohesion_table] No data loaded. "
            "Check that ch6_cohesion_patch.py has been run for this arch."
        )

    return pd.DataFrame(rows)


# =============================================================================
# Save helpers
# =============================================================================

def _save_table(df: pd.DataFrame, out_dir: str, stem: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"{stem}.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Saved → {csv_path}")

    tex_path = os.path.join(out_dir, f"{stem}.tex")
    latex_str = df.to_latex(
        index=False,
        escape=False,
        float_format="%.4f",
        na_rep="—",
        caption="Invariant cohesion ratio across fine-tuning variants.",
        label=f"tab:{stem}",
    )
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"  Saved → {tex_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chapter 6.1 — Cohesion ratio table."
    )
    parser.add_argument(
        "--arch", type=str, default="B30",
        choices=["B30", "ProtoVAE"],
        help="Architecture (SENN not supported).",
    )
    parser.add_argument("--attack", type=str, default="PGDLInf_attack")
    parser.add_argument("--seed",   type=int, default=DEFAULT_SEED)
    parser.add_argument("--run_id", type=str, default=DEFAULT_RUN_ID)
    args = parser.parse_args()

    out_dir = os.path.join(FIGURES_ROOT, "ch6")

    print(f"\nBuilding cohesion table for {args.arch} …")
    df = build_cohesion_table(args.arch, args.attack, args.seed, args.run_id)

    print("\n" + df.to_string(index=False))
    _save_table(df, out_dir, f"ch6_cohesion_table_{args.arch}")

    print("\nDone.")


if __name__ == "__main__":
    main()
