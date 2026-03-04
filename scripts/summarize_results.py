import argparse
import glob
import json
import os
from typing import List, Dict, Any, Tuple

import pandas as pd


def parse_eps_list(s: str) -> List[float]:
    if not s:
        return []
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def nearest_index(values: List[float], target: float) -> int:
    # Find the index of the value in values that is closest to target
    best_i, best_d = 0, float("inf")
    for i, v in enumerate(values):
        d = abs(v - target)
        if d < best_d:
            best_i, best_d = i, d
    return best_i


def load_metrics_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def metrics_to_long_rows(m: Dict[str, Any]) -> List[Dict[str, Any]]:
    eps = m["eps"]
    acc = m["acc"]
    ci = m.get("ci95_bootstrap", {})
    low = ci.get("low", [None] * len(eps))
    high = ci.get("high", [None] * len(eps))
    rel = m.get("relative_degradation_pct", [None] * len(eps))

    base = {
        "dataset": m.get("dataset"),
        "threat_model": m.get("threat_model"),
        "model": m.get("model"),
        "attack": m.get("attack"),
        "seed": m.get("seed"),
        "run_id": m.get("run_id"),
        "runtime_s": m.get("runtime_s"),
        "gpu_name": m.get("gpu_name"),
        "n_test": m.get("n_test"),
    }

    rows = []
    for i, e in enumerate(eps):
        rows.append({
            **base,
            "eps": float(e),
            "acc": float(acc[i]),
            "ci_low": None if low[i] is None else float(low[i]),
            "ci_high": None if high[i] is None else float(high[i]),
            "rel_deg_pct": None if rel[i] is None else float(rel[i]),
        })
    return rows


def build_summary(df_long: pd.DataFrame, pick_eps: List[float]) -> pd.DataFrame:
    # Always include the clean (eps=0) and max_eps (last row) results, y opcionalmente incluir los eps que se pidan en pick_eps
    # Optionally include the requested eps in pick_eps, but always include clean (eps=0) and max_eps (last row)
    grouped = []
    for (model, attack, seed, run_id), g in df_long.groupby(["model", "attack", "seed", "run_id"], dropna=False):
        g = g.sort_values("eps")
        eps_list = g["eps"].tolist()

        idx_clean = nearest_index(eps_list, 0.0)
        idx_last = len(eps_list) - 1

        def row_for_idx(idx: int, label: str) -> Dict[str, Any]:
            r = g.iloc[idx]
            return {
                f"{label}_eps": r["eps"],
                f"{label}_acc": r["acc"],
                f"{label}_ci_low": r["ci_low"],
                f"{label}_ci_high": r["ci_high"],
                f"{label}_rel_deg_pct": r["rel_deg_pct"],
            }

        base = {
            "dataset": g.iloc[0]["dataset"],
            "threat_model": g.iloc[0]["threat_model"],
            "model": model,
            "attack": attack,
            "seed": seed,
            "run_id": run_id,
            "runtime_s": g.iloc[0]["runtime_s"],
            "gpu_name": g.iloc[0]["gpu_name"],
            "n_test": g.iloc[0]["n_test"],
        }

        out = {**base, **row_for_idx(idx_clean, "clean"), **row_for_idx(idx_last, "maxeps")}

        # eps extra solicitados
        for e in pick_eps:
            idx = nearest_index(eps_list, e)
            out.update(row_for_idx(idx, f"eps{str(e).replace('.','_')}"))

        grouped.append(out)

    return pd.DataFrame(grouped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="resultsTFG/runs_v1", help="Root directory to search for metrics.json files")
    ap.add_argument("--out_dir", type=str, default="resultsTFG/summary", help="Where to save the summary CSV files")
    ap.add_argument("--pick_eps", type=str, default="", help="Comma-separated list of epsilon values to include in the summary (in addition to clean and max_eps)")
    args = ap.parse_args()

    paths = glob.glob(os.path.join(args.root, "**", "metrics.json"), recursive=True)
    if not paths:
        raise SystemExit(f"Do not found any metrics.json under {args.root}")

    all_rows = []
    for p in paths:
        m = load_metrics_json(p)
        all_rows.extend(metrics_to_long_rows(m))

    df_long = pd.DataFrame(all_rows)
    os.makedirs(args.out_dir, exist_ok=True)

    long_path = os.path.join(args.out_dir, "results_long.csv")
    df_long.to_csv(long_path, index=False)

    pick_eps = parse_eps_list(args.pick_eps)
    df_sum = build_summary(df_long, pick_eps)
    sum_path = os.path.join(args.out_dir, "results_summary.csv")
    df_sum.to_csv(sum_path, index=False)

    df_pretty = df_sum.sort_values(["attack", "model", "seed"])
    pretty_path = os.path.join(args.out_dir, "results_summary_sorted.csv")
    df_pretty.to_csv(pretty_path, index=False)

    print("OK")
    print(f"Long:   {long_path}")
    print(f"Summary:{sum_path}")
    print(f"Sorted: {pretty_path}")


if __name__ == "__main__":
    main()