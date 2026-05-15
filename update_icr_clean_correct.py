"""
Update existing metrics.json files so ICR uses the clean-correct population.

The script does not rerun attacks. It reads:
  - per_example_metrics.npz  (m_pred plus m_proto or R_param)
  - per_example_correct.npz  (clean correctness mask in column 0)

It then writes the clean-correct ICR as the primary `icr` / `icr_senn`, while
preserving the legacy all-sample value as `icr_all` / `icr_senn_all`.

Example:
    python update_icr_clean_correct.py --models B30 B30-FT-0 --run_id ablation_B30_v1
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from metric_calculators import compute_icr, compute_icr_senn, compute_tau_param
from run_metrics import DATASET_NAME, OUT_ROOT, THREAT_NAME


def find_metrics_files(
    results_root: Path,
    attacks: set[str] | None,
    models: set[str] | None,
    run_id: str | None,
    seed: int | None,
) -> list[Path]:
    files = sorted(results_root.glob("*/*/seed=*/run_id=*/metrics.json"))

    selected: list[Path] = []
    for metrics_path in files:
        attack_name = metrics_path.parents[3].name
        model_name = metrics_path.parents[2].name
        seed_dir = metrics_path.parents[1].name
        run_dir = metrics_path.parent.name

        if attacks is not None and attack_name not in attacks:
            continue
        if models is not None and model_name not in models:
            continue
        if run_id is not None and run_dir != f"run_id={run_id}":
            continue
        if seed is not None and seed_dir != f"seed={seed}":
            continue

        selected.append(metrics_path)

    return selected


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def patch_metrics(metrics_path: Path, no_backup: bool) -> str:
    run_dir = metrics_path.parent
    metrics_npz_path = run_dir / "per_example_metrics.npz"
    correct_npz_path = run_dir / "per_example_correct.npz"

    if not metrics_npz_path.exists():
        return f"skip missing per_example_metrics.npz: {metrics_path}"
    if not correct_npz_path.exists():
        return f"skip missing per_example_correct.npz: {metrics_path}"

    per_metrics = np.load(metrics_npz_path)
    per_correct = np.load(correct_npz_path)

    if "m_pred" not in per_metrics:
        return f"skip missing m_pred: {metrics_path}"

    clean_correct_mask = per_correct["correct"][:, 0].astype(bool)
    payload = load_json(metrics_path)
    payload["icr_population"] = {
        "n_total": int(clean_correct_mask.size),
        "n_clean_correct": int(clean_correct_mask.sum()),
        "n_excluded_clean_incorrect": int((~clean_correct_mask).sum()),
        "frac_clean_correct": float(clean_correct_mask.mean()),
    }

    changed = False
    if "m_proto" in per_metrics:
        icr_all = compute_icr(per_metrics["m_proto"], per_metrics["m_pred"])
        icr_clean = compute_icr(
            per_metrics["m_proto"],
            per_metrics["m_pred"],
            clean_correct_mask=clean_correct_mask,
        )
        payload["icr_all"] = icr_all
        payload["icr"] = icr_clean
        payload["icr_definition"] = (
            "(1/N_clean_correct) * sum_i max_{eps} "
            "I[m_proto(x_i,eps)<0 & m_pred(x_i,eps)>0], restricted to samples "
            "correctly classified at eps=0. Legacy all-sample value is stored "
            "as icr_all."
        )
        changed = True

    if "R_param" in per_metrics:
        tau_all = compute_tau_param(per_metrics["R_param"], eps_idx=1)
        icr_senn_all = compute_icr_senn(
            per_metrics["R_param"],
            per_metrics["m_pred"],
            tau_all,
        )
        tau_clean = compute_tau_param(
            per_metrics["R_param"],
            eps_idx=1,
            clean_correct_mask=clean_correct_mask,
        )
        icr_senn_clean = compute_icr_senn(
            per_metrics["R_param"],
            per_metrics["m_pred"],
            tau_clean,
            clean_correct_mask=clean_correct_mask,
        )
        payload["tau_param_all"] = tau_all
        payload["icr_senn_all"] = icr_senn_all
        payload["tau_param"] = tau_clean
        payload["icr_senn"] = icr_senn_clean
        payload["icr_senn_definition"] = (
            "(1/N_clean_correct) * sum_i max_{eps} "
            "I[R_param(x_i,eps)>tau & m_pred(x_i,eps)>0], restricted to samples "
            "correctly classified at eps=0. tau is p95 of R_param at eps_min "
            "over the same clean-correct population. Legacy all-sample values "
            "are stored as tau_param_all and icr_senn_all."
        )
        changed = True

    if not changed:
        return f"skip no ICR metric found: {metrics_path}"

    if not no_backup:
        backup_path = metrics_path.with_suffix(metrics_path.suffix + ".bak_icr_all")
        if not backup_path.exists():
            shutil.copy2(metrics_path, backup_path)

    save_json(metrics_path, payload)
    model = payload.get("model", metrics_path.parents[2].name)
    attack = payload.get("attack", metrics_path.parents[3].name)
    return f"updated {attack}/{model}: {metrics_path}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute ICR over clean-correct samples for existing runs."
    )
    parser.add_argument(
        "--results_root",
        type=Path,
        default=Path(OUT_ROOT) / DATASET_NAME / THREAT_NAME,
    )
    parser.add_argument("--attacks", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_backup", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    metrics_files = find_metrics_files(
        results_root=args.results_root,
        attacks=set(args.attacks) if args.attacks else None,
        models=set(args.models) if args.models else None,
        run_id=args.run_id,
        seed=args.seed,
    )

    if not metrics_files:
        print("No matching metrics.json files found.")
        return

    print(f"Found {len(metrics_files)} result folder(s).")
    if args.dry_run:
        for path in metrics_files:
            print(path)
        return

    for metrics_path in metrics_files:
        print(patch_metrics(metrics_path, no_backup=args.no_backup))


if __name__ == "__main__":
    main()
