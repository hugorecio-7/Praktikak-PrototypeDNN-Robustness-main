"""
Update existing PGDLInf_attack result folders with binary-search r-PGD.

This avoids rerunning the full metric sweep. The script reuses each run's
per_example_correct.npz as the coarse epsilon-grid bracket, then evaluates only
the PGD binary-search points needed to refine r_PGD.

Example:
    python update_rpgd_binary_search.py --models B30 B30-FT-0 --run_id ablation_B30_v1
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from data_loader import get_test_loader
from metric_calculators import compute_r_pgd_binary_search
from run_metrics import DATASET_NAME, OUT_ROOT, THREAT_NAME, attack_params, load_models


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def find_metrics_files(
    results_root: Path,
    models: set[str] | None,
    run_id: str | None,
    seed: int | None,
) -> list[Path]:
    files = sorted(results_root.glob("*/seed=*/run_id=*/metrics.json"))
    selected: list[Path] = []

    for metrics_path in files:
        model_name = metrics_path.parents[2].name
        seed_dir = metrics_path.parents[1].name
        run_dir = metrics_path.parent.name

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


def update_metrics_payload(metrics: dict, result: dict) -> dict:
    rob = metrics.setdefault("empirical_robustness_interval", {})

    if "grid_empirical_robustness_interval" not in metrics:
        metrics["grid_empirical_robustness_interval"] = dict(rob)

    rob.update({
        "grid_mean_r_minus_failing": metrics["grid_empirical_robustness_interval"].get("mean_r_minus_failing"),
        "grid_mean_r_minus_all": metrics["grid_empirical_robustness_interval"].get("mean_r_minus_all"),
        "grid_hist_r_minus": metrics["grid_empirical_robustness_interval"].get("hist_r_minus"),
        "mean_r_minus_failing": result["mean_r_pgd_failing"],
        "mean_r_minus_all": result["mean_r_pgd"],
        "min_r_minus_all": result["min_r_pgd"],
        "frac_never_fail": result["frac_never_fail"],
        "n_clean_correct": result["n_clean_correct"],
        "n_excluded_clean_incorrect": result["n_excluded_clean_incorrect"],
        "frac_clean_correct": result["frac_clean_correct"],
        "hist_r_minus": result["hist_r_pgd"],
        "binary_search_refined": True,
        "definition": (
            "r_minus is refined by binary search inside the coarse epsilon-grid "
            "interval [last correct eps, first failing eps]. r_plus remains the "
            "first failing epsilon from the coarse grid. Samples misclassified at "
            "eps=0 are excluded because their attack robustness radius is undefined."
        ),
    })

    metrics["r_pgd_binary_search"] = {
        "mean_r_pgd": result["mean_r_pgd"],
        "mean_r_pgd_failing": result["mean_r_pgd_failing"],
        "min_r_pgd": result["min_r_pgd"],
        "frac_never_fail": result["frac_never_fail"],
        "n_total": result["n_total"],
        "n_clean_correct": result["n_clean_correct"],
        "n_excluded_clean_incorrect": result["n_excluded_clean_incorrect"],
        "frac_clean_correct": result["frac_clean_correct"],
        "binary_steps": result["binary_steps"],
        "tol": result["tol"],
        "pgd_iters": result["pgd_iters"],
        "pgd_alpha": result["pgd_alpha"],
        "pgd_random_start": result["pgd_random_start"],
        "npz_file": "r_pgd_binary_search.npz",
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add binary-search r-PGD to existing PGDLInf_attack metrics."
    )
    parser.add_argument("--results_root", type=Path,
                        default=Path(OUT_ROOT) / DATASET_NAME / THREAT_NAME / "PGDLInf_attack")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Optional model names to update. Defaults to all models found.")
    parser.add_argument("--run_id", type=str, default=None,
                        help="Optional run id without the run_id= prefix.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=250)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--binary_steps", type=int, default=10)
    parser.add_argument("--binary_tol", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even if r_pgd_binary_search.npz already exists.")
    parser.add_argument("--no_backup", action="store_true",
                        help="Do not create metrics.json.bak before writing.")
    parser.add_argument("--dry_run", action="store_true",
                        help="List the runs that would be updated without computing PGD.")
    args = parser.parse_args()

    set_seed(args.seed)

    selected_models = set(args.models) if args.models else None
    metrics_files = find_metrics_files(
        results_root=args.results_root,
        models=selected_models,
        run_id=args.run_id,
        seed=args.seed,
    )

    if not metrics_files:
        print("No matching metrics.json files found.")
        return

    print(f"Found {len(metrics_files)} PGD result folder(s).")
    if args.dry_run:
        for path in metrics_files:
            print(path)
        return

    test_loader = get_test_loader(
        "data",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    pgd_params = attack_params.get("PGDLInf_attack", {})
    loaded_models: dict[str, torch.nn.Module] = {}

    for metrics_path in metrics_files:
        run_dir = metrics_path.parent
        metrics = load_json(metrics_path)
        model_name = metrics.get("model") or metrics_path.parents[2].name

        npz_path = run_dir / "r_pgd_binary_search.npz"
        if npz_path.exists() and not args.overwrite:
            print(f"Skipping {model_name} at {run_dir}: r_pgd_binary_search.npz already exists.")
            continue

        correct_path = run_dir / "per_example_correct.npz"
        if not correct_path.exists():
            raise FileNotFoundError(f"Missing coarse grid file: {correct_path}")

        coarse = np.load(correct_path)
        coarse_correct = coarse["correct"].astype(bool)
        eps_grid = coarse["eps"].astype(np.float64)

        if model_name not in loaded_models:
            models, _ = load_models([model_name])
            loaded_models[model_name] = models[0]

        print(f"Updating {model_name} at {run_dir}")
        result = compute_r_pgd_binary_search(
            model=loaded_models[model_name],
            data_loader=test_loader,
            eps_grid=eps_grid,
            coarse_correct=coarse_correct,
            pgd_iters=pgd_params.get("iters", 80),
            pgd_alpha=pgd_params.get("alpha", 0.01),
            pgd_random_start=pgd_params.get("random_start", True),
            binary_steps=args.binary_steps,
            tol=args.binary_tol,
            seed=args.seed,
            verbose=True,
        )

        np.savez_compressed(
            npz_path,
            r_pgd=result["r_pgd"],
            r_pgd_upper=result["r_pgd_upper"],
            failing_mask=result["failing_mask"],
            clean_correct_mask=result["clean_correct_mask"],
            eps_grid=result["eps_grid"],
        )

        updated = update_metrics_payload(metrics, result)
        if not args.no_backup:
            backup_path = metrics_path.with_suffix(metrics_path.suffix + ".bak")
            if not backup_path.exists():
                shutil.copy2(metrics_path, backup_path)
        save_json(metrics_path, updated)

        print(
            f"  mean r-PGD={result['mean_r_pgd']:.6f} | "
            f"min r-PGD={result['min_r_pgd']:.6f}"
        )


if __name__ == "__main__":
    main()
