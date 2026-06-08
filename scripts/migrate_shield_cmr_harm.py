"""
Migrate old Shield metrics to the CMR / conditional Harm Rate convention.

This script updates saved runs in resultsTFG/runs_v1 by:
  - renaming the old "suppressed_matches" curve to "cmr";
  - recomputing harm_rate as
        risk_harm_match_count / total_count;
  - adding "corrected_match" and "latent_correct_match" arrays to
    per_example_metrics.npz when they are missing.

By default the script is a dry run. Pass --apply to write changes.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_ROOT = ROOT / "resultsTFG" / "runs_v1"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def backup_once(path: Path) -> None:
    backup = path.with_name(path.name + ".bak")
    if not backup.exists():
        shutil.copy2(path, backup)


def bool_array(data: np.ndarray) -> np.ndarray:
    return np.asarray(data).astype(bool)


def percent_curve(num: np.ndarray, den: np.ndarray) -> list[float]:
    curve = []
    for numerator, denominator in zip(num.tolist(), den.tolist()):
        curve.append(float((numerator / denominator) * 100.0) if denominator > 0 else 0.0)
    return curve


def recompute_curves(npz: dict[str, np.ndarray]) -> dict[str, Any]:
    required = {"false_match_mu", "risk_harm_match"}
    missing = required.difference(npz)
    if missing:
        raise KeyError(f"Missing required arrays: {', '.join(sorted(missing))}")

    false_match = bool_array(npz["false_match_mu"])
    risk_harm = bool_array(npz["risk_harm_match"])

    if "corrected_match" in npz:
        corrected = bool_array(npz["corrected_match"])
    elif "suppressed_match" in npz:
        corrected = bool_array(npz["suppressed_match"])
    else:
        raise KeyError("Missing corrected_match/suppressed_match array.")

    if "latent_correct_match" in npz:
        latent_correct = bool_array(npz["latent_correct_match"])
    elif "m_proto_latent" in npz:
        latent_correct = np.asarray(npz["m_proto_latent"]) > 0.0
    else:
        raise KeyError(
            "Missing latent_correct_match and m_proto_latent. Cannot recompute "
            "the latent-correct counts exactly."
        )

    if false_match.shape != corrected.shape or false_match.shape != risk_harm.shape:
        raise ValueError(
            "Metric arrays must have matching shape: "
            f"false={false_match.shape}, corrected={corrected.shape}, harm={risk_harm.shape}"
        )
    if latent_correct.shape != false_match.shape:
        raise ValueError(
            "latent_correct shape does not match metric arrays: "
            f"{latent_correct.shape} != {false_match.shape}"
        )

    false_count = false_match.sum(axis=0).astype(np.int64)
    corrected_count = corrected.sum(axis=0).astype(np.int64)
    latent_correct_count = latent_correct.sum(axis=0).astype(np.int64)
    risk_harm_count = risk_harm.sum(axis=0).astype(np.int64)
    total_count = np.full(false_match.shape[1], false_match.shape[0], dtype=np.int64)

    return {
        "cmr": percent_curve(corrected_count, false_count),
        "harm_rate": percent_curve(risk_harm_count, total_count),
        "total_count": total_count.tolist(),
        "latent_wrong_count": false_count.tolist(),
        "false_match_mu_count": false_count.tolist(),
        "corrected_count": corrected_count.tolist(),
        "corrected_match_count": corrected_count.tolist(),
        "latent_correct_count": latent_correct_count.tolist(),
        "damaged_count": risk_harm_count.tolist(),
        "risk_harm_match_count": risk_harm_count.tolist(),
        "corrected_match": corrected.astype(np.float32),
        "latent_correct_match": latent_correct.astype(np.float32),
        "definition": (
            "cmr is 100 * corrected_match / false_match_mu per epsilon; "
            "harm_rate is 100 * risk_harm_match / total_count per epsilon."
        ),
    }


def update_metrics_json(
    path: Path,
    curves: dict[str, Any],
    apply: bool,
    backup: bool,
) -> bool:
    payload = load_json(path)
    changed = False

    if "suppressed_matches" in payload:
        payload.pop("suppressed_matches")
        changed = True

    if payload.get("cmr") != curves["cmr"]:
        payload["cmr"] = curves["cmr"]
        changed = True
    if payload.get("harm_rate") != curves["harm_rate"]:
        payload["harm_rate"] = curves["harm_rate"]
        changed = True

    shield_metrics = payload.get("shield_metrics")
    if isinstance(shield_metrics, dict):
        if "suppressed_matches" in shield_metrics:
            shield_metrics.pop("suppressed_matches")
            changed = True
        for key in (
            "cmr",
            "harm_rate",
            "total_count",
            "latent_wrong_count",
            "false_match_mu_count",
            "corrected_count",
            "corrected_match_count",
            "latent_correct_count",
            "damaged_count",
            "risk_harm_match_count",
            "definition",
        ):
            if shield_metrics.get(key) != curves[key]:
                shield_metrics[key] = curves[key]
                changed = True
        if "suppressed_match_count" in shield_metrics:
            shield_metrics.pop("suppressed_match_count")
            changed = True
    else:
        payload["shield_metrics"] = {
            "cmr": curves["cmr"],
            "harm_rate": curves["harm_rate"],
            "total_count": curves["total_count"],
            "latent_wrong_count": curves["latent_wrong_count"],
            "false_match_mu_count": curves["false_match_mu_count"],
            "corrected_count": curves["corrected_count"],
            "corrected_match_count": curves["corrected_match_count"],
            "latent_correct_count": curves["latent_correct_count"],
            "damaged_count": curves["damaged_count"],
            "risk_harm_match_count": curves["risk_harm_match_count"],
            "definition": curves["definition"],
        }
        changed = True

    mean_internal = payload.get("mean_internal_metrics")
    if isinstance(mean_internal, dict):
        if "suppressed_match" in mean_internal and "corrected_match" not in mean_internal:
            mean_internal["corrected_match"] = mean_internal["suppressed_match"]
            changed = True
        if "latent_correct_match" not in mean_internal:
            latent_curve = np.asarray(curves["latent_correct_count"], dtype=np.float64)
            n_test = float(payload.get("n_test", 0) or 0)
            if n_test > 0:
                mean_internal["latent_correct_match"] = (latent_curve / n_test).tolist()
                changed = True

    if changed and apply:
        if backup:
            backup_once(path)
        save_json(path, payload)

    return changed


def update_npz(
    path: Path,
    curves: dict[str, Any],
    apply: bool,
    backup: bool,
    drop_legacy_key: bool,
) -> bool:
    with np.load(path) as loaded:
        payload = {key: loaded[key] for key in loaded.files}

    changed = False
    if "corrected_match" not in payload:
        payload["corrected_match"] = curves["corrected_match"]
        changed = True
    if "latent_correct_match" not in payload:
        payload["latent_correct_match"] = curves["latent_correct_match"]
        changed = True
    if drop_legacy_key and "suppressed_match" in payload:
        payload.pop("suppressed_match")
        changed = True

    if changed and apply:
        if backup:
            backup_once(path)
        np.savez_compressed(path, **payload)

    return changed


def iter_run_dirs(
    runs_root: Path,
    attack: str | None,
    model: str | None,
    run_id: str | None,
) -> list[Path]:
    pattern = "*"
    candidates = []
    for metrics_json in runs_root.rglob("metrics.json"):
        run_dir = metrics_json.parent
        if not (run_dir / "per_example_metrics.npz").exists():
            continue
        parts = run_dir.parts
        if "Shield" not in str(run_dir):
            continue
        if attack is not None and attack not in parts:
            continue
        if model is not None and model not in parts:
            continue
        if run_id is not None and run_dir.name != f"run_id={run_id}":
            continue
        candidates.append(run_dir)
    return sorted(candidates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Migrate old Shield suppressed_matches/harm_rate results to CMR."
    )
    parser.add_argument("--runs_root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--attack", default=None, help="Optional attack filter.")
    parser.add_argument("--model", default=None, help="Optional model filter.")
    parser.add_argument("--run_id", default=None, help="Optional run_id without prefix.")
    parser.add_argument("--apply", action="store_true", help="Write changes in place.")
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Do not create .bak files before writing changes.",
    )
    parser.add_argument(
        "--drop_legacy_npz_key",
        action="store_true",
        help="Remove suppressed_match from per_example_metrics.npz.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = iter_run_dirs(
        runs_root=args.runs_root,
        attack=args.attack,
        model=args.model,
        run_id=args.run_id,
    )

    if not run_dirs:
        print("No matching Shield runs found.")
        return

    print(f"Found {len(run_dirs)} matching Shield run(s).")
    if not args.apply:
        print("Dry run only. Re-run with --apply to write changes.")

    changed_runs = 0
    for run_dir in run_dirs:
        npz_path = run_dir / "per_example_metrics.npz"
        json_path = run_dir / "metrics.json"

        with np.load(npz_path) as loaded:
            curves = recompute_curves({key: loaded[key] for key in loaded.files})

        json_changed = update_metrics_json(
            json_path,
            curves,
            apply=args.apply,
            backup=not args.no_backup,
        )
        npz_changed = update_npz(
            npz_path,
            curves,
            apply=args.apply,
            backup=not args.no_backup,
            drop_legacy_key=args.drop_legacy_npz_key,
        )

        if json_changed or npz_changed:
            changed_runs += 1
        status = "changed" if (json_changed or npz_changed) else "already up to date"
        print(f"[{status}] {run_dir}")

    action = "Updated" if args.apply else "Would update"
    print(f"{action} {changed_runs}/{len(run_dirs)} run(s).")


if __name__ == "__main__":
    main()
