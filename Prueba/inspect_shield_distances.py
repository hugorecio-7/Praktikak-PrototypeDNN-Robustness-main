"""Print the latent and Shield distance terms for a B30 Shield model.

By default this script only loads a model, runs clean MNIST test samples through
the Shield wrapper, and prints the distance decomposition:

    d_final = d_latent + gamma * (1 - SSIM(x, Dec(p_j))) ** power

Use --save to write CSV files, or --write_metrics_json to add a compact global
summary to an existing run's metrics.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_loader import get_test_loader
from adversarial_attacks import AutoAttack_adv, PGDLInf_attack
from loss_functions import CELoss
from run_metrics import B30_SHIELD_BASE_NAMES, is_shield_model_name, load_models


def resolve_shield_name(model_name: str) -> str:
    if is_shield_model_name(model_name):
        base_name = model_name[: -len("-Shield")]
        if base_name not in B30_SHIELD_BASE_NAMES:
            available = ", ".join(f"{name}-Shield" for name in B30_SHIELD_BASE_NAMES)
            raise ValueError(
                f"Shield is only available for B30 models. Got {model_name!r}. "
                f"Available Shield models: {available}."
            )
        return model_name

    if model_name in B30_SHIELD_BASE_NAMES:
        return f"{model_name}-Shield"

    available = ", ".join(f"{name}-Shield" for name in B30_SHIELD_BASE_NAMES)
    raise ValueError(
        f"This script inspects Shield distances, so the model must be a B30 "
        f"Shield model. Got {model_name!r}. Available Shield models: {available}."
    )


def tensor_stats(name: str, tensor: torch.Tensor) -> str:
    values = tensor.detach().float().reshape(-1)
    return (
        f"{name:<24} "
        f"mean={values.mean().item():.6f} "
        f"std={values.std(unbiased=False).item():.6f} "
        f"min={values.min().item():.6f} "
        f"max={values.max().item():.6f}"
    )


def tensor_stats_values(tensor: torch.Tensor) -> dict[str, float]:
    values = tensor.detach().float().reshape(-1)
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "min": values.min().item(),
        "max": values.max().item(),
    }


def empty_running_stats() -> dict[str, float]:
    return {
        "count": 0,
        "sum": 0.0,
        "sum_sq": 0.0,
        "min": float("inf"),
        "max": float("-inf"),
    }


def update_running_stats(stats: dict[str, float], tensor: torch.Tensor) -> None:
    values = tensor.detach().float().reshape(-1)
    if values.numel() == 0:
        return
    stats["count"] += int(values.numel())
    stats["sum"] += float(values.sum().item())
    stats["sum_sq"] += float((values * values).sum().item())
    stats["min"] = min(stats["min"], float(values.min().item()))
    stats["max"] = max(stats["max"], float(values.max().item()))


def finalize_running_stats(stats: dict[str, float]) -> dict[str, float | int]:
    count = int(stats["count"])
    if count == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }

    mean = stats["sum"] / count
    variance = max((stats["sum_sq"] / count) - (mean * mean), 0.0)
    return {
        "count": count,
        "mean": mean,
        "std": variance ** 0.5,
        "min": stats["min"],
        "max": stats["max"],
    }


def safe_percent(num: torch.Tensor, den: torch.Tensor) -> float:
    den_value = den.detach().float().mean().item()
    if abs(den_value) < 1e-12:
        return float("nan")
    return (num.detach().float().mean().item() / den_value) * 100.0


def safe_percent_from_stats(num_stats: dict, den_stats: dict) -> float:
    den_mean = float(den_stats["mean"])
    if abs(den_mean) < 1e-12:
        return float("nan")
    return (float(num_stats["mean"]) / den_mean) * 100.0


def safe_filename(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(":", "_")


def term_contribution_scope(
    d_latent: torch.Tensor,
    gamma_term: torch.Tensor,
    d_final: torch.Tensor,
) -> dict:
    latent_pct = safe_percent(d_latent, d_final)
    gamma_pct = safe_percent(gamma_term, d_final)
    if latent_pct != latent_pct or gamma_pct != gamma_pct:
        dominant_term = "unknown"
    elif gamma_pct > latent_pct:
        dominant_term = "gamma_term"
    else:
        dominant_term = "d_latent"

    return {
        "d_latent": tensor_stats_values(d_latent),
        "gamma_term": tensor_stats_values(gamma_term),
        "d_final": tensor_stats_values(d_final),
        "d_latent_pct_of_d_final": latent_pct,
        "gamma_term_pct_of_d_final": gamma_pct,
        "dominant_term": dominant_term,
    }


def build_distance_contribution_summary(
    model_name: str,
    model,
    processed: int,
    latent_all: torch.Tensor,
    one_minus_all: torch.Tensor,
    powered_all: torch.Tensor,
    penalty_all: torch.Tensor,
    final_all: torch.Tensor,
    nearest_latent: torch.Tensor,
    nearest_penalty: torch.Tensor,
    nearest_final: torch.Tensor,
) -> dict:
    return {
        "model": model_name,
        "computed_on": "clean_test_set",
        "n_samples": int(processed),
        "n_prototypes": int(model.n_prototypes),
        "gamma": float(model.gamma),
        "power": int(model.power),
        "formula": (
            "d_final(x,p_j) = d_latent(x,p_j) + "
            "gamma * (1 - SSIM(x, Dec(p_j))) ** power"
        ),
        "aggregation": (
            "Percentages are ratios of means: "
            "100 * mean(term) / mean(d_final)."
        ),
        "all_sample_prototype_pairs": {
            **term_contribution_scope(latent_all, penalty_all, final_all),
            "one_minus_ssim": tensor_stats_values(one_minus_all),
            "powered_term": tensor_stats_values(powered_all),
        },
        "nearest_final_prototype": term_contribution_scope(
            nearest_latent,
            nearest_penalty,
            nearest_final,
        ),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_metrics_json(metrics_json_path: Path) -> dict:
    with metrics_json_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def infer_shield_attack_mode(metrics: dict) -> str:
    shield = metrics.get("shield") or {}
    attack_params = metrics.get("attack_params") or {}
    mode = (
        shield.get("attack_mode")
        or shield.get("pgd_mode")
        or attack_params.get("shield_attack_mode")
        or attack_params.get("shield_pgd_mode")
        or "white-box"
    )
    if mode not in {"white-box", "grey-box"}:
        raise ValueError(f"Unsupported Shield attack mode in metrics.json: {mode!r}")
    return mode


def make_empty_contribution_accumulator() -> dict:
    return {
        "n_samples": 0,
        "n_pairs": 0,
        "all_sample_prototype_pairs": {
            "d_latent": empty_running_stats(),
            "one_minus_ssim": empty_running_stats(),
            "powered_term": empty_running_stats(),
            "gamma_term": empty_running_stats(),
            "d_final": empty_running_stats(),
        },
        "nearest_final_prototype": {
            "d_latent": empty_running_stats(),
            "gamma_term": empty_running_stats(),
            "d_final": empty_running_stats(),
        },
    }


def update_contribution_accumulator(
    accumulator: dict,
    d_latent: torch.Tensor,
    one_minus_ssim: torch.Tensor,
    powered: torch.Tensor,
    penalty: torch.Tensor,
    d_final: torch.Tensor,
) -> None:
    batch_size, n_prototypes = d_final.shape
    accumulator["n_samples"] += int(batch_size)
    accumulator["n_pairs"] += int(batch_size * n_prototypes)

    all_pairs = accumulator["all_sample_prototype_pairs"]
    update_running_stats(all_pairs["d_latent"], d_latent)
    update_running_stats(all_pairs["one_minus_ssim"], one_minus_ssim)
    update_running_stats(all_pairs["powered_term"], powered)
    update_running_stats(all_pairs["gamma_term"], penalty)
    update_running_stats(all_pairs["d_final"], d_final)

    nearest_final_idx = torch.argmin(d_final, dim=1)
    row_idx = torch.arange(batch_size, device=d_final.device)
    nearest = accumulator["nearest_final_prototype"]
    update_running_stats(nearest["d_latent"], d_latent[row_idx, nearest_final_idx])
    update_running_stats(nearest["gamma_term"], penalty[row_idx, nearest_final_idx])
    update_running_stats(nearest["d_final"], d_final[row_idx, nearest_final_idx])


def finalize_contribution_scope(raw_scope: dict) -> dict:
    d_latent = finalize_running_stats(raw_scope["d_latent"])
    gamma_term = finalize_running_stats(raw_scope["gamma_term"])
    d_final = finalize_running_stats(raw_scope["d_final"])
    latent_pct = safe_percent_from_stats(d_latent, d_final)
    gamma_pct = safe_percent_from_stats(gamma_term, d_final)

    if latent_pct != latent_pct or gamma_pct != gamma_pct:
        dominant_term = "unknown"
    elif gamma_pct > latent_pct:
        dominant_term = "gamma_term"
    else:
        dominant_term = "d_latent"

    finalized = {
        "d_latent": d_latent,
        "gamma_term": gamma_term,
        "d_final": d_final,
        "d_latent_pct_of_d_final": latent_pct,
        "gamma_term_pct_of_d_final": gamma_pct,
        "dominant_term": dominant_term,
    }
    if "one_minus_ssim" in raw_scope:
        finalized["one_minus_ssim"] = finalize_running_stats(raw_scope["one_minus_ssim"])
    if "powered_term" in raw_scope:
        finalized["powered_term"] = finalize_running_stats(raw_scope["powered_term"])
    return finalized


def finalize_contribution_accumulator(eps: float, accumulator: dict) -> dict:
    return {
        "eps": float(eps),
        "n_samples": int(accumulator["n_samples"]),
        "n_pairs": int(accumulator["n_pairs"]),
        "all_sample_prototype_pairs": finalize_contribution_scope(
            accumulator["all_sample_prototype_pairs"]
        ),
        "nearest_final_prototype": finalize_contribution_scope(
            accumulator["nearest_final_prototype"]
        ),
    }


def finalize_compact_by_epsilon(
    eps_values: list[float],
    accumulators: list[dict],
) -> dict:
    latent_pct = []
    gamma_pct = []
    dominant_terms = []
    n_samples = []

    for accumulator in accumulators:
        nearest = finalize_contribution_scope(accumulator["nearest_final_prototype"])
        latent_value = float(nearest["d_latent_pct_of_d_final"])
        gamma_value = float(nearest["gamma_term_pct_of_d_final"])
        latent_pct.append(latent_value)
        gamma_pct.append(gamma_value)
        dominant_terms.append(nearest["dominant_term"])
        n_samples.append(int(accumulator["n_samples"]))

    if all(term == dominant_terms[0] for term in dominant_terms):
        dominant_summary = dominant_terms[0]
    else:
        dominant_summary = "changes_by_epsilon"

    return {
        "scope": "nearest_final_prototype",
        "eps": [float(eps) for eps in eps_values],
        "d_latent_pct_of_d_final": latent_pct,
        "gamma_term_pct_of_d_final": gamma_pct,
        "dominant_term_by_epsilon": dominant_terms,
        "dominant_term_summary": dominant_summary,
        "n_samples_by_epsilon": n_samples,
    }


def extract_distance_terms(model, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
    logits = model(images)
    d_latent = model.last_d_latente.detach()
    ssim = model._pairwise_ssim(images).detach()
    one_minus_ssim = 1.0 - ssim
    powered = one_minus_ssim ** int(model.power)
    penalty = float(model.gamma) * powered
    d_final = d_latent + penalty

    stored_diff = (d_final - model.last_d_final.detach()).abs().max().item()
    if stored_diff > 1e-5:
        print(
            f"WARNING: manual d_final differs from wrapper by "
            f"{stored_diff:.8f}."
        )

    return logits, d_latent, ssim, one_minus_ssim, powered, penalty, d_final


def generate_adversarial_batch(
    attack_name: str,
    attack_params: dict,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    attack_model,
    eps: float,
) -> torch.Tensor:
    if float(eps) == 0.0:
        return batch_x

    if attack_name == "PGDLInf_attack":
        def loss_f(batch_x: torch.Tensor) -> torch.Tensor:
            return CELoss(model=attack_model, batch_x=batch_x, batch_y=batch_y)

        return PGDLInf_attack(
            batch_x=batch_x,
            loss_f=loss_f,
            eps=float(eps),
            iters=int(attack_params.get("iters", 80)),
            alpha=float(attack_params.get("alpha", 0.01)),
            random_start=bool(attack_params.get("random_start", True)),
        )

    if attack_name == "AutoAttack_adv":
        return AutoAttack_adv(
            batch_x=batch_x,
            batch_y=batch_y,
            model=attack_model,
            epsilon=float(eps),
            version=attack_params.get("version", "standard"),
        )

    raise ValueError(
        "By-epsilon Shield contribution is currently implemented for "
        f"PGDLInf_attack and AutoAttack_adv, got {attack_name!r}."
    )


def compute_distance_contribution_by_epsilon(
    model_name: str,
    model,
    attack_model,
    test_loader,
    metrics: dict,
    device: torch.device,
    num_batches: int,
    shield_attack_mode: str,
) -> dict:
    attack_name = metrics.get("attack")
    if attack_name is None:
        raise ValueError("metrics.json is missing the 'attack' field.")
    eps_values = [float(value) for value in metrics.get("eps", [])]
    if not eps_values:
        raise ValueError("metrics.json is missing a non-empty 'eps' field.")
    attack_params = dict(metrics.get("attack_params") or {})

    accumulators = [make_empty_contribution_accumulator() for _ in eps_values]

    for batch_idx, (images, labels) in enumerate(test_loader):
        if batch_idx >= num_batches:
            break
        print(f"By-epsilon contribution batch {batch_idx + 1}/{num_batches}")

        images = images.to(device)
        labels = labels.to(device)

        for eps_idx, eps in enumerate(eps_values):
            print(
                f"  eps {eps_idx + 1}/{len(eps_values)} "
                f"| eps={eps:g}"
            )
            x_eval = generate_adversarial_batch(
                attack_name=attack_name,
                attack_params=attack_params,
                batch_x=images,
                batch_y=labels,
                attack_model=attack_model,
                eps=eps,
            )

            with torch.no_grad():
                (
                    _logits,
                    d_latent,
                    _ssim,
                    one_minus_ssim,
                    powered,
                    penalty,
                    d_final,
                ) = extract_distance_terms(model, x_eval)

            update_contribution_accumulator(
                accumulator=accumulators[eps_idx],
                d_latent=d_latent,
                one_minus_ssim=one_minus_ssim,
                powered=powered,
                penalty=penalty,
                d_final=d_final,
            )

    compact = finalize_compact_by_epsilon(eps_values, accumulators)

    return {
        "model": model_name,
        "computed_on": "adversarial_examples_recomputed_from_run",
        "attack": attack_name,
        "shield_attack_mode": shield_attack_mode,
        "n_eps": len(eps_values),
        "eps": eps_values,
        "gamma": float(model.gamma),
        "power": int(model.power),
        "formula": (
            "d_final(x,p_j) = d_latent(x,p_j) + "
            "gamma * (1 - SSIM(x, Dec(p_j))) ** power"
        ),
        "aggregation": (
            "For each epsilon, percentages are ratios of means: "
            "100 * mean(term) / mean(d_final). Adversarial examples are "
            "recomputed with the attack and Shield attack mode stored in this run. "
            "Only the nearest-final prototype is stored to keep metrics.json compact."
        ),
        "loop_order": "batch_then_epsilon",
        **compact,
    }

def resolve_metrics_json_path(args: argparse.Namespace, model_name: str) -> Path:
    if args.metrics_json is not None:
        return args.metrics_json

    missing = [
        name for name in ("attack", "run_id") if getattr(args, name) in (None, "")
    ]
    if missing:
        raise ValueError(
            "To write metrics.json, pass either --metrics_json or "
            "--attack and --run_id. Missing: " + ", ".join(f"--{name}" for name in missing)
        )

    return (
        ROOT
        / args.out_root
        / args.dataset
        / args.threat
        / args.attack
        / model_name
        / f"seed={args.seed}"
        / f"run_id={args.run_id}"
        / "metrics.json"
    )


def write_distance_contribution_to_metrics_json(
    metrics_json_path: Path,
    model_name: str,
    contribution: dict | None = None,
    contribution_by_epsilon: dict | None = None,
) -> None:
    if not metrics_json_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_json_path}")

    with metrics_json_path.open("r", encoding="utf-8") as fh:
        metrics = json.load(fh)

    metrics_model = metrics.get("model")
    if metrics_model is not None and metrics_model != model_name:
        raise ValueError(
            f"metrics.json belongs to model {metrics_model!r}, but the inspected "
            f"model is {model_name!r}: {metrics_json_path}"
        )

    shield_block = metrics.setdefault("shield", {})
    if contribution is not None:
        shield_block["distance_contribution"] = contribution
    if contribution_by_epsilon is not None:
        shield_block["distance_contribution_by_epsilon"] = contribution_by_epsilon

    with metrics_json_path.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
        fh.write("\n")


def infer_proto_labels(model) -> list[int | None]:
    n_prototypes = model.n_prototypes
    n_classes = model.base_model.fc.linear.out_features
    if n_prototypes % n_classes != 0:
        return [None for _ in range(n_prototypes)]
    prototypes_by_class = n_prototypes // n_classes
    return [idx // prototypes_by_class for idx in range(n_prototypes)]


def open_detail_csv(model_name: str):
    output_dir = SCRIPT_DIR / "Inspect"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = safe_filename(model_name)
    detail_path = output_dir / f"shield_distances_{safe_model}_{timestamp}.csv"
    summary_path = output_dir / f"shield_distance_summary_{safe_model}_{timestamp}.csv"
    detail_file = detail_path.open("w", newline="", encoding="utf-8")
    fieldnames = [
        "sample_idx",
        "batch_idx",
        "batch_sample_idx",
        "label",
        "pred_shield",
        "proto_idx",
        "proto_class",
        "d_latent",
        "ssim",
        "one_minus_ssim",
        "powered_term",
        "gamma_term",
        "d_final",
        "d_latent_pct_of_final",
        "gamma_term_pct_of_final",
        "nearest_latent_proto",
        "nearest_final_proto",
        "is_nearest_latent",
        "is_nearest_final",
    ]
    writer = csv.DictWriter(detail_file, fieldnames=fieldnames)
    writer.writeheader()
    return detail_file, writer, detail_path, summary_path


def write_detail_rows(
    writer: csv.DictWriter,
    batch_idx: int,
    global_start_idx: int,
    labels: torch.Tensor,
    preds: torch.Tensor,
    d_latent: torch.Tensor,
    ssim: torch.Tensor,
    one_minus_ssim: torch.Tensor,
    powered: torch.Tensor,
    penalty: torch.Tensor,
    d_final: torch.Tensor,
    proto_labels: list[int | None],
) -> None:
    labels_cpu = labels.detach().cpu().tolist()
    preds_cpu = preds.detach().cpu().tolist()
    d_latent_cpu = d_latent.detach().cpu()
    ssim_cpu = ssim.detach().cpu()
    one_minus_cpu = one_minus_ssim.detach().cpu()
    powered_cpu = powered.detach().cpu()
    penalty_cpu = penalty.detach().cpu()
    d_final_cpu = d_final.detach().cpu()
    nearest_latent = torch.argmin(d_latent_cpu, dim=1).tolist()
    nearest_final = torch.argmin(d_final_cpu, dim=1).tolist()

    rows = []
    batch_size, n_prototypes = d_final_cpu.shape
    for i in range(batch_size):
        sample_idx = global_start_idx + i
        nearest_latent_i = int(nearest_latent[i])
        nearest_final_i = int(nearest_final[i])
        for proto_idx in range(n_prototypes):
            final_value = d_final_cpu[i, proto_idx].item()
            if abs(final_value) < 1e-12:
                latent_pct = ""
                gamma_pct = ""
            else:
                latent_pct = f"{100.0 * d_latent_cpu[i, proto_idx].item() / final_value:.10g}"
                gamma_pct = f"{100.0 * penalty_cpu[i, proto_idx].item() / final_value:.10g}"
            rows.append(
                {
                    "sample_idx": sample_idx,
                    "batch_idx": batch_idx,
                    "batch_sample_idx": i,
                    "label": int(labels_cpu[i]),
                    "pred_shield": int(preds_cpu[i]),
                    "proto_idx": proto_idx,
                    "proto_class": proto_labels[proto_idx],
                    "d_latent": f"{d_latent_cpu[i, proto_idx].item():.10g}",
                    "ssim": f"{ssim_cpu[i, proto_idx].item():.10g}",
                    "one_minus_ssim": f"{one_minus_cpu[i, proto_idx].item():.10g}",
                    "powered_term": f"{powered_cpu[i, proto_idx].item():.10g}",
                    "gamma_term": f"{penalty_cpu[i, proto_idx].item():.10g}",
                    "d_final": f"{d_final_cpu[i, proto_idx].item():.10g}",
                    "d_latent_pct_of_final": latent_pct,
                    "gamma_term_pct_of_final": gamma_pct,
                    "nearest_latent_proto": nearest_latent_i,
                    "nearest_final_proto": nearest_final_i,
                    "is_nearest_latent": int(proto_idx == nearest_latent_i),
                    "is_nearest_final": int(proto_idx == nearest_final_i),
                }
            )
    writer.writerows(rows)


def print_sample_details(
    sample_idx: int,
    label: int,
    pred: int,
    d_latent: torch.Tensor,
    ssim: torch.Tensor,
    one_minus_ssim: torch.Tensor,
    powered: torch.Tensor,
    penalty: torch.Tensor,
    d_final: torch.Tensor,
    proto_labels: list[int | None],
    top_k: int,
) -> None:
    nearest_latent = int(torch.argmin(d_latent).item())
    nearest_final = int(torch.argmin(d_final).item())
    top_final = torch.argsort(d_final)[:top_k].tolist()

    def proto_label(proto_idx: int) -> str:
        value = proto_labels[proto_idx]
        return "?" if value is None else str(value)

    print()
    print(f"Sample {sample_idx} | y={label} | pred_shield={pred}")
    print(
        "  nearest latent: "
        f"proto={nearest_latent:02d} class={proto_label(nearest_latent)} "
        f"d_latent={d_latent[nearest_latent].item():.6f} "
        f"gamma_term={penalty[nearest_latent].item():.6f} "
        f"d_final={d_final[nearest_latent].item():.6f} "
        f"ssim={ssim[nearest_latent].item():.6f}"
    )
    print(
        "  nearest final : "
        f"proto={nearest_final:02d} class={proto_label(nearest_final)} "
        f"d_latent={d_latent[nearest_final].item():.6f} "
        f"gamma_term={penalty[nearest_final].item():.6f} "
        f"d_final={d_final[nearest_final].item():.6f} "
        f"ssim={ssim[nearest_final].item():.6f}"
    )
    print("  top final prototypes:")
    print(
        "    proto cls  d_latent   gamma_term  d_final   latent%  gamma%   1-ssim    (1-ssim)^p"
    )
    for proto_idx in top_final:
        final_value = d_final[proto_idx].item()
        if abs(final_value) < 1e-12:
            latent_pct = float("nan")
            gamma_pct = float("nan")
        else:
            latent_pct = 100.0 * d_latent[proto_idx].item() / final_value
            gamma_pct = 100.0 * penalty[proto_idx].item() / final_value
        print(
            f"    {proto_idx:>5d} {proto_label(proto_idx):>3} "
            f"{d_latent[proto_idx].item():>9.6f} "
            f"{penalty[proto_idx].item():>11.6f} "
            f"{d_final[proto_idx].item():>8.6f} "
            f"{latent_pct:>8.2f} "
            f"{gamma_pct:>7.2f} "
            f"{one_minus_ssim[proto_idx].item():>9.6f} "
            f"{powered[proto_idx].item():>12.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect how much d_latent and the Shield structural term contribute "
            "to d_final for a B30 Shield model."
        )
    )
    parser.add_argument(
        "--model",
        required=True,
        help=(
            "B30 Shield model to inspect, e.g. B30-Shield or B30-FT-E-M-Shield. "
            "If a B30 base name is passed, '-Shield' is added automatically."
        ),
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data"))
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5,
        help="Number of individual samples to print in detail.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of nearest-final prototypes to print per sample.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help=(
            "Save detailed CSV files under Prueba/Inspect. By default the script "
            "only prints to the terminal."
        ),
    )
    parser.add_argument(
        "--write_metrics_json",
        action="store_true",
        help=(
            "Add the global Shield distance contribution summary to the selected "
            "run's metrics.json under shield.distance_contribution."
        ),
    )
    parser.add_argument(
        "--by_epsilon",
        action="store_true",
        help=(
            "Recompute the run attack for every epsilon in metrics.json and add "
            "shield.distance_contribution_by_epsilon. Requires --write_metrics_json."
        ),
    )
    parser.add_argument(
        "--metrics_json",
        type=Path,
        default=None,
        help=(
            "Direct path to the metrics.json file to update. If omitted, the "
            "path is built from --out_root, --dataset, --threat, --attack, "
            "--model, --seed and --run_id."
        ),
    )
    parser.add_argument("--out_root", type=Path, default=Path("resultsTFG/runs_v1"))
    parser.add_argument("--dataset", type=str, default="mnist")
    parser.add_argument("--threat", type=str, default="Linf")
    parser.add_argument(
        "--attack",
        type=str,
        default=None,
        help="Attack folder for the run to update, e.g. PGDLInf_attack or AutoAttack_adv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed folder for the run to update.",
    )
    parser.add_argument(
        "--run_id",
        type=str,
        default=None,
        help="Run id folder for the run to update.",
    )
    args = parser.parse_args()
    if args.by_epsilon and not args.write_metrics_json:
        raise ValueError("--by_epsilon requires --write_metrics_json.")

    model_name = resolve_shield_name(args.model)
    set_seed(args.seed)
    metrics_json_target = None
    metrics_for_run = None
    shield_attack_mode = "white-box"
    if args.write_metrics_json:
        metrics_json_target = resolve_metrics_json_path(args, model_name)
        if not metrics_json_target.exists():
            raise FileNotFoundError(f"metrics.json not found: {metrics_json_target}")
        metrics_for_run = load_metrics_json(metrics_json_target)
        if args.by_epsilon:
            shield_attack_mode = infer_shield_attack_mode(metrics_for_run)

    models, _, attack_models = load_models(
        [model_name],
        shield_pgd_mode=shield_attack_mode,
    )
    model = models[0]
    attack_model = attack_models[0]
    device = next(model.parameters()).device
    proto_labels = infer_proto_labels(model)

    test_loader = get_test_loader(
        str(args.data_dir),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    all_latent: list[torch.Tensor] = []
    all_one_minus_ssim: list[torch.Tensor] = []
    all_powered: list[torch.Tensor] = []
    all_penalty: list[torch.Tensor] = []
    all_final: list[torch.Tensor] = []
    all_nearest_latent: list[torch.Tensor] = []
    all_nearest_penalty: list[torch.Tensor] = []
    all_nearest_final: list[torch.Tensor] = []

    detail_file = None
    detail_writer = None
    detail_path = None
    summary_path = None
    if args.save:
        detail_file, detail_writer, detail_path, summary_path = open_detail_csv(model_name)
        print(f"Saving detailed distances to: {detail_path}")

    printed = 0
    processed = 0

    try:
        print()
        print(f"Model: {model_name}")
        print(f"gamma={model.gamma:g} | power={model.power}")
        print(
            "Formula: d_final(x,p_j) = d_latent(x,p_j) + "
            "gamma * (1 - SSIM(x, Dec(p_j))) ** power"
        )

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                if batch_idx >= args.num_batches:
                    break

                images = images.to(device)
                labels = labels.to(device)

                logits = model(images)
                preds = logits.argmax(dim=1)

                d_latent = model.last_d_latente.detach()
                ssim = model._pairwise_ssim(images).detach()
                one_minus_ssim = 1.0 - ssim
                powered = one_minus_ssim ** int(model.power)
                penalty = float(model.gamma) * powered
                d_final = d_latent + penalty

                stored_diff = (d_final - model.last_d_final.detach()).abs().max().item()
                if stored_diff > 1e-5:
                    print(
                        f"WARNING: manual d_final differs from wrapper by "
                        f"{stored_diff:.8f} in batch {batch_idx}."
                    )

                if detail_writer is not None:
                    write_detail_rows(
                        writer=detail_writer,
                        batch_idx=batch_idx,
                        global_start_idx=processed,
                        labels=labels,
                        preds=preds,
                        d_latent=d_latent,
                        ssim=ssim,
                        one_minus_ssim=one_minus_ssim,
                        powered=powered,
                        penalty=penalty,
                        d_final=d_final,
                        proto_labels=proto_labels,
                    )

                nearest_final_idx = torch.argmin(d_final, dim=1)
                row_idx = torch.arange(d_final.shape[0], device=d_final.device)

                all_latent.append(d_latent.cpu())
                all_one_minus_ssim.append(one_minus_ssim.cpu())
                all_powered.append(powered.cpu())
                all_penalty.append(penalty.cpu())
                all_final.append(d_final.cpu())
                all_nearest_latent.append(d_latent[row_idx, nearest_final_idx].cpu())
                all_nearest_penalty.append(penalty[row_idx, nearest_final_idx].cpu())
                all_nearest_final.append(d_final[row_idx, nearest_final_idx].cpu())

                for i in range(images.shape[0]):
                    if printed >= args.num_samples:
                        break
                    print_sample_details(
                        sample_idx=processed + i,
                        label=int(labels[i].item()),
                        pred=int(preds[i].item()),
                        d_latent=d_latent[i].cpu(),
                        ssim=ssim[i].cpu(),
                        one_minus_ssim=one_minus_ssim[i].cpu(),
                        powered=powered[i].cpu(),
                        penalty=penalty[i].cpu(),
                        d_final=d_final[i].cpu(),
                        proto_labels=proto_labels,
                        top_k=min(args.top_k, d_final.shape[1]),
                    )
                    printed += 1

                processed += images.shape[0]
    finally:
        if detail_file is not None:
            detail_file.close()

    if processed == 0:
        raise RuntimeError("No samples were processed. Check --batch_size and --num_batches.")

    latent_all = torch.cat([x.reshape(-1) for x in all_latent])
    one_minus_all = torch.cat([x.reshape(-1) for x in all_one_minus_ssim])
    powered_all = torch.cat([x.reshape(-1) for x in all_powered])
    penalty_all = torch.cat([x.reshape(-1) for x in all_penalty])
    final_all = torch.cat([x.reshape(-1) for x in all_final])
    nearest_latent = torch.cat(all_nearest_latent)
    nearest_penalty = torch.cat(all_nearest_penalty)
    nearest_final = torch.cat(all_nearest_final)
    contribution_summary = build_distance_contribution_summary(
        model_name=model_name,
        model=model,
        processed=processed,
        latent_all=latent_all,
        one_minus_all=one_minus_all,
        powered_all=powered_all,
        penalty_all=penalty_all,
        final_all=final_all,
        nearest_latent=nearest_latent,
        nearest_penalty=nearest_penalty,
        nearest_final=nearest_final,
    )
    contribution_by_epsilon = None

    print()
    print(f"Summary over {processed} samples and {model.n_prototypes} prototypes")
    print(tensor_stats("d_latent", latent_all))
    print(tensor_stats("1 - SSIM", one_minus_all))
    print(tensor_stats("(1 - SSIM)^power", powered_all))
    print(tensor_stats("gamma * term", penalty_all))
    print(tensor_stats("d_final", final_all))
    print()
    print("Contribution using all sample-prototype pairs")
    print(f"  mean(gamma_term) / mean(d_latent) = {safe_percent(penalty_all, latent_all):.2f}%")
    print(f"  mean(d_latent) / mean(d_final)   = {safe_percent(latent_all, final_all):.2f}%")
    print(f"  mean(gamma_term) / mean(d_final) = {safe_percent(penalty_all, final_all):.2f}%")
    print()
    print("Contribution only at each sample's nearest-final prototype")
    print(tensor_stats("nearest d_latent", nearest_latent))
    print(tensor_stats("nearest gamma term", nearest_penalty))
    print(tensor_stats("nearest d_final", nearest_final))
    print(
        "  mean(nearest gamma_term) / mean(nearest d_latent) = "
        f"{safe_percent(nearest_penalty, nearest_latent):.2f}%"
    )
    print(
        "  mean(nearest d_latent) / mean(nearest d_final)   = "
        f"{safe_percent(nearest_latent, nearest_final):.2f}%"
    )
    print(
        "  mean(nearest gamma_term) / mean(nearest d_final) = "
        f"{safe_percent(nearest_penalty, nearest_final):.2f}%"
    )

    if args.by_epsilon:
        print()
        print(
            "Computing Shield distance contribution by epsilon "
            f"({shield_attack_mode})..."
        )
        contribution_by_epsilon = compute_distance_contribution_by_epsilon(
            model_name=model_name,
            model=model,
            attack_model=attack_model,
            test_loader=test_loader,
            metrics=metrics_for_run,
            device=device,
            num_batches=args.num_batches,
            shield_attack_mode=shield_attack_mode,
        )

    if summary_path is not None:
        summary_rows = []
        for scope, tensors in [
            (
                "all_sample_prototype_pairs",
                [
                    ("d_latent", latent_all),
                    ("1_minus_ssim", one_minus_all),
                    ("powered_term", powered_all),
                    ("gamma_term", penalty_all),
                    ("d_final", final_all),
                ],
            ),
            (
                "nearest_final_prototype",
                [
                    ("d_latent", nearest_latent),
                    ("gamma_term", nearest_penalty),
                    ("d_final", nearest_final),
                ],
            ),
        ]:
            for term_name, tensor in tensors:
                stats = tensor_stats_values(tensor)
                summary_rows.append(
                    {
                        "scope": scope,
                        "name": term_name,
                        "mean": f"{stats['mean']:.10g}",
                        "std": f"{stats['std']:.10g}",
                        "min": f"{stats['min']:.10g}",
                        "max": f"{stats['max']:.10g}",
                        "value": "",
                    }
                )
        summary_rows.extend(
            [
                {
                    "scope": "all_sample_prototype_pairs",
                    "name": "mean_gamma_term_over_mean_d_latent_percent",
                    "mean": "",
                    "std": "",
                    "min": "",
                    "max": "",
                    "value": f"{safe_percent(penalty_all, latent_all):.10g}",
                },
                {
                    "scope": "all_sample_prototype_pairs",
                    "name": "mean_d_latent_over_mean_d_final_percent",
                    "mean": "",
                    "std": "",
                    "min": "",
                    "max": "",
                    "value": f"{safe_percent(latent_all, final_all):.10g}",
                },
                {
                    "scope": "all_sample_prototype_pairs",
                    "name": "mean_gamma_term_over_mean_d_final_percent",
                    "mean": "",
                    "std": "",
                    "min": "",
                    "max": "",
                    "value": f"{safe_percent(penalty_all, final_all):.10g}",
                },
                {
                    "scope": "nearest_final_prototype",
                    "name": "mean_gamma_term_over_mean_d_latent_percent",
                    "mean": "",
                    "std": "",
                    "min": "",
                    "max": "",
                    "value": f"{safe_percent(nearest_penalty, nearest_latent):.10g}",
                },
                {
                    "scope": "nearest_final_prototype",
                    "name": "mean_d_latent_over_mean_d_final_percent",
                    "mean": "",
                    "std": "",
                    "min": "",
                    "max": "",
                    "value": f"{safe_percent(nearest_latent, nearest_final):.10g}",
                },
                {
                    "scope": "nearest_final_prototype",
                    "name": "mean_gamma_term_over_mean_d_final_percent",
                    "mean": "",
                    "std": "",
                    "min": "",
                    "max": "",
                    "value": f"{safe_percent(nearest_penalty, nearest_final):.10g}",
                },
            ]
        )
        with summary_path.open("w", newline="", encoding="utf-8") as summary_file:
            writer = csv.DictWriter(
                summary_file,
                fieldnames=["scope", "name", "mean", "std", "min", "max", "value"],
            )
            writer.writeheader()
            writer.writerows(summary_rows)
        print()
        print(f"Saved detailed CSV: {detail_path}")
        print(f"Saved summary CSV : {summary_path}")

    if args.write_metrics_json:
        write_distance_contribution_to_metrics_json(
            metrics_json_path=metrics_json_target,
            model_name=model_name,
            contribution=contribution_summary,
            contribution_by_epsilon=contribution_by_epsilon,
        )
        print()
        print(f"Updated metrics.json: {metrics_json_target}")


if __name__ == "__main__":
    main()
