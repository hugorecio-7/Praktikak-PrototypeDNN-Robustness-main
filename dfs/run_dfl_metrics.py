"""Structured DFS/DFL evaluation for B30 deterministic, mean, and risk variants."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPO_DFL_HEAD_ROOT = REPO_ROOT / "dfs" / "results" / "dfl_heads"

from adversarial_attacks import PGDLInf_attack  # noqa: E402
from data_loader import get_test_loader  # noqa: E402
from metric_calculators import (  # noqa: E402
    calc_R_dec,
    calc_R_enc,
    calc_artificial_prototype_match_suppression,
    calc_m_pred,
    calc_m_proto,
    calc_m_proto_from_distances,
    compute_empirical_robustness_interval,
    compute_icr,
    compute_r_pgd_binary_search,
)
from metric_extractor import extract_internals, get_proto_labels  # noqa: E402
from modules import CAEModel_Balanced  # noqa: E402
from dfs.dfl_layer import DistributionalPrototypeDistanceLayer  # noqa: E402
from dfs.risk_b30_wrapper import RiskAwareB30Wrapper  # noqa: E402


DATASET_NAME = "mnist"
THREAT_NAME = "Linf"

B30_PATHS = {
    "B30": (
        "saved_model/mnist_model/"
        "mnist_cae_balanced_clstsep_1500_0.002_250_True_0.0_20_1_1_1_"
        "1.0_0.0_30_4_32_1/mnist_cae00750.pth"
    ),
    "B30-FT-0": (
        "tfg_models/B30/B30-FT-0/seed=1/checkpoints/"
        "B30_B30-FT-0_seed1_best_val_adv_acc.pth"
    ),
    "B30-FT-1": (
        "tfg_models/B30/B30-FT-1/seed=1/checkpoints/"
        "B30_B30-FT-1_seed1_best_val_adv_acc.pth"
    ),
    "B30-FT-2": (
        "tfg_models/B30/B30-FT-2/seed=1/checkpoints/"
        "B30_B30-FT-2_seed1_best_val_adv_acc.pth"
    ),
    "B30-FT-3": (
        "tfg_models/B30/B30-FT-3/seed=1/checkpoints/"
        "B30_B30-FT-3_seed1_best_val_adv_acc.pth"
    ),
    "B30-FT-E": (
        "tfg_models/B30/B30-FT-E/seed=1/checkpoints/"
        "B30_B30-FT-E_seed1_best_val_adv_acc.pth"
    ),
    "B30-FT-E-M": (
        "tfg_models/B30/B30-FT-E-M/seed=1/checkpoints/"
        "B30_B30-FT-E_seed1_best_val_adv_acc.pth"
    ),
}

CURVE_COLUMNS = [
    "model_key",
    "variant_name",
    "mode",
    "beta",
    "epsilon",
    "accuracy",
    "mean_ce_loss",
    "num_samples",
    "mean_m_pred",
    "frac_m_pred_positive",
    "mean_m_proto",
    "frac_m_proto_positive",
    "mean_R_enc",
    "mean_R_dec",
    "mean_m_proto_used",
    "frac_m_proto_used_positive",
    "mean_m_proto_mu",
    "frac_m_proto_mu_positive",
    "mean_sigma_mean_all",
    "mean_sigma_median_all",
    "mean_sigma_min_correct_mu",
    "mean_sigma_min_wrong_mu",
    "mean_mu_mae_clamped_sample",
    "mean_target_saturation_rate_sample",
    "apmsr",
    "apmsr_false_match_count",
    "apmsr_suppressed_count",
    "false_match_rate_mu",
    "suppressed_rate_all",
    "risk_harm_count",
    "risk_harm_rate_all",
    "risk_harm_rate_on_mu_correct",
]

METRIC_NAMES = [
    "m_pred",
    "m_proto",
    "R_enc",
    "R_dec",
    "m_proto_used",
    "m_proto_mu",
    "sigma_mean_all",
    "sigma_median_all",
    "sigma_min_correct_mu",
    "sigma_min_wrong_mu",
    "mu_mae_clamped_sample",
    "target_saturation_rate_sample",
    "false_match_mu",
    "suppressed_match",
    "risk_harm_match",
]

DFL_FLOAT_METRICS = [
    "m_proto_mu",
    "sigma_mean_all",
    "sigma_median_all",
    "sigma_min_correct_mu",
    "sigma_min_wrong_mu",
    "mu_mae_clamped_sample",
    "target_saturation_rate_sample",
]

APMSR_BOOL_METRICS = ["false_match_mu", "suppressed_match", "risk_harm_match"]


@dataclass
class EvalConfig:
    models: List[str]
    seed: int
    data_dir: str
    batch_size: int
    num_workers: int
    device: str
    dfl_head_root: str
    dfl_run_tag: str
    dfl_head: Optional[List[str]]
    allow_random_dfl_head: bool
    betas: List[float]
    clean_only: bool
    attack: str
    max_eps: float
    step: float
    pgd_iters: int
    pgd_alpha: float
    random_start: bool
    max_batches: Optional[int]
    out_root: str
    run_id: str
    save_per_example: bool
    rpgd_binary_steps: int
    rpgd_binary_tol: Optional[float]
    make_plots: bool


@dataclass
class DFLHeadInfo:
    layer: DistributionalPrototypeDistanceLayer
    checkpoint_path: Optional[str]
    latent_dim: int
    num_bins: int
    d_min: float
    d_max: float
    hidden_dim: int
    random: bool


@dataclass
class EvalVariant:
    model_key: str
    variant_name: str
    mode: str
    beta: Optional[float]
    model: RiskAwareB30Wrapper
    dfl_info: Optional[DFLHeadInfo]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_attack_seed(seed: int) -> None:
    seed = int(seed) % (2**31 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stable_model_hash(model_key: str) -> int:
    digest = hashlib.md5(model_key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def attack_seed(cfg: EvalConfig, model_key: str, eps_idx: int, batch_idx: int) -> int:
    return cfg.seed * 1_000_003 + stable_model_hash(model_key) + eps_idx * 10_007 + batch_idx


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=True)


def save_curve_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CURVE_COLUMNS)
        writer.writeheader()
        writer.writerows([{key: row.get(key, float("nan")) for key in CURVE_COLUMNS} for row in rows])


def resolve_default_dfl_head_root() -> str:
    return str(REPO_DFL_HEAD_ROOT)


def resolve_default_out_root() -> str:
    if Path("/kaggle/working").exists():
        return "/kaggle/working/dfs_dfl_metrics"
    if Path("/content").exists():
        return "/content/dfs_dfl_metrics"
    return "dfs/results/dfl_metrics"


def resolve_repo_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def ensure_load_path_is_in_repo(path: Path, label: str) -> None:
    try:
        path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{label} must be loaded from inside the repository. Got: {path}"
        ) from exc


def torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def extract_state_dict(checkpoint: Any) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model_state"):
            if key in checkpoint:
                return checkpoint[key]
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    return None


def load_full_or_state_model(path: Path, device: torch.device) -> nn.Module:
    if not path.exists():
        raise FileNotFoundError(f"B30 model checkpoint path is missing: {path}")

    checkpoint = torch_load(path, device)
    if isinstance(checkpoint, nn.Module):
        return checkpoint.to(device)

    state_dict = extract_state_dict(checkpoint)
    if state_dict is None:
        observed = list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)
        raise TypeError(f"Unsupported B30 checkpoint format at {path}. Observed: {observed}")

    model = CAEModel_Balanced().to(device)
    state_dict = strip_module_prefix(state_dict)
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as exc:
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        missing = sorted(model_keys - ckpt_keys)
        unexpected = sorted(ckpt_keys - model_keys)
        raise RuntimeError(
            "Failed to load B30 state dict into CAEModel_Balanced.\n"
            f"Missing keys ({len(missing)}): {missing[:30]}\n"
            f"Unexpected keys ({len(unexpected)}): {unexpected[:30]}"
        ) from exc
    return model


def load_b30_model(model_key: str, device: torch.device) -> nn.Module:
    if model_key not in B30_PATHS:
        raise ValueError(f"Unknown B30 model key {model_key!r}. Known: {sorted(B30_PATHS)}")

    if model_key.startswith("B30-FT-"):
        base_path = resolve_repo_path(B30_PATHS["B30"])
        state_path = resolve_repo_path(B30_PATHS[model_key])
        if not state_path.exists():
            raise FileNotFoundError(f"B30 fine-tuned checkpoint path is missing: {state_path}")
        model = load_full_or_state_model(base_path, device)
        state = torch_load(state_path, device)
        state_dict = extract_state_dict(state)
        if state_dict is None:
            observed = list(state.keys()) if isinstance(state, dict) else type(state)
            raise TypeError(f"Unsupported B30-FT checkpoint format at {state_path}: {observed}")
        model.load_state_dict(strip_module_prefix(state_dict))
        print(f"Loaded {model_key}: base={base_path}, state={state_path}")
    else:
        model_path = resolve_repo_path(B30_PATHS[model_key])
        model = load_full_or_state_model(model_path, device)
        print(f"Loaded {model_key}: {model_path}")

    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model


def parse_dfl_head_overrides(items: Optional[List[str]]) -> Dict[str, Path]:
    overrides: Dict[str, Path] = {}
    if not items:
        return overrides
    for item in items:
        if ":" not in item:
            raise ValueError(f"Invalid --dfl-head override {item!r}. Expected MODEL_KEY:PATH.")
        model_key, path = item.split(":", 1)
        resolved = resolve_repo_path(path)
        ensure_load_path_is_in_repo(resolved, "DFL head override")
        overrides[model_key] = resolved
    return overrides


def default_dfl_head_path(cfg: EvalConfig, model_key: str) -> Path:
    root = resolve_repo_path(cfg.dfl_head_root)
    ensure_load_path_is_in_repo(root, "DFL head root")
    return root / model_key / f"seed={cfg.seed}_{cfg.dfl_run_tag}" / "checkpoints" / "best_dfl_head.pt"


def freeze_dfl_head(layer: DistributionalPrototypeDistanceLayer) -> None:
    layer.eval()
    for parameter in layer.parameters():
        parameter.requires_grad = False


def load_dfl_head(
    model_key: str,
    base_model: nn.Module,
    cfg: EvalConfig,
    overrides: Dict[str, Path],
    device: torch.device,
) -> DFLHeadInfo:
    head_path = overrides.get(model_key, default_dfl_head_path(cfg, model_key))
    if not head_path.exists():
        if not cfg.allow_random_dfl_head:
            raise FileNotFoundError(
                "Missing DFL head checkpoint. Attempted path:\n"
                f"{head_path}\n"
                "Pass --dfl-head MODEL_KEY:PATH to override, or "
                "--allow-random-dfl-head for smoke testing only."
            )
        print("\nWARNING: using a random DFL head for smoke testing only.")
        print(f"Missing checkpoint was: {head_path}\n")
        layer = DistributionalPrototypeDistanceLayer(
            latent_dim=int(base_model.in_channels_prototype),
            num_bins=32,
            d_min=0.0,
            d_max=12.0,
            hidden_dim=64,
        ).to(device)
        freeze_dfl_head(layer)
        return DFLHeadInfo(
            layer=layer,
            checkpoint_path=None,
            latent_dim=int(base_model.in_channels_prototype),
            num_bins=32,
            d_min=0.0,
            d_max=12.0,
            hidden_dim=64,
            random=True,
        )

    checkpoint = torch_load(head_path, device)
    if not isinstance(checkpoint, dict) or "dfl_layer_state" not in checkpoint:
        observed = list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)
        raise TypeError(f"DFL checkpoint {head_path} is missing dfl_layer_state. Observed: {observed}")

    latent_dim = int(checkpoint.get("latent_dim", base_model.in_channels_prototype))
    num_bins = int(checkpoint["num_bins"])
    d_min = float(checkpoint["d_min"])
    d_max = float(checkpoint["d_max"])
    hidden_dim = int(checkpoint["hidden_dim"])
    if latent_dim != int(base_model.in_channels_prototype):
        raise ValueError(
            f"DFL latent_dim={latent_dim} does not match base model "
            f"in_channels_prototype={base_model.in_channels_prototype}."
        )

    layer = DistributionalPrototypeDistanceLayer(
        latent_dim=latent_dim,
        num_bins=num_bins,
        d_min=d_min,
        d_max=d_max,
        hidden_dim=hidden_dim,
    ).to(device)
    layer.load_state_dict(checkpoint["dfl_layer_state"])
    freeze_dfl_head(layer)
    print(f"Loaded DFL head for {model_key}: {head_path}")
    return DFLHeadInfo(
        layer=layer,
        checkpoint_path=str(head_path),
        latent_dim=latent_dim,
        num_bins=num_bins,
        d_min=d_min,
        d_max=d_max,
        hidden_dim=hidden_dim,
        random=False,
    )


def beta_label(beta: float) -> str:
    text = f"{beta:g}".replace("-", "m").replace(".", "p")
    return text


def make_variants(
    model_key: str,
    base_model: nn.Module,
    dfl_info: DFLHeadInfo,
    betas: List[float],
    device: torch.device,
) -> List[EvalVariant]:
    variants = [
        EvalVariant(
            model_key=model_key,
            variant_name=f"{model_key}__det",
            mode="deterministic",
            beta=None,
            model=RiskAwareB30Wrapper(base_model, mode="deterministic").to(device).eval(),
            dfl_info=None,
        ),
        EvalVariant(
            model_key=model_key,
            variant_name=f"{model_key}__dfl_mean",
            mode="mean",
            beta=0.0,
            model=RiskAwareB30Wrapper(base_model, mode="mean", beta=0.0, dfl_layer=dfl_info.layer).to(device).eval(),
            dfl_info=dfl_info,
        ),
    ]
    for beta in betas:
        variants.append(
            EvalVariant(
                model_key=model_key,
                variant_name=f"{model_key}__dfl_risk_beta{beta_label(beta)}",
                mode="risk",
                beta=beta,
                model=RiskAwareB30Wrapper(base_model, mode="risk", beta=beta, dfl_layer=dfl_info.layer).to(device).eval(),
                dfl_info=dfl_info,
            )
        )
    for variant in variants:
        variant.model.eval()
        for parameter in variant.model.parameters():
            parameter.requires_grad = False
    return variants


def build_epsilon_grid(clean_only: bool, max_eps: float, step: float) -> np.ndarray:
    if clean_only:
        return np.asarray([0.0], dtype=np.float32)
    if step <= 0:
        raise ValueError("--step must be > 0.")
    values = [0.0]
    current = step
    while current <= max_eps + step * 1e-6:
        values.append(round(current, 10))
        current += step
    if values[-1] < max_eps - step * 1e-6:
        values.append(float(max_eps))
    return np.asarray(values, dtype=np.float32)


def make_empty_metric_store(num_eps: int) -> Dict[str, List[List[np.ndarray]]]:
    return {name: [[] for _ in range(num_eps)] for name in METRIC_NAMES}


def append_metric(store: Dict[str, List[List[np.ndarray]]], name: str, eps_idx: int, values: np.ndarray) -> None:
    store[name][eps_idx].append(np.asarray(values))


def nan_array(batch_size: int) -> np.ndarray:
    return np.full(batch_size, np.nan, dtype=np.float32)


def zeros_uint8(batch_size: int) -> np.ndarray:
    return np.zeros(batch_size, dtype=np.uint8)


def tensor_to_numpy(values: torch.Tensor, dtype=np.float32) -> np.ndarray:
    return values.detach().cpu().numpy().astype(dtype)


def finite_mean(values: np.ndarray) -> float:
    values = np.asarray(values)
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanmean(values))


def positive_fraction(values: np.ndarray) -> float:
    values = np.asarray(values)
    mask = ~np.isnan(values)
    if not mask.any():
        return float("nan")
    return float((values[mask] > 0).mean())


def legacy_clean_or_adv_metrics(
    model: nn.Module,
    x_eval: torch.Tensor,
    y: torch.Tensor,
    proto_labels: torch.Tensor,
    clean_int: Dict[str, torch.Tensor],
    epsilon: float,
) -> Dict[str, np.ndarray]:
    if epsilon == 0.0:
        eval_int = clean_int
        batch_size = y.shape[0]
        r_enc = np.zeros(batch_size, dtype=np.float32)
        r_dec = np.zeros(batch_size, dtype=np.float32)
    else:
        eval_int = extract_internals(model, x_eval)
        r_enc = calc_R_enc(clean_int["z"], eval_int["z"])
        r_dec = calc_R_dec(clean_int["x_hat"], eval_int["x_hat"])

    return {
        "logits": eval_int["logits"],
        "m_pred": calc_m_pred(eval_int["logits"], y),
        "m_proto": calc_m_proto(eval_int["d_proto"], y, proto_labels),
        "R_enc": r_enc,
        "R_dec": r_dec,
    }


def nearest_sigma_under_mu(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor,
    proto_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = proto_labels.to(device=mu.device, dtype=y.dtype)
    y = y.to(mu.device)
    correct_mask = labels.unsqueeze(0).eq(y.unsqueeze(1))
    wrong_mask = ~correct_mask
    if not correct_mask.any(dim=1).all():
        raise ValueError("At least one sample has no correct-class prototype.")
    if not wrong_mask.any(dim=1).all():
        raise ValueError("At least one sample has no wrong-class prototype.")
    inf = torch.tensor(float("inf"), device=mu.device, dtype=mu.dtype)
    correct_idx = torch.where(correct_mask, mu, inf).argmin(dim=1)
    wrong_idx = torch.where(wrong_mask, mu, inf).argmin(dim=1)
    row_idx = torch.arange(mu.shape[0], device=mu.device)
    return sigma[row_idx, correct_idx], sigma[row_idx, wrong_idx]


def append_dfl_metrics(
    store: Dict[str, List[List[np.ndarray]]],
    eps_idx: int,
    variant: EvalVariant,
    x_eval: torch.Tensor,
    y: torch.Tensor,
    proto_labels: torch.Tensor,
    batch_size: int,
) -> None:
    with torch.no_grad():
        outputs = variant.model.compute_distance_outputs(x_eval)
    d_used = outputs["d_used"]
    m_proto_used = calc_m_proto_from_distances(d_used, y, proto_labels)
    append_metric(store, "m_proto_used", eps_idx, tensor_to_numpy(m_proto_used))

    if variant.mode == "deterministic" or variant.dfl_info is None:
        for name in DFL_FLOAT_METRICS:
            append_metric(store, name, eps_idx, nan_array(batch_size))
        for name in APMSR_BOOL_METRICS:
            append_metric(store, name, eps_idx, zeros_uint8(batch_size))
        return

    d_det = outputs["d_det"]
    mu = outputs["mu"]
    sigma = outputs["sigma"]
    d_det_clamped = torch.clamp(d_det, min=variant.dfl_info.d_min, max=variant.dfl_info.d_max)
    m_proto_mu = calc_m_proto_from_distances(mu, y, proto_labels)
    sigma_min_correct_mu, sigma_min_wrong_mu = nearest_sigma_under_mu(mu, sigma, y, proto_labels)
    apmsr = calc_artificial_prototype_match_suppression(mu, d_used, y, proto_labels)

    append_metric(store, "m_proto_mu", eps_idx, tensor_to_numpy(m_proto_mu))
    append_metric(store, "sigma_mean_all", eps_idx, tensor_to_numpy(sigma.mean(dim=1)))
    append_metric(store, "sigma_median_all", eps_idx, tensor_to_numpy(sigma.median(dim=1).values))
    append_metric(store, "sigma_min_correct_mu", eps_idx, tensor_to_numpy(sigma_min_correct_mu))
    append_metric(store, "sigma_min_wrong_mu", eps_idx, tensor_to_numpy(sigma_min_wrong_mu))
    append_metric(store, "mu_mae_clamped_sample", eps_idx, tensor_to_numpy((mu - d_det_clamped).abs().mean(dim=1)))
    append_metric(store, "target_saturation_rate_sample", eps_idx, tensor_to_numpy((d_det > variant.dfl_info.d_max).float().mean(dim=1)))
    append_metric(store, "false_match_mu", eps_idx, tensor_to_numpy(apmsr["false_match_mu"], dtype=np.uint8))
    append_metric(store, "suppressed_match", eps_idx, tensor_to_numpy(apmsr["suppressed_match"], dtype=np.uint8))
    append_metric(store, "risk_harm_match", eps_idx, tensor_to_numpy(apmsr["risk_harm_match"], dtype=np.uint8))


def attack_batch(
    variant: EvalVariant,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    cfg: EvalConfig,
    eps_idx: int,
    batch_idx: int,
) -> torch.Tensor:
    if cfg.attack != "PGDLInf_attack":
        raise ValueError("Only PGDLInf_attack is supported in this DFS evaluator.")
    set_attack_seed(attack_seed(cfg, variant.model_key, eps_idx, batch_idx))

    def loss_f(batch_x: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(variant.model(batch_x), y)

    return PGDLInf_attack(
        x,
        loss_f,
        iters=cfg.pgd_iters,
        eps=epsilon,
        alpha=cfg.pgd_alpha,
        random_start=cfg.random_start,
    )


def collect_variant_metrics(
    variant: EvalVariant,
    loader: torch.utils.data.DataLoader,
    eps_grid: np.ndarray,
    cfg: EvalConfig,
) -> tuple[np.ndarray, Dict[str, np.ndarray], List[Dict[str, Any]]]:
    num_eps = len(eps_grid)
    store = make_empty_metric_store(num_eps)
    correct_store: List[List[np.ndarray]] = [[] for _ in range(num_eps)]
    ce_loss_store: List[List[np.ndarray]] = [[] for _ in range(num_eps)]
    proto_labels = get_proto_labels(variant.model).to(cfg.device)

    for batch_idx, (x, y) in enumerate(tqdm(loader, desc=variant.variant_name, leave=False)):
        if cfg.max_batches is not None and batch_idx >= cfg.max_batches:
            break

        x = x.to(cfg.device)
        y = y.to(cfg.device)
        clean_int = extract_internals(variant.model, x)

        for eps_idx, epsilon_value in enumerate(eps_grid):
            epsilon = float(epsilon_value)
            x_eval = x if epsilon == 0.0 else attack_batch(variant, x, y, epsilon, cfg, eps_idx, batch_idx).detach()

            legacy = legacy_clean_or_adv_metrics(
                model=variant.model,
                x_eval=x_eval,
                y=y,
                proto_labels=proto_labels,
                clean_int=clean_int,
                epsilon=epsilon,
            )
            logits = legacy["logits"]
            correct = (logits.argmax(dim=1) == y).detach().cpu().numpy().astype(np.uint8)
            ce_loss = F.cross_entropy(logits, y, reduction="none").detach().cpu().numpy().astype(np.float32)

            correct_store[eps_idx].append(correct)
            ce_loss_store[eps_idx].append(ce_loss)
            append_metric(store, "m_pred", eps_idx, legacy["m_pred"])
            append_metric(store, "m_proto", eps_idx, legacy["m_proto"])
            append_metric(store, "R_enc", eps_idx, legacy["R_enc"])
            append_metric(store, "R_dec", eps_idx, legacy["R_dec"])
            append_dfl_metrics(store, eps_idx, variant, x_eval, y, proto_labels, batch_size=x.shape[0])

    if not correct_store[0]:
        raise RuntimeError("No evaluation batches were processed.")

    correct_matrix = np.stack([np.concatenate(correct_store[i], axis=0) for i in range(num_eps)], axis=1)
    ce_matrix = np.stack([np.concatenate(ce_loss_store[i], axis=0) for i in range(num_eps)], axis=1)
    metric_matrices = {
        name: np.stack([np.concatenate(store[name][i], axis=0) for i in range(num_eps)], axis=1)
        for name in METRIC_NAMES
    }

    rows = build_curve_rows(variant, eps_grid, correct_matrix, ce_matrix, metric_matrices)
    return correct_matrix, metric_matrices, rows


def aggregate_apmsr(false_match: np.ndarray, suppressed: np.ndarray, risk_harm: np.ndarray) -> Dict[str, float]:
    false_count = int(false_match.sum())
    suppressed_count = int(suppressed.sum())
    risk_harm_count = int(risk_harm.sum())
    n = int(false_match.shape[0])
    mu_correct_count = n - false_count
    return {
        "apmsr": float(suppressed_count / false_count) if false_count > 0 else float("nan"),
        "apmsr_false_match_count": false_count,
        "apmsr_suppressed_count": suppressed_count,
        "false_match_rate_mu": float(false_count / n) if n > 0 else float("nan"),
        "suppressed_rate_all": float(suppressed_count / n) if n > 0 else float("nan"),
        "risk_harm_count": risk_harm_count,
        "risk_harm_rate_all": float(risk_harm_count / n) if n > 0 else float("nan"),
        "risk_harm_rate_on_mu_correct": float(risk_harm_count / mu_correct_count) if mu_correct_count > 0 else float("nan"),
    }


def build_curve_rows(
    variant: EvalVariant,
    eps_grid: np.ndarray,
    correct_matrix: np.ndarray,
    ce_matrix: np.ndarray,
    metrics: Dict[str, np.ndarray],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    num_eps = len(eps_grid)
    for eps_idx in range(num_eps):
        m_pred = metrics["m_pred"][:, eps_idx]
        m_proto = metrics["m_proto"][:, eps_idx]
        m_proto_used = metrics["m_proto_used"][:, eps_idx]
        m_proto_mu = metrics["m_proto_mu"][:, eps_idx]
        apmsr = aggregate_apmsr(
            metrics["false_match_mu"][:, eps_idx].astype(bool),
            metrics["suppressed_match"][:, eps_idx].astype(bool),
            metrics["risk_harm_match"][:, eps_idx].astype(bool),
        )
        row = {
            "model_key": variant.model_key,
            "variant_name": variant.variant_name,
            "mode": variant.mode,
            "beta": "" if variant.beta is None else variant.beta,
            "epsilon": float(eps_grid[eps_idx]),
            "accuracy": float(correct_matrix[:, eps_idx].mean()),
            "mean_ce_loss": float(ce_matrix[:, eps_idx].mean()),
            "num_samples": int(correct_matrix.shape[0]),
            "mean_m_pred": finite_mean(m_pred),
            "frac_m_pred_positive": positive_fraction(m_pred),
            "mean_m_proto": finite_mean(m_proto),
            "frac_m_proto_positive": positive_fraction(m_proto),
            "mean_R_enc": finite_mean(metrics["R_enc"][:, eps_idx]),
            "mean_R_dec": finite_mean(metrics["R_dec"][:, eps_idx]),
            "mean_m_proto_used": finite_mean(m_proto_used),
            "frac_m_proto_used_positive": positive_fraction(m_proto_used),
            "mean_m_proto_mu": finite_mean(m_proto_mu),
            "frac_m_proto_mu_positive": positive_fraction(m_proto_mu),
            "mean_sigma_mean_all": finite_mean(metrics["sigma_mean_all"][:, eps_idx]),
            "mean_sigma_median_all": finite_mean(metrics["sigma_median_all"][:, eps_idx]),
            "mean_sigma_min_correct_mu": finite_mean(metrics["sigma_min_correct_mu"][:, eps_idx]),
            "mean_sigma_min_wrong_mu": finite_mean(metrics["sigma_min_wrong_mu"][:, eps_idx]),
            "mean_mu_mae_clamped_sample": finite_mean(metrics["mu_mae_clamped_sample"][:, eps_idx]),
            "mean_target_saturation_rate_sample": finite_mean(metrics["target_saturation_rate_sample"][:, eps_idx]),
            **apmsr,
        }
        if variant.mode == "deterministic":
            row["apmsr"] = float("nan")
        rows.append(row)
    return rows


def relative_degradation_curve(accuracy_curve: List[float]) -> List[float]:
    clean = accuracy_curve[0] if accuracy_curve else float("nan")
    if clean == 0 or np.isnan(clean):
        return [float("nan") for _ in accuracy_curve]
    return [float((clean - acc) / clean) for acc in accuracy_curve]


def monotonicity_violations(accuracy_curve: List[float]) -> int:
    return int(sum(accuracy_curve[i] > accuracy_curve[i - 1] + 1e-12 for i in range(1, len(accuracy_curve))))


def mean_metric_curves(metrics: Dict[str, np.ndarray]) -> Dict[str, List[float]]:
    return {name: [finite_mean(matrix[:, i]) for i in range(matrix.shape[1])] for name, matrix in metrics.items()}


def save_variant_outputs(
    variant: EvalVariant,
    eps_grid: np.ndarray,
    correct_matrix: np.ndarray,
    metrics: Dict[str, np.ndarray],
    rows: List[Dict[str, Any]],
    cfg: EvalConfig,
    variant_dir: Path,
    loader: torch.utils.data.DataLoader,
) -> Dict[str, Any]:
    ensure_dir(variant_dir)
    save_curve_csv(variant_dir / "curve.csv", rows)
    np.savez_compressed(
        variant_dir / "per_example_correct.npz",
        correct=correct_matrix.astype(np.uint8),
        eps=eps_grid.astype(np.float32),
    )
    metrics_to_save = {"eps": eps_grid.astype(np.float32)}
    for name, matrix in metrics.items():
        if name in APMSR_BOOL_METRICS:
            metrics_to_save[name] = matrix.astype(np.uint8)
        else:
            metrics_to_save[name] = matrix.astype(np.float32)
    np.savez_compressed(variant_dir / "per_example_metrics.npz", **metrics_to_save)

    accuracy_curve = [float(row["accuracy"]) for row in rows]
    clean_correct_mask = correct_matrix[:, 0].astype(bool)
    robustness = compute_empirical_robustness_interval(correct_matrix.astype(bool), eps_grid.astype(np.float64))
    icr_legacy = compute_icr(metrics["m_proto"], metrics["m_pred"], clean_correct_mask=clean_correct_mask)
    icr_used = compute_icr(metrics["m_proto_used"], metrics["m_pred"], clean_correct_mask=clean_correct_mask)
    icr_mu = compute_icr(metrics["m_proto_mu"], metrics["m_pred"], clean_correct_mask=clean_correct_mask)

    apmsr_curves = {
        "apmsr_curve": [row["apmsr"] for row in rows],
        "false_match_count_curve": [row["apmsr_false_match_count"] for row in rows],
        "suppressed_count_curve": [row["apmsr_suppressed_count"] for row in rows],
        "false_match_rate_mu_curve": [row["false_match_rate_mu"] for row in rows],
        "suppressed_rate_all_curve": [row["suppressed_rate_all"] for row in rows],
        "risk_harm_count_curve": [row["risk_harm_count"] for row in rows],
        "risk_harm_rate_all_curve": [row["risk_harm_rate_all"] for row in rows],
        "risk_harm_rate_on_mu_correct_curve": [row["risk_harm_rate_on_mu_correct"] for row in rows],
    }

    robustness_json = {
        key: value
        for key, value in robustness.items()
        if key not in {"r_minus", "r_plus"}
    }
    robustness_json["r_minus"] = robustness["r_minus"].tolist()
    robustness_json["r_plus"] = robustness["r_plus"].tolist()

    rpgd_summary = None
    if cfg.rpgd_binary_steps > 0 and cfg.attack == "PGDLInf_attack" and len(eps_grid) > 1:
        rpgd_result = compute_r_pgd_binary_search(
            model=variant.model,
            data_loader=loader,
            eps_grid=eps_grid.astype(np.float64),
            coarse_correct=correct_matrix.astype(bool),
            pgd_iters=cfg.pgd_iters,
            pgd_alpha=cfg.pgd_alpha,
            pgd_random_start=cfg.random_start,
            binary_steps=cfg.rpgd_binary_steps,
            tol=cfg.rpgd_binary_tol,
            seed=cfg.seed,
            verbose=False,
        )
        np.savez_compressed(
            variant_dir / "r_pgd_binary_search.npz",
            r_pgd=rpgd_result["r_pgd"],
            r_pgd_upper=rpgd_result["r_pgd_upper"],
            failing_mask=rpgd_result["failing_mask"],
            clean_correct_mask=rpgd_result["clean_correct_mask"],
            eps_grid=rpgd_result["eps_grid"],
        )
        rpgd_summary = {
            key: value
            for key, value in rpgd_result.items()
            if key not in {"r_pgd", "r_pgd_upper", "failing_mask", "clean_correct_mask", "eps_grid"}
        }

    metrics_json = {
        "model_key": variant.model_key,
        "variant_name": variant.variant_name,
        "mode": variant.mode,
        "beta": variant.beta,
        "attack_name": cfg.attack,
        "seed": cfg.seed,
        "run_id": cfg.run_id,
        "dataset": DATASET_NAME,
        "threat": THREAT_NAME,
        "dfl_head_checkpoint": None if variant.dfl_info is None else variant.dfl_info.checkpoint_path,
        "dfl_d_min": None if variant.dfl_info is None else variant.dfl_info.d_min,
        "dfl_d_max": None if variant.dfl_info is None else variant.dfl_info.d_max,
        "dfl_num_bins": None if variant.dfl_info is None else variant.dfl_info.num_bins,
        "pgd_parameters": {
            "pgd_iters": cfg.pgd_iters,
            "pgd_alpha": cfg.pgd_alpha,
            "random_start": cfg.random_start,
            "max_eps": cfg.max_eps,
            "step": cfg.step,
        },
        "num_samples": int(correct_matrix.shape[0]),
        "epsilon_grid": eps_grid.astype(float).tolist(),
        "clean_accuracy": accuracy_curve[0],
        "accuracy_curve": accuracy_curve,
        "relative_degradation_curve": relative_degradation_curve(accuracy_curve),
        "monotonicity_violations": monotonicity_violations(accuracy_curve),
        "robustness": robustness_json,
        "r_pgd_binary_search": rpgd_summary,
        "ICR_legacy": icr_legacy,
        "ICR_used": icr_used,
        "ICR_mu": icr_mu,
        "APMSR": apmsr_curves,
        "mean_internal_metrics": mean_metric_curves(metrics),
        "notes": [
            "legacy m_proto uses original deterministic prototype geometry and is comparable with previous chapters.",
            "m_proto_used uses the actual decision distance passed to the classifier.",
            "APMSR is meaningful only when false_match_count is non-zero.",
        ],
    }
    save_json(variant_dir / "metrics.json", metrics_json)
    if cfg.make_plots:
        make_variant_plots(variant_dir, eps_grid, rows)
    return metrics_json


def make_variant_plots(variant_dir: Path, eps_grid: np.ndarray, rows: List[Dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save_plot(filename: str, y_key: str, ylabel: str) -> None:
        y = [row[y_key] for row in rows]
        if all(np.isnan(value) for value in y):
            return
        plt.figure()
        plt.plot(eps_grid, y, marker="o")
        plt.xlabel("epsilon")
        plt.ylabel(ylabel)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(variant_dir / filename, dpi=160)
        plt.close()

    save_plot("accuracy_plot.png", "accuracy", "accuracy")
    save_plot("m_proto_plot.png", "mean_m_proto", "mean m_proto")
    save_plot("m_proto_used_plot.png", "mean_m_proto_used", "mean m_proto_used")
    save_plot("apmsr_plot.png", "apmsr", "APMSR")


def variant_output_dir(cfg: EvalConfig, variant: EvalVariant) -> Path:
    return (
        resolve_repo_path(cfg.out_root)
        / DATASET_NAME
        / THREAT_NAME
        / cfg.attack
        / variant.model_key
        / variant.variant_name
        / f"seed={cfg.seed}"
        / f"run_id={cfg.run_id}"
    )


def index_output_dir(cfg: EvalConfig) -> Path:
    return resolve_repo_path(cfg.out_root) / DATASET_NAME / THREAT_NAME / cfg.attack / "_index" / f"run_id={cfg.run_id}"


def run_clean_sanity_check(
    model_key: str,
    variants: List[EvalVariant],
    loader: torch.utils.data.DataLoader,
    cfg: EvalConfig,
) -> None:
    first_x, _ = next(iter(loader))
    first_x = first_x.to(cfg.device)
    mean_variant = next(variant for variant in variants if variant.mode == "mean")
    risk0 = [
        variant
        for variant in variants
        if variant.mode == "risk" and variant.beta is not None and abs(variant.beta) < 1e-12
    ]
    if not risk0:
        return
    with torch.no_grad():
        mean_logits = mean_variant.model(first_x)
        risk_logits = risk0[0].model(first_x)
    diff = (mean_logits - risk_logits).abs().max().item()
    print(f"{model_key}: mean vs risk beta=0 first-batch max_abs_diff={diff:.6g}")
    if not cfg.allow_random_dfl_head:
        assert diff < 1e-5, "Risk beta=0 logits should match Mean logits."


def parse_args() -> argparse.Namespace:
    default_run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Run structured DFS/DFL metrics.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dfl-head-root", default=resolve_default_dfl_head_root())
    parser.add_argument("--dfl-run-tag", default="dmax12_bins32")
    parser.add_argument("--dfl-head", action="append", default=None)
    parser.add_argument("--allow-random-dfl-head", action="store_true")
    parser.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.25, 0.5, 1.0, 2.0])
    parser.add_argument("--clean-only", action="store_true")
    parser.add_argument("--attack", default="PGDLInf_attack")
    parser.add_argument("--max-eps", type=float, default=0.3)
    parser.add_argument("--step", type=float, default=0.025)
    parser.add_argument("--pgd-iters", type=int, default=80)
    parser.add_argument("--pgd-alpha", type=float, default=0.01)
    parser.add_argument("--random-start", dest="random_start", action="store_true", default=True)
    parser.add_argument("--no-random-start", dest="random_start", action="store_false")
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--out-root", default=resolve_default_out_root())
    parser.add_argument("--run-id", default=default_run_id)
    parser.add_argument("--save-per-example", action="store_true")
    parser.add_argument("--rpgd-binary-steps", type=int, default=0)
    parser.add_argument("--rpgd-binary-tol", type=float, default=None)
    parser.add_argument("--make-plots", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> EvalConfig:
    return EvalConfig(
        models=args.models,
        seed=args.seed,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        dfl_head_root=args.dfl_head_root,
        dfl_run_tag=args.dfl_run_tag,
        dfl_head=args.dfl_head,
        allow_random_dfl_head=args.allow_random_dfl_head,
        betas=args.betas,
        clean_only=args.clean_only,
        attack=args.attack,
        max_eps=args.max_eps,
        step=args.step,
        pgd_iters=args.pgd_iters,
        pgd_alpha=args.pgd_alpha,
        random_start=args.random_start,
        max_batches=args.max_batches,
        out_root=args.out_root,
        run_id=args.run_id,
        save_per_example=args.save_per_example,
        rpgd_binary_steps=args.rpgd_binary_steps,
        rpgd_binary_tol=args.rpgd_binary_tol,
        make_plots=args.make_plots,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    set_seed(cfg.seed)

    if cfg.attack != "PGDLInf_attack":
        raise ValueError("Only PGDLInf_attack is supported in this DFS evaluator.")
    if not cfg.clean_only and torch.cuda.is_available() and torch.device(cfg.device).type != "cuda":
        raise ValueError(
            "PGDLInf_attack internally selects CUDA when CUDA is available. "
            "Use --device cuda for PGD or pass --clean-only for CPU clean evaluation."
        )

    device = torch.device(cfg.device)
    eps_grid = build_epsilon_grid(cfg.clean_only, cfg.max_eps, cfg.step)
    index_dir = index_output_dir(cfg)
    ensure_dir(index_dir)
    save_json(index_dir / "config.json", asdict(cfg))

    loader = get_test_loader(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    dfl_overrides = parse_dfl_head_overrides(cfg.dfl_head)

    all_rows: List[Dict[str, Any]] = []
    summary_variants: Dict[str, Any] = {}

    for model_idx, model_key in enumerate(cfg.models):
        del model_idx
        base_model = load_b30_model(model_key, device)
        dfl_info = load_dfl_head(model_key, base_model, cfg, dfl_overrides, device)
        variants = make_variants(model_key, base_model, dfl_info, cfg.betas, device)
        run_clean_sanity_check(model_key, variants, loader, cfg)

        for variant in variants:
            correct_matrix, metric_matrices, rows = collect_variant_metrics(
                variant=variant,
                loader=loader,
                eps_grid=eps_grid,
                cfg=cfg,
            )
            out_dir = variant_output_dir(cfg, variant)
            metrics_json = save_variant_outputs(
                variant=variant,
                eps_grid=eps_grid,
                correct_matrix=correct_matrix,
                metrics=metric_matrices,
                rows=rows,
                cfg=cfg,
                variant_dir=out_dir,
                loader=loader,
            )
            all_rows.extend(rows)
            summary_variants[variant.variant_name] = {
                "path": str(out_dir),
                "clean_accuracy": metrics_json["clean_accuracy"],
                "accuracy_curve": metrics_json["accuracy_curve"],
                "ICR_legacy": metrics_json["ICR_legacy"],
                "ICR_used": metrics_json["ICR_used"],
                "ICR_mu": metrics_json["ICR_mu"],
                "APMSR": metrics_json["APMSR"],
            }

    save_curve_csv(index_dir / "all_variants_curve.csv", all_rows)
    save_json(
        index_dir / "summary.json",
        {
            "run_id": cfg.run_id,
            "dataset": DATASET_NAME,
            "threat": THREAT_NAME,
            "attack": cfg.attack,
            "epsilon_grid": eps_grid.astype(float).tolist(),
            "num_rows": len(all_rows),
            "variants": summary_variants,
            "config": asdict(cfg),
        },
    )

    print("\nDFL metrics summary")
    print("variant, eps, acc, mean_m_proto, mean_m_proto_used, apmsr, risk_harm_rate")
    for row in all_rows:
        print(
            f"{row['variant_name']}, {row['epsilon']}, {row['accuracy']:.6f}, "
            f"{row['mean_m_proto']:.6g}, {row['mean_m_proto_used']:.6g}, "
            f"{row['apmsr']:.6g}, {row['risk_harm_rate_all']:.6g}"
        )
    print(f"\nSaved global index to {index_dir}")


if __name__ == "__main__":
    main()
