"""Train only the DFL head to approximate B30 squared prototype distances."""

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPO_DFL_HEAD_ROOT = REPO_ROOT / "dfs" / "results" / "dfl_heads"

from data_loader import get_train_val_loader  # noqa: E402
from adversarial_attacks import PGDLInf_attack  # noqa: E402
from modules import CAEModel_Balanced  # noqa: E402
from dfs.dfl_layer import DistributionalPrototypeDistanceLayer  # noqa: E402
from dfs.dfl_losses import DistributionFocalDistanceLoss  # noqa: E402
from dfs.risk_b30_wrapper import RiskAwareB30Wrapper  # noqa: E402


KNOWN_MODEL_PATHS = {
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


@dataclass
class TrainConfig:
    model_key: str
    checkpoint: Optional[str]
    seed: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    val_size: float
    early_stopping_patience: int
    data_dir: str
    save_root: str
    num_workers: int
    device: str
    num_bins: int
    d_min: float
    d_max: float
    hidden_dim: int
    lambda_dfl: float
    lambda_ce: float
    adv_train: bool
    adv_eps: float
    adv_alpha: float
    adv_steps: int
    adv_random_start: bool
    adv_beta: float
    lambda_clean_dfl: float
    lambda_clean_ce: float
    lambda_adv_dfl: float
    lambda_adv_ce: float
    use_b30_clstsep_loss: bool
    lambda_clean_b30_loss: float
    lambda_adv_b30_loss: float
    b30_lambda_class: float
    b30_lambda_ae: float
    b30_lambda_1: float
    b30_lambda_clus: float
    b30_lambda_sep: float
    init_dfl_head: Optional[str]
    init_run_tag: Optional[str]
    early_stop_monitor: str
    max_train_batches: Optional[int]
    max_val_batches: Optional[int]
    run_tag: str
    smoke_test: bool


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_history_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable


def save_trainable_parameter_names(model: nn.Module, path: str | Path) -> None:
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    with open(path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")


def resolve_default_save_root() -> str:
    if Path("/kaggle/working").exists():
        return "/kaggle/working/dfs_dfl_heads"
    if Path("/content").exists():
        return "/content/dfs_dfl_heads"
    return "dfs/results/dfl_heads"


def resolve_default_num_workers() -> int:
    if Path("/kaggle/working").exists():
        cpu_count = os.cpu_count() or 2
        return max(1, min(4, cpu_count - 1))
    return 0


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


def resolve_checkpoint_path(cfg: TrainConfig) -> Optional[Path]:
    checkpoint = cfg.checkpoint or KNOWN_MODEL_PATHS.get(cfg.model_key)
    if checkpoint is None:
        return None

    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path
    ensure_load_path_is_in_repo(checkpoint_path, "B30 checkpoint")
    return checkpoint_path


def load_b30_model(checkpoint_path: Optional[Path], device: torch.device) -> nn.Module:
    if checkpoint_path is None:
        print("No checkpoint provided; using randomly initialized CAEModel_Balanced.")
        return CAEModel_Balanced().to(device)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    checkpoint = torch_load(checkpoint_path, device)
    if isinstance(checkpoint, nn.Module):
        print(f"Loaded full B30 model object from {checkpoint_path}.")
        return checkpoint.to(device)

    state_dict = extract_state_dict(checkpoint)
    if state_dict is None:
        raise TypeError(
            "Unsupported checkpoint format. Expected full torch.nn.Module, raw "
            "state dict, or dict with 'state_dict', 'model_state_dict', or "
            "'model_state'."
        )

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
            "Failed to load checkpoint state dict into CAEModel_Balanced.\n"
            f"Missing keys ({len(missing)}): {missing[:30]}\n"
            f"Unexpected keys ({len(unexpected)}): {unexpected[:30]}"
        ) from exc

    print(f"Loaded B30 state dict from {checkpoint_path}.")
    return model


def freeze_base_model(base_model: nn.Module) -> None:
    for parameter in base_model.parameters():
        parameter.requires_grad = False


def pearson_corr_from_sums(
    n: int,
    sum_x: float,
    sum_y: float,
    sum_x2: float,
    sum_y2: float,
    sum_xy: float,
    eps: float = 1e-12,
) -> float:
    if n <= 1:
        return float("nan")
    cov = sum_xy - (sum_x * sum_y / n)
    var_x = sum_x2 - (sum_x * sum_x / n)
    var_y = sum_y2 - (sum_y * sum_y / n)
    denom = max(var_x, 0.0) * max(var_y, 0.0)
    if denom <= eps:
        return float("nan")
    return cov / (denom ** 0.5)


def update_corr_sums(
    sums: Dict[str, float],
    x: torch.Tensor,
    y: torch.Tensor,
) -> None:
    x64 = x.detach().reshape(-1).double()
    y64 = y.detach().reshape(-1).double()
    sums["n"] += int(x64.numel())
    sums["sum_x"] += x64.sum().item()
    sums["sum_y"] += y64.sum().item()
    sums["sum_x2"] += x64.pow(2).sum().item()
    sums["sum_y2"] += y64.pow(2).sum().item()
    sums["sum_xy"] += (x64 * y64).sum().item()


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> int:
    return int((logits.argmax(dim=1) == y).sum().item())


def safe_float_label(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def ensure_load_path_is_in_repo(path: Path, label: str) -> None:
    try:
        path.resolve().relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"{label} must be loaded from inside the repository. Got: {path}"
        ) from exc


def resolve_relative_to_repo(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    ensure_load_path_is_in_repo(resolved, "DFL head checkpoint")
    return resolved


def resolve_init_dfl_head_path(cfg: TrainConfig) -> Optional[Path]:
    if cfg.init_dfl_head is not None:
        return resolve_relative_to_repo(cfg.init_dfl_head)
    if cfg.init_run_tag is None:
        return None

    path = (
        REPO_DFL_HEAD_ROOT
        / cfg.model_key
        / f"seed={cfg.seed}_{cfg.init_run_tag}"
        / "checkpoints"
        / "best_dfl_head.pt"
    )
    ensure_load_path_is_in_repo(path, "Initial DFL head checkpoint")
    return path


def load_initial_dfl_head_if_requested(
    dfl_layer: DistributionalPrototypeDistanceLayer,
    cfg: TrainConfig,
    latent_dim: int,
    device: torch.device,
) -> Optional[Path]:
    init_path = resolve_init_dfl_head_path(cfg)
    if init_path is None:
        if cfg.adv_train:
            print(
                "WARNING: adversarial calibration is starting from a random DFL "
                "head. Normally this should start from a clean-trained DFL head."
            )
        return None

    if not init_path.exists():
        raise FileNotFoundError(f"Initial DFL head checkpoint does not exist: {init_path}")

    checkpoint = torch_load(init_path, device)
    if not isinstance(checkpoint, dict) or "dfl_layer_state" not in checkpoint:
        keys = list(checkpoint.keys()) if isinstance(checkpoint, dict) else type(checkpoint)
        raise TypeError(
            f"Initial DFL checkpoint has unsupported format at {init_path}. "
            f"Observed keys/type: {keys}"
        )

    expected = {
        "latent_dim": latent_dim,
        "num_bins": cfg.num_bins,
        "d_min": cfg.d_min,
        "d_max": cfg.d_max,
        "hidden_dim": cfg.hidden_dim,
    }
    actual = {
        "latent_dim": int(checkpoint.get("latent_dim", -1)),
        "num_bins": int(checkpoint.get("num_bins", -1)),
        "d_min": float(checkpoint.get("d_min", float("nan"))),
        "d_max": float(checkpoint.get("d_max", float("nan"))),
        "hidden_dim": int(checkpoint.get("hidden_dim", -1)),
    }
    mismatches = {
        key: (expected[key], actual[key])
        for key in expected
        if expected[key] != actual[key]
    }
    if mismatches:
        raise ValueError(
            f"Initial DFL head metadata does not match current config: {mismatches}"
        )

    dfl_layer.load_state_dict(checkpoint["dfl_layer_state"])
    print(f"Initialized DFL head from {init_path}")
    return init_path


def assert_no_base_gradients(base_model: nn.Module) -> None:
    bad = [
        name
        for name, parameter in base_model.named_parameters()
        if parameter.grad is not None
    ]
    if bad:
        base_model.zero_grad(set_to_none=True)
        raise RuntimeError(f"Frozen base model received gradients: {bad[:20]}")


def assert_dfl_has_finite_gradient(dfl_layer: nn.Module) -> None:
    has_grad = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in dfl_layer.parameters()
    )
    if not has_grad:
        raise RuntimeError("DFL layer did not receive a finite gradient.")


def generate_adv_batch_for_dfl_head(
    wrapper: RiskAwareB30Wrapper,
    dfl_layer: DistributionalPrototypeDistanceLayer,
    x_clean: torch.Tensor,
    y: torch.Tensor,
    cfg: TrainConfig,
) -> torch.Tensor:
    base_training = wrapper.training
    dfl_training = dfl_layer.training
    wrapper.eval()
    dfl_layer.eval()

    def loss_f(batch_x: torch.Tensor) -> torch.Tensor:
        logits = wrapper(batch_x)
        return F.cross_entropy(logits, y)

    x_adv = PGDLInf_attack(
        x_clean,
        loss_f,
        iters=cfg.adv_steps,
        eps=cfg.adv_eps,
        alpha=cfg.adv_alpha,
        random_start=cfg.adv_random_start,
    )

    wrapper.train(base_training)
    dfl_layer.train(dfl_training)
    return x_adv.detach()


def compute_dfl_branch_terms(
    wrapper: RiskAwareB30Wrapper,
    dfl_loss_fn: DistributionFocalDistanceLoss,
    x_input: torch.Tensor,
    y: torch.Tensor,
    cfg: TrainConfig,
    branch_name: str,
) -> Dict[str, torch.Tensor]:
    outputs = wrapper.compute_distance_outputs(x_input)
    d_det = outputs["d_det"]
    mu = outputs["mu"]
    sigma = outputs["sigma"]
    bin_logits = outputs["bin_logits"]
    logits = outputs["logits"]

    loss_dfl = dfl_loss_fn(bin_logits, d_det.detach())
    loss_ce = F.cross_entropy(logits, y)
    zero = loss_dfl.new_zeros(())

    terms: Dict[str, torch.Tensor] = {
        "logits": logits,
        "d_det": d_det,
        "mu": mu,
        "sigma": sigma,
        "bin_logits": bin_logits,
        "loss_dfl": loss_dfl,
        "loss_ce": loss_ce,
        "b30_full_loss": zero,
        "b30_ce": zero,
        "b30_e1": zero,
        "b30_clst": zero,
        "b30_sep": zero,
        "b30_recon": zero,
    }

    if cfg.use_b30_clstsep_loss:
        try:
            from loss_functions import ClstSepLoss
        except Exception as exc:  # pragma: no cover - import failure should be explicit
            raise RuntimeError("Failed to import ClstSepLoss from loss_functions.py") from exc

        try:
            (
                b30_full,
                b30_ce,
                b30_e1,
                b30_clst,
                b30_sep,
                b30_recon,
            ) = ClstSepLoss(
                wrapper,
                x_input,
                y,
                logits,
                cfg.b30_lambda_class,
                cfg.b30_lambda_1,
                cfg.b30_lambda_clus,
                cfg.b30_lambda_sep,
                cfg.b30_lambda_ae,
            )
        except Exception as exc:
            raise RuntimeError(
                "ClstSepLoss is not compatible with the current DFL wrapper call."
            ) from exc

        terms.update(
            {
                "b30_full_loss": b30_full,
                "b30_ce": b30_ce,
                "b30_e1": b30_e1,
                "b30_clst": b30_clst,
                "b30_sep": b30_sep,
                "b30_recon": b30_recon,
            }
        )

    if branch_name == "clean":
        total = (
            cfg.lambda_clean_dfl * loss_dfl
            + cfg.lambda_clean_ce * loss_ce
            + cfg.lambda_clean_b30_loss * terms["b30_full_loss"]
        )
    elif branch_name == "adv":
        total = (
            cfg.lambda_adv_dfl * loss_dfl
            + cfg.lambda_adv_ce * loss_ce
            + cfg.lambda_adv_b30_loss * terms["b30_full_loss"]
        )
    else:
        raise ValueError(f"Unknown branch_name={branch_name!r}")

    terms["loss"] = total
    return terms


def new_metric_accumulators() -> tuple[Dict[str, float], Dict[str, float]]:
    totals = {
        "loss": 0.0,
        "dfl_loss": 0.0,
        "ce_loss": 0.0,
        "b30_full_loss": 0.0,
        "b30_ce": 0.0,
        "b30_e1": 0.0,
        "b30_clst": 0.0,
        "b30_sep": 0.0,
        "b30_recon": 0.0,
        "samples": 0,
        "distances": 0,
        "mu_abs_raw": 0.0,
        "mu_abs_clamped": 0.0,
        "mu_sq_clamped": 0.0,
        "saturated": 0.0,
        "det_correct": 0,
        "risk_correct": 0,
    }
    corr_sums = {
        "n": 0,
        "sum_x": 0.0,
        "sum_y": 0.0,
        "sum_x2": 0.0,
        "sum_y2": 0.0,
        "sum_xy": 0.0,
    }
    return totals, corr_sums


def accumulate_branch_metrics(
    totals: Dict[str, float],
    corr_sums: Dict[str, float],
    terms: Dict[str, torch.Tensor],
    wrapper: RiskAwareB30Wrapper,
    y: torch.Tensor,
    cfg: TrainConfig,
) -> None:
    d_det = terms["d_det"]
    mu = terms["mu"]
    logits = terms["logits"]
    det_logits = wrapper.fc(d_det)
    target_clamped = torch.clamp(d_det, min=cfg.d_min, max=cfg.d_max)
    clamped_error = mu - target_clamped

    batch_samples = y.shape[0]
    batch_distances = d_det.numel()

    totals["samples"] += batch_samples
    totals["distances"] += batch_distances
    totals["loss"] += terms["loss"].detach().item() * batch_samples
    totals["dfl_loss"] += terms["loss_dfl"].detach().item() * batch_samples
    totals["ce_loss"] += terms["loss_ce"].detach().item() * batch_samples
    for key in ("b30_full_loss", "b30_ce", "b30_e1", "b30_clst", "b30_sep", "b30_recon"):
        totals[key] += terms[key].detach().item() * batch_samples
    totals["mu_abs_raw"] += (mu - d_det).abs().detach().sum().item()
    totals["mu_abs_clamped"] += clamped_error.abs().detach().sum().item()
    totals["mu_sq_clamped"] += clamped_error.pow(2).detach().sum().item()
    totals["saturated"] += (d_det > cfg.d_max).float().detach().sum().item()
    totals["det_correct"] += accuracy_from_logits(det_logits, y)
    totals["risk_correct"] += accuracy_from_logits(logits, y)
    update_corr_sums(corr_sums, mu, target_clamped)


def finalize_branch_metrics(
    totals: Dict[str, float],
    corr_sums: Dict[str, float],
    include_b30: bool,
) -> Dict[str, float]:
    if totals["samples"] == 0 or totals["distances"] == 0:
        raise RuntimeError("No batches were processed for a branch.")

    result = {
        "loss": totals["loss"] / totals["samples"],
        "dfl_loss": totals["dfl_loss"] / totals["samples"],
        "ce_loss": totals["ce_loss"] / totals["samples"],
        "mu_mae_raw": totals["mu_abs_raw"] / totals["distances"],
        "mu_mae_clamped": totals["mu_abs_clamped"] / totals["distances"],
        "mu_rmse_clamped": (totals["mu_sq_clamped"] / totals["distances"]) ** 0.5,
        "mu_corr_clamped": pearson_corr_from_sums(**corr_sums),
        "target_saturation_rate": totals["saturated"] / totals["distances"],
        "det_acc": totals["det_correct"] / totals["samples"],
        "risk_acc": totals["risk_correct"] / totals["samples"],
    }
    if include_b30:
        result.update(
            {
                "b30_full_loss": totals["b30_full_loss"] / totals["samples"],
                "b30_ce": totals["b30_ce"] / totals["samples"],
                "b30_e1": totals["b30_e1"] / totals["samples"],
                "b30_clst": totals["b30_clst"] / totals["samples"],
                "b30_sep": totals["b30_sep"] / totals["samples"],
                "b30_recon": totals["b30_recon"] / totals["samples"],
            }
        )
    return result


def run_epoch(
    wrapper: RiskAwareB30Wrapper,
    base_model: nn.Module,
    dfl_layer: DistributionalPrototypeDistanceLayer,
    dfl_loss_fn: DistributionFocalDistanceLoss,
    loader: torch.utils.data.DataLoader,
    cfg: TrainConfig,
    optimizer: Optional[optim.Optimizer],
    max_batches: Optional[int],
    phase: str,
) -> Dict[str, float]:
    is_train = optimizer is not None
    base_model.eval()
    wrapper.eval()
    dfl_layer.train(is_train)

    totals = {
        "loss": 0.0,
        "dfl_loss": 0.0,
        "ce_loss": 0.0,
        "samples": 0,
        "distances": 0,
        "mu_abs_raw": 0.0,
        "mu_abs_clamped": 0.0,
        "mu_sq_clamped": 0.0,
        "saturated": 0.0,
        "det_correct": 0,
        "mean_correct": 0,
    }
    corr_sums = {
        "n": 0,
        "sum_x": 0.0,
        "sum_y": 0.0,
        "sum_x2": 0.0,
        "sum_y2": 0.0,
        "sum_xy": 0.0,
    }

    context = torch.enable_grad() if is_train else torch.no_grad()
    pbar = tqdm(loader, desc=phase, leave=False)

    with context:
        for batch_idx, (x, y) in enumerate(pbar):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x = x.to(cfg.device)
            y = y.to(cfg.device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            outputs = wrapper.compute_distance_outputs(x)
            d_det = outputs["d_det"]
            mu = outputs["mu"]
            bin_logits = outputs["bin_logits"]
            mean_logits = outputs["logits"]
            det_logits = wrapper.fc(d_det)
            target_clamped = torch.clamp(d_det, min=cfg.d_min, max=cfg.d_max)

            loss_dfl = dfl_loss_fn(bin_logits, d_det.detach())
            loss_ce = F.cross_entropy(mean_logits, y)
            loss = cfg.lambda_dfl * loss_dfl + cfg.lambda_ce * loss_ce

            if is_train:
                loss.backward()
                optimizer.step()

            batch_samples = x.shape[0]
            batch_distances = d_det.numel()
            totals["samples"] += batch_samples
            totals["distances"] += batch_distances
            totals["loss"] += loss.item() * batch_samples
            totals["dfl_loss"] += loss_dfl.item() * batch_samples
            totals["ce_loss"] += loss_ce.item() * batch_samples
            totals["mu_abs_raw"] += (mu - d_det).abs().sum().item()
            clamped_error = mu - target_clamped
            totals["mu_abs_clamped"] += clamped_error.abs().sum().item()
            totals["mu_sq_clamped"] += clamped_error.pow(2).sum().item()
            totals["saturated"] += (d_det > cfg.d_max).float().sum().item()
            totals["det_correct"] += accuracy_from_logits(det_logits, y)
            totals["mean_correct"] += accuracy_from_logits(mean_logits, y)
            update_corr_sums(corr_sums, mu, target_clamped)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                mae=f"{(clamped_error.abs().mean().item()):.4f}",
            )

    if totals["samples"] == 0 or totals["distances"] == 0:
        raise RuntimeError(f"No batches were processed during {phase}.")

    return {
        "loss": totals["loss"] / totals["samples"],
        "dfl_loss": totals["dfl_loss"] / totals["samples"],
        "ce_loss": totals["ce_loss"] / totals["samples"],
        "mu_mae_raw": totals["mu_abs_raw"] / totals["distances"],
        "mu_mae_clamped": totals["mu_abs_clamped"] / totals["distances"],
        "mu_rmse_clamped": (totals["mu_sq_clamped"] / totals["distances"]) ** 0.5,
        "mu_corr_clamped": pearson_corr_from_sums(**corr_sums),
        "target_saturation_rate": totals["saturated"] / totals["distances"],
        "det_acc": totals["det_correct"] / totals["samples"],
        "mean_acc": totals["mean_correct"] / totals["samples"],
    }


def run_epoch_adv(
    wrapper: RiskAwareB30Wrapper,
    base_model: nn.Module,
    dfl_layer: DistributionalPrototypeDistanceLayer,
    dfl_loss_fn: DistributionFocalDistanceLoss,
    loader: torch.utils.data.DataLoader,
    cfg: TrainConfig,
    optimizer: Optional[optim.Optimizer],
    max_batches: Optional[int],
    phase: str,
) -> Dict[str, float]:
    is_train = optimizer is not None
    base_model.eval()
    wrapper.eval()
    dfl_layer.train(is_train)

    clean_totals, clean_corr = new_metric_accumulators()
    adv_totals, adv_corr = new_metric_accumulators()
    pbar = tqdm(loader, desc=phase, leave=False)

    for batch_idx, (x, y) in enumerate(pbar):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(cfg.device)
        y = y.to(cfg.device)

        x_adv = generate_adv_batch_for_dfl_head(wrapper, dfl_layer, x, y, cfg)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            adv_terms = compute_dfl_branch_terms(
                wrapper=wrapper,
                dfl_loss_fn=dfl_loss_fn,
                x_input=x_adv,
                y=y,
                cfg=cfg,
                branch_name="adv",
            )
            adv_terms["loss"].backward()
            assert_no_base_gradients(base_model)
            if batch_idx == 0:
                assert_dfl_has_finite_gradient(dfl_layer)
            optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            clean_terms = compute_dfl_branch_terms(
                wrapper=wrapper,
                dfl_loss_fn=dfl_loss_fn,
                x_input=x,
                y=y,
                cfg=cfg,
                branch_name="clean",
            )
            clean_terms["loss"].backward()
            assert_no_base_gradients(base_model)
            if batch_idx == 0:
                assert_dfl_has_finite_gradient(dfl_layer)
            optimizer.step()
        else:
            with torch.no_grad():
                clean_terms = compute_dfl_branch_terms(
                    wrapper=wrapper,
                    dfl_loss_fn=dfl_loss_fn,
                    x_input=x,
                    y=y,
                    cfg=cfg,
                    branch_name="clean",
                )
                adv_terms = compute_dfl_branch_terms(
                    wrapper=wrapper,
                    dfl_loss_fn=dfl_loss_fn,
                    x_input=x_adv,
                    y=y,
                    cfg=cfg,
                    branch_name="adv",
                )

        accumulate_branch_metrics(clean_totals, clean_corr, clean_terms, wrapper, y, cfg)
        accumulate_branch_metrics(adv_totals, adv_corr, adv_terms, wrapper, y, cfg)

        pbar.set_postfix(
            clean=f"{clean_terms['loss'].detach().item():.4f}",
            adv=f"{adv_terms['loss'].detach().item():.4f}",
        )

    clean_metrics = finalize_branch_metrics(clean_totals, clean_corr, cfg.use_b30_clstsep_loss)
    adv_metrics = finalize_branch_metrics(adv_totals, adv_corr, cfg.use_b30_clstsep_loss)
    return {
        **{f"clean_{name}": value for name, value in clean_metrics.items()},
        **{f"adv_{name}": value for name, value in adv_metrics.items()},
    }


def save_dfl_checkpoint(
    path: Path,
    cfg: TrainConfig,
    dfl_layer: DistributionalPrototypeDistanceLayer,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val_mu_mae_clamped: float,
    best_val_mean_acc: float,
    best_monitor_value: float,
    latent_dim: int,
    checkpoint_path: Optional[Path],
    init_dfl_head_path: Optional[Path],
) -> None:
    torch.save(
        {
            "model_key": cfg.model_key,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
            "epoch": epoch,
            "dfl_layer_state": dfl_layer.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_mu_mae_clamped": best_val_mu_mae_clamped,
            "best_val_mean_acc": best_val_mean_acc,
            "best_monitor_value": best_monitor_value,
            "latent_dim": latent_dim,
            "num_bins": cfg.num_bins,
            "d_min": cfg.d_min,
            "d_max": cfg.d_max,
            "hidden_dim": cfg.hidden_dim,
            "adv_train": cfg.adv_train,
            "adv_eps": cfg.adv_eps,
            "adv_alpha": cfg.adv_alpha,
            "adv_steps": cfg.adv_steps,
            "adv_random_start": cfg.adv_random_start,
            "adv_beta": cfg.adv_beta,
            "lambda_clean_dfl": cfg.lambda_clean_dfl,
            "lambda_clean_ce": cfg.lambda_clean_ce,
            "lambda_adv_dfl": cfg.lambda_adv_dfl,
            "lambda_adv_ce": cfg.lambda_adv_ce,
            "lambda_clean_b30_loss": cfg.lambda_clean_b30_loss,
            "lambda_adv_b30_loss": cfg.lambda_adv_b30_loss,
            "early_stop_monitor": cfg.early_stop_monitor,
            "init_dfl_head": str(init_dfl_head_path) if init_dfl_head_path is not None else None,
            "config": asdict(cfg),
        },
        path,
    )


def build_config(args: argparse.Namespace) -> TrainConfig:
    run_tag = args.run_tag
    epochs = args.epochs
    max_train_batches = args.max_train_batches
    max_val_batches = args.max_val_batches
    adv_steps = args.adv_steps

    if args.adv_train and not run_tag:
        run_tag = f"adv_eps{safe_float_label(args.adv_eps)}_beta{safe_float_label(args.adv_beta)}"

    if args.smoke_test:
        epochs = 1
        max_train_batches = 2 if max_train_batches is None else min(max_train_batches, 2)
        max_val_batches = 1 if max_val_batches is None else min(max_val_batches, 1)
        if args.adv_train and adv_steps == 40:
            adv_steps = 2
        run_tag = "smoke_test" if not run_tag else f"{run_tag}_smoke_test"

    lambda_clean_dfl = (
        args.lambda_dfl if args.lambda_clean_dfl is None else args.lambda_clean_dfl
    )
    lambda_clean_ce = (
        args.lambda_ce if args.lambda_clean_ce is None else args.lambda_clean_ce
    )

    return TrainConfig(
        model_key=args.model_key,
        checkpoint=args.checkpoint,
        seed=args.seed,
        epochs=epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_size=args.val_size,
        early_stopping_patience=args.early_stopping_patience,
        data_dir=args.data_dir,
        save_root=args.save_root
        if args.save_root is not None
        else resolve_default_save_root(),
        num_workers=args.num_workers
        if args.num_workers is not None
        else resolve_default_num_workers(),
        device=args.device,
        num_bins=args.num_bins,
        d_min=args.d_min,
        d_max=args.d_max,
        hidden_dim=args.hidden_dim,
        lambda_dfl=args.lambda_dfl,
        lambda_ce=args.lambda_ce,
        adv_train=args.adv_train,
        adv_eps=args.adv_eps,
        adv_alpha=args.adv_alpha,
        adv_steps=adv_steps,
        adv_random_start=not args.no_adv_random_start,
        adv_beta=args.adv_beta,
        lambda_clean_dfl=lambda_clean_dfl,
        lambda_clean_ce=lambda_clean_ce,
        lambda_adv_dfl=args.lambda_adv_dfl,
        lambda_adv_ce=args.lambda_adv_ce,
        use_b30_clstsep_loss=args.use_b30_clstsep_loss,
        lambda_clean_b30_loss=args.lambda_clean_b30_loss,
        lambda_adv_b30_loss=args.lambda_adv_b30_loss,
        b30_lambda_class=args.b30_lambda_class,
        b30_lambda_ae=args.b30_lambda_ae,
        b30_lambda_1=args.b30_lambda_1,
        b30_lambda_clus=args.b30_lambda_clus,
        b30_lambda_sep=args.b30_lambda_sep,
        init_dfl_head=args.init_dfl_head,
        init_run_tag=args.init_run_tag,
        early_stop_monitor=args.early_stop_monitor,
        max_train_batches=max_train_batches,
        max_val_batches=max_val_batches,
        run_tag=run_tag,
        smoke_test=args.smoke_test,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train only a DistributionalPrototypeDistanceLayer for B30."
    )
    parser.add_argument("--model-key", choices=tuple(KNOWN_MODEL_PATHS.keys()), default="B30")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--early-stopping-patience", type=int, default=5)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--save-root", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-bins", type=int, default=32)
    parser.add_argument("--d-min", type=float, default=0.0)
    parser.add_argument("--d-max", type=float, default=12.0)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lambda-dfl", type=float, default=1.0)
    parser.add_argument("--lambda-ce", type=float, default=0.0)
    parser.add_argument("--adv-train", action="store_true")
    parser.add_argument("--adv-eps", type=float, default=0.3)
    parser.add_argument("--adv-alpha", type=float, default=0.01)
    parser.add_argument("--adv-steps", type=int, default=40)
    parser.add_argument("--no-adv-random-start", action="store_true")
    parser.add_argument("--adv-beta", type=float, default=0.5)
    parser.add_argument("--lambda-clean-dfl", type=float, default=None)
    parser.add_argument("--lambda-clean-ce", type=float, default=None)
    parser.add_argument("--lambda-adv-dfl", type=float, default=1.0)
    parser.add_argument("--lambda-adv-ce", type=float, default=1.0)
    parser.add_argument("--use-b30-clstsep-loss", action="store_true")
    parser.add_argument("--lambda-clean-b30-loss", type=float, default=0.0)
    parser.add_argument("--lambda-adv-b30-loss", type=float, default=0.0)
    parser.add_argument("--b30-lambda-class", type=float, default=20.0)
    parser.add_argument("--b30-lambda-ae", type=float, default=1.0)
    parser.add_argument("--b30-lambda-1", type=float, default=1.0)
    parser.add_argument("--b30-lambda-clus", type=float, default=0.8)
    parser.add_argument("--b30-lambda-sep", type=float, default=0.2)
    parser.add_argument("--init-dfl-head", default=None)
    parser.add_argument("--init-run-tag", default=None)
    parser.add_argument(
        "--early-stop-monitor",
        choices=(
            "val_adv_risk_acc",
            "val_adv_loss",
            "val_clean_risk_acc",
            "val_clean_mu_mae_clamped",
        ),
        default="val_adv_risk_acc",
    )
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    set_seed(cfg.seed)

    if cfg.smoke_test and cfg.adv_train:
        print(
            "Smoke-test adversarial calibration mode enabled: this is not a "
            "real training run."
        )
    elif cfg.smoke_test:
        print("Smoke-test mode enabled: this is not a real training run.")

    run_leaf = f"seed={cfg.seed}" if not cfg.run_tag else f"seed={cfg.seed}_{cfg.run_tag}"
    run_dir = Path(cfg.save_root) / cfg.model_key / run_leaf
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    device = torch.device(cfg.device)
    pin_memory = device.type == "cuda"

    train_loader, val_loader = get_train_val_loader(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        random_seed=cfg.seed,
        val_size=cfg.val_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    checkpoint_path = resolve_checkpoint_path(cfg)
    base_model = load_b30_model(checkpoint_path, device)
    base_model.to(device)
    freeze_base_model(base_model)
    base_model.eval()

    latent_dim = int(base_model.in_channels_prototype)
    dfl_layer = DistributionalPrototypeDistanceLayer(
        latent_dim=latent_dim,
        num_bins=cfg.num_bins,
        d_min=cfg.d_min,
        d_max=cfg.d_max,
        hidden_dim=cfg.hidden_dim,
    ).to(device)
    init_dfl_head_path = load_initial_dfl_head_if_requested(
        dfl_layer=dfl_layer,
        cfg=cfg,
        latent_dim=latent_dim,
        device=device,
    )
    wrapper_mode = "risk" if cfg.adv_train else "mean"
    wrapper = RiskAwareB30Wrapper(
        base_model=base_model,
        mode=wrapper_mode,
        beta=cfg.adv_beta if cfg.adv_train else 0.0,
        dfl_layer=dfl_layer,
    ).to(device)
    wrapper.eval()

    base_total_params, base_trainable_params = count_parameters(base_model)
    dfl_total_params, dfl_trainable_params = count_parameters(dfl_layer)
    trainable_total_params = base_trainable_params + dfl_trainable_params

    print(f"Model key:                  {cfg.model_key}")
    print(f"Checkpoint:                 {checkpoint_path}")
    print(f"Device:                     {cfg.device}")
    print(f"Run dir:                    {run_dir}")
    print(f"Adv train:                  {cfg.adv_train}")
    if cfg.adv_train:
        print(f"Adv eps / alpha / steps:    {cfg.adv_eps} / {cfg.adv_alpha} / {cfg.adv_steps}")
        print(f"Adv beta:                   {cfg.adv_beta}")
        print(f"Early-stop monitor:         {cfg.early_stop_monitor}")
        print(f"Init DFL head:              {init_dfl_head_path}")
    print(f"Total base parameters:      {base_total_params:,}")
    print(f"Trainable base parameters:  {base_trainable_params:,}")
    print(f"DFL layer parameters:       {dfl_total_params:,}")
    print(f"Trainable total parameters: {trainable_total_params:,}")

    assert base_trainable_params == 0, "base model parameters must be frozen"
    assert all(not parameter.requires_grad for parameter in base_model.parameters()), (
        "at least one base model parameter still requires grad"
    )
    assert dfl_trainable_params > 0, "DFL layer must have trainable parameters"

    save_trainable_parameter_names(dfl_layer, run_dir / "trainable_parameters.txt")
    save_json(run_dir / "config.json", asdict(cfg))

    optimizer = optim.Adam(
        [parameter for parameter in dfl_layer.parameters() if parameter.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    dfl_loss_fn = DistributionFocalDistanceLoss(
        num_bins=cfg.num_bins,
        d_min=cfg.d_min,
        d_max=cfg.d_max,
        reduction="mean",
    )

    history: List[Dict[str, Any]] = []
    best_val_mu_mae_clamped = float("inf")
    best_val_mean_acc = 0.0
    best_monitor_value = -float("inf") if cfg.early_stop_monitor.endswith("_acc") else float("inf")
    best_epoch = -1
    patience_counter = 0
    start_time = time.time()

    best_ckpt_path = ckpt_dir / "best_dfl_head.pt"
    last_ckpt_path = ckpt_dir / "last_dfl_head.pt"

    for epoch in range(1, cfg.epochs + 1):
        if cfg.adv_train:
            train_metrics = run_epoch_adv(
                wrapper=wrapper,
                base_model=base_model,
                dfl_layer=dfl_layer,
                dfl_loss_fn=dfl_loss_fn,
                loader=train_loader,
                cfg=cfg,
                optimizer=optimizer,
                max_batches=cfg.max_train_batches,
                phase=f"train {epoch}/{cfg.epochs}",
            )
            val_metrics = run_epoch_adv(
                wrapper=wrapper,
                base_model=base_model,
                dfl_layer=dfl_layer,
                dfl_loss_fn=dfl_loss_fn,
                loader=val_loader,
                cfg=cfg,
                optimizer=None,
                max_batches=cfg.max_val_batches,
                phase=f"val {epoch}/{cfg.epochs}",
            )
        else:
            train_metrics = run_epoch(
                wrapper=wrapper,
                base_model=base_model,
                dfl_layer=dfl_layer,
                dfl_loss_fn=dfl_loss_fn,
                loader=train_loader,
                cfg=cfg,
                optimizer=optimizer,
                max_batches=cfg.max_train_batches,
                phase=f"train {epoch}/{cfg.epochs}",
            )
            val_metrics = run_epoch(
                wrapper=wrapper,
                base_model=base_model,
                dfl_layer=dfl_layer,
                dfl_loss_fn=dfl_loss_fn,
                loader=val_loader,
                cfg=cfg,
                optimizer=None,
                max_batches=cfg.max_val_batches,
                phase=f"val {epoch}/{cfg.epochs}",
            )

        row = {
            "epoch": epoch,
            **{f"train_{name}": value for name, value in train_metrics.items()},
            **{f"val_{name}": value for name, value in val_metrics.items()},
        }
        history.append(row)
        save_history_csv(run_dir / "history.csv", history)
        save_json(run_dir / "history.json", {"history": history})

        if cfg.adv_train:
            monitor_value = row[cfg.early_stop_monitor]
            improved = (
                monitor_value > best_monitor_value
                if cfg.early_stop_monitor.endswith("_acc")
                else monitor_value < best_monitor_value
            )
            print(
                "epoch "
                f"{epoch:03d} | "
                f"train_clean_loss={train_metrics['clean_loss']:.5f} "
                f"train_adv_loss={train_metrics['adv_loss']:.5f} "
                f"val_clean_acc={val_metrics['clean_risk_acc']:.4f} "
                f"val_adv_acc={val_metrics['adv_risk_acc']:.4f} "
                f"val_adv_loss={val_metrics['adv_loss']:.5f} "
                f"monitor={cfg.early_stop_monitor}:{monitor_value:.5f}"
            )
        else:
            monitor_value = val_metrics["mu_mae_clamped"]
            improved = monitor_value < best_val_mu_mae_clamped
            print(
                "epoch "
                f"{epoch:03d} | "
                f"train_loss={train_metrics['loss']:.5f} "
                f"train_mae={train_metrics['mu_mae_clamped']:.5f} "
                f"val_loss={val_metrics['loss']:.5f} "
                f"val_mae={val_metrics['mu_mae_clamped']:.5f} "
                f"val_mean_acc={val_metrics['mean_acc']:.4f} "
                f"val_det_acc={val_metrics['det_acc']:.4f} "
                f"sat={val_metrics['target_saturation_rate']:.6f}"
            )

        if improved:
            if cfg.adv_train:
                best_val_mu_mae_clamped = val_metrics["clean_mu_mae_clamped"]
                best_val_mean_acc = val_metrics["clean_risk_acc"]
                best_monitor_value = monitor_value
            else:
                best_val_mu_mae_clamped = val_metrics["mu_mae_clamped"]
                best_val_mean_acc = val_metrics["mean_acc"]
                best_monitor_value = monitor_value
            best_epoch = epoch
            patience_counter = 0
            save_dfl_checkpoint(
                path=best_ckpt_path,
                cfg=cfg,
                dfl_layer=dfl_layer,
                optimizer=optimizer,
                epoch=epoch,
                best_val_mu_mae_clamped=best_val_mu_mae_clamped,
                best_val_mean_acc=best_val_mean_acc,
                best_monitor_value=best_monitor_value,
                latent_dim=latent_dim,
                checkpoint_path=checkpoint_path,
                init_dfl_head_path=init_dfl_head_path,
            )
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(
                    "Early stopping triggered after "
                    f"{patience_counter} epochs without improvement."
                )
                break

    epochs_run = history[-1]["epoch"] if history else 0
    total_runtime_s = time.time() - start_time
    save_dfl_checkpoint(
        path=last_ckpt_path,
        cfg=cfg,
        dfl_layer=dfl_layer,
        optimizer=optimizer,
        epoch=epochs_run,
        best_val_mu_mae_clamped=best_val_mu_mae_clamped,
        best_val_mean_acc=best_val_mean_acc,
        best_monitor_value=best_monitor_value,
        latent_dim=latent_dim,
        checkpoint_path=checkpoint_path,
        init_dfl_head_path=init_dfl_head_path,
    )

    summary = {
        "model_key": cfg.model_key,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_mu_mae_clamped": best_val_mu_mae_clamped,
        "best_val_mean_acc": best_val_mean_acc,
        "best_monitor_value": best_monitor_value,
        "early_stop_monitor": cfg.early_stop_monitor,
        "adv_train": cfg.adv_train,
        "init_dfl_head": str(init_dfl_head_path) if init_dfl_head_path is not None else None,
        "epochs_run": epochs_run,
        "total_runtime_s": total_runtime_s,
        "base_total_params": base_total_params,
        "base_trainable_params": base_trainable_params,
        "dfl_total_params": dfl_total_params,
        "dfl_trainable_params": dfl_trainable_params,
        "trainable_total_params": trainable_total_params,
        "last_epoch_metrics": history[-1] if history else None,
        "config": asdict(cfg),
    }
    save_json(run_dir / "summary.json", summary)

    print("\nFinished DFL-head training.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val_mu_mae_clamped: {best_val_mu_mae_clamped:.6g}")
    print(f"Best val_mean_acc: {best_val_mean_acc:.6g}")
    print(f"Best monitor value: {best_monitor_value:.6g}")
    print(f"Saved best checkpoint: {best_ckpt_path}")
    print(f"Saved last checkpoint: {last_ckpt_path}")


if __name__ == "__main__":
    main()
