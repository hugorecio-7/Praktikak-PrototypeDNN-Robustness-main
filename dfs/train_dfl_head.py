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

from data_loader import get_train_val_loader  # noqa: E402
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


def save_dfl_checkpoint(
    path: Path,
    cfg: TrainConfig,
    dfl_layer: DistributionalPrototypeDistanceLayer,
    optimizer: optim.Optimizer,
    epoch: int,
    best_val_mu_mae_clamped: float,
    best_val_mean_acc: float,
    latent_dim: int,
    checkpoint_path: Optional[Path],
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
            "latent_dim": latent_dim,
            "num_bins": cfg.num_bins,
            "d_min": cfg.d_min,
            "d_max": cfg.d_max,
            "hidden_dim": cfg.hidden_dim,
            "config": asdict(cfg),
        },
        path,
    )


def build_config(args: argparse.Namespace) -> TrainConfig:
    run_tag = args.run_tag
    epochs = args.epochs
    max_train_batches = args.max_train_batches
    max_val_batches = args.max_val_batches

    if args.smoke_test:
        epochs = 1
        max_train_batches = 2 if max_train_batches is None else min(max_train_batches, 2)
        max_val_batches = 1 if max_val_batches is None else min(max_val_batches, 1)
        run_tag = "smoke_test" if not run_tag else f"{run_tag}_smoke_test"

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
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    set_seed(cfg.seed)

    if cfg.smoke_test:
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
    wrapper = RiskAwareB30Wrapper(
        base_model=base_model,
        mode="mean",
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
    best_epoch = -1
    patience_counter = 0
    start_time = time.time()

    best_ckpt_path = ckpt_dir / "best_dfl_head.pt"
    last_ckpt_path = ckpt_dir / "last_dfl_head.pt"

    for epoch in range(1, cfg.epochs + 1):
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

        if val_metrics["mu_mae_clamped"] < best_val_mu_mae_clamped:
            best_val_mu_mae_clamped = val_metrics["mu_mae_clamped"]
            best_val_mean_acc = val_metrics["mean_acc"]
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
                latent_dim=latent_dim,
                checkpoint_path=checkpoint_path,
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
        latent_dim=latent_dim,
        checkpoint_path=checkpoint_path,
    )

    summary = {
        "model_key": cfg.model_key,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path is not None else None,
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_val_mu_mae_clamped": best_val_mu_mae_clamped,
        "best_val_mean_acc": best_val_mean_acc,
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
    print(f"Saved best checkpoint: {best_ckpt_path}")
    print(f"Saved last checkpoint: {last_ckpt_path}")


if __name__ == "__main__":
    main()
