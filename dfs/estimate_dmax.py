"""Estimate the real range of B30 squared prototype distances on MNIST."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules import CAEModel_Balanced  # noqa: E402


KNOWN_MODEL_PATHS = {
    "B30":  "saved_model/mnist_model/mnist_cae_balanced_clstsep_1500_0.002_250_True_0.0_20_1_1_1_1.0_0.0_30_4_32_1/mnist_cae00750.pth",
    "B30-FT-0": "tfg_models/B30/B30-FT-0/seed=1/checkpoints/B30_B30-FT-0_seed1_best_val_adv_acc.pth",
    "B30-FT-1": "tfg_models/B30/B30-FT-1/seed=1/checkpoints/B30_B30-FT-1_seed1_best_val_adv_acc.pth",
    "B30-FT-2": "tfg_models/B30/B30-FT-2/seed=1/checkpoints/B30_B30-FT-2_seed1_best_val_adv_acc.pth",
    "B30-FT-3": "tfg_models/B30/B30-FT-3/seed=1/checkpoints/B30_B30-FT-3_seed1_best_val_adv_acc.pth",
    "B30-FT-E": "tfg_models/B30/B30-FT-E/seed=1/checkpoints/B30_B30-FT-E_seed1_best_val_adv_acc.pth",
    "B30-FT-E-M": "tfg_models/B30/B30-FT-E-M/seed=1/checkpoints/B30_B30-FT-E_seed1_best_val_adv_acc.pth",
}


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="Estimate squared B30 prototype-distance statistics on MNIST."
    )
    parser.add_argument("--dataset-root", default="./data")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--model-key",
        choices=tuple(KNOWN_MODEL_PATHS.keys()),
        default=None,
        help="Optional known B30 checkpoint key. Ignored when --checkpoint is set.",
    )
    parser.add_argument("--output", default="dfs/results/dmax_estimate.json")
    return parser.parse_args()


def torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {
        key.removeprefix("module."): value
        for key, value in state_dict.items()
    }


def extract_state_dict(checkpoint: Any) -> Optional[Dict[str, torch.Tensor]]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model_state"):
            if key in checkpoint:
                return checkpoint[key]
        if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    return None


def load_model(checkpoint_path: Optional[Path], device: torch.device) -> nn.Module:
    if checkpoint_path is None:
        model = CAEModel_Balanced().to(device)
        print("Loaded randomly initialized CAEModel_Balanced.")
        return model

    checkpoint = torch_load(checkpoint_path, device)
    if isinstance(checkpoint, nn.Module):
        model = checkpoint.to(device)
        print(f"Loaded full model object from {checkpoint_path}.")
        return model

    state_dict = extract_state_dict(checkpoint)
    if state_dict is None:
        raise TypeError(
            "Checkpoint is neither a torch.nn.Module nor a supported state dict "
            "format. Expected raw state dict or dict with one of: "
            "'state_dict', 'model_state_dict', 'model_state'."
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
            f"Missing keys ({len(missing)}): {missing[:20]}\n"
            f"Unexpected keys ({len(unexpected)}): {unexpected[:20]}"
        ) from exc

    print(f"Loaded CAEModel_Balanced state dict from {checkpoint_path}.")
    return model


def make_mnist_loader(
    dataset_root: str,
    split: str,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    dataset = datasets.MNIST(
        root=dataset_root,
        train=(split == "train"),
        download=True,
        transform=transforms.ToTensor(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def compute_percentiles(distances: torch.Tensor) -> Dict[str, float]:
    percentile_specs = {
        "p50": 0.50,
        "p90": 0.90,
        "p95": 0.95,
        "p99": 0.99,
        "p99.5": 0.995,
        "p99.9": 0.999,
    }
    return {
        name: torch.quantile(distances, q).item()
        for name, q in percentile_specs.items()
    }


def estimate_distances(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int],
) -> tuple[torch.Tensor, int]:
    collected = []
    samples_processed = 0

    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            x = x.to(device)
            encoder_out = model.encoder(x)
            z = encoder_out.view(-1, model.in_channels_prototype)
            d_target = model.prototype_layer(z)
            collected.append(d_target.detach().cpu().reshape(-1))
            samples_processed += x.shape[0]

    if not collected:
        raise RuntimeError("No distances were collected. Check --max-batches and dataset size.")

    return torch.cat(collected), samples_processed


def build_summary(
    distances: torch.Tensor,
    samples_processed: int,
    split: str,
    checkpoint: Optional[str],
) -> Dict[str, Any]:
    distances = distances.float()
    percentiles = compute_percentiles(distances)
    candidate_values = {
        "p95": percentiles["p95"],
        "p99": percentiles["p99"],
        "p99.5": percentiles["p99.5"],
        "p99.9": percentiles["p99.9"],
        "max": distances.max().item(),
    }
    saturation_rates = {
        name: (distances > value).float().mean().item()
        for name, value in candidate_values.items()
    }

    return {
        "split": split,
        "checkpoint": checkpoint,
        "num_samples_processed": samples_processed,
        "num_distances_processed": distances.numel(),
        "min": distances.min().item(),
        "max": distances.max().item(),
        "mean": distances.mean().item(),
        "std": distances.std(unbiased=False).item(),
        "percentiles": percentiles,
        "candidate_d_max": candidate_values,
        "saturation_rates": saturation_rates,
        "recommended_d_max": percentiles["p99.5"],
        "recommendation_note": (
            "p99.5 is a balanced default: smaller d_max gives more bin "
            "resolution for common distances, while larger d_max such as p99.9 "
            "or max avoids saturating rare large distances."
        ),
    }


def print_summary(summary: Dict[str, Any]) -> None:
    print("\nB30 squared prototype-distance range estimate")
    print(f"split: {summary['split']}")
    print(f"checkpoint: {summary['checkpoint'] or 'random initialization'}")
    print(f"samples processed: {summary['num_samples_processed']}")
    print(f"distances processed: {summary['num_distances_processed']}")
    print(f"min: {summary['min']:.6g}")
    print(f"max: {summary['max']:.6g}")
    print(f"mean: {summary['mean']:.6g}")
    print(f"std: {summary['std']:.6g}")

    print("\nPercentiles")
    for name, value in summary["percentiles"].items():
        print(f"{name}: {value:.6g}")

    print("\nCandidate d_max saturation rates")
    for name, value in summary["candidate_d_max"].items():
        rate = summary["saturation_rates"][name]
        print(f"{name} = {value:.6g}: fraction clamped above = {rate:.6g}")

    print(f"\nrecommended_d_max: {summary['recommended_d_max']:.6g}")
    print(summary["recommendation_note"])


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = args.checkpoint
    if checkpoint is None and args.model_key is not None:
        checkpoint = KNOWN_MODEL_PATHS[args.model_key]

    checkpoint_path = Path(checkpoint) if checkpoint is not None else None
    if checkpoint_path is not None and not checkpoint_path.is_absolute():
        checkpoint_path = REPO_ROOT / checkpoint_path

    model = load_model(checkpoint_path, device)
    model.to(device)
    if hasattr(model, "prototype_layer") and hasattr(model.prototype_layer, "prototype_distances"):
        model.prototype_layer.prototype_distances = (
            model.prototype_layer.prototype_distances.to(device)
        )

    loader = make_mnist_loader(
        dataset_root=args.dataset_root,
        split=args.split,
        batch_size=args.batch_size,
        device=device,
    )
    distances, samples_processed = estimate_distances(
        model=model,
        loader=loader,
        device=device,
        max_batches=args.max_batches,
    )
    summary = build_summary(
        distances=distances,
        samples_processed=samples_processed,
        split=args.split,
        checkpoint=str(checkpoint_path) if checkpoint_path is not None else None,
    )

    print_summary(summary)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved JSON summary to {output_path}")


if __name__ == "__main__":
    main()
