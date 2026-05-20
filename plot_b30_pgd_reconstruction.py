"""
Plot one MNIST sample, its PGD adversarial version, and the B30 decoder
reconstruction of the adversarial image.

Edit the variables below or override them from the command line, for example:

    python plot_b30_pgd_reconstruction.py --sample-index 12 --model-name B30-FT-E-M
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms

import modules  # noqa: F401  # Needed when loading pickled B30 model objects.


# ============================================================
# User-editable variables
# ============================================================
SAMPLE_INDEX = 0
MODEL_NAME = "B30"
DATASET_SPLIT = "test"  # "test" or "train"
DATA_DIR = "data"
OUTPUT_PATH = "resultsTFG/b30_pgd_reconstruction/sample_reconstruction.png"

PGD_EPS = 0.1
PGD_ALPHA = 0.01
PGD_STEPS = 40
PGD_RANDOM_START = True


# Same B30 paths used by the existing metrics/fine-tuning scripts.
B30_BASE_CKPT = (
    "saved_model/mnist_model/"
    "mnist_cae_balanced_clstsep_1500_0.002_250_True_0.0_20_1_1_1_1.0_0.0_30_4_32_1/"
    "mnist_cae00750.pth"
)

B30_FT_CKPTS = {
    "B30-FT-0": "tfg_models/B30/B30-FT-0/seed=1/checkpoints/B30_B30-FT-0_seed1_best_val_adv_acc.pth",
    "B30-FT-1": "tfg_models/B30/B30-FT-1/seed=1/checkpoints/B30_B30-FT-1_seed1_best_val_adv_acc.pth",
    "B30-FT-2": "tfg_models/B30/B30-FT-2/seed=1/checkpoints/B30_B30-FT-2_seed1_best_val_adv_acc.pth",
    "B30-FT-3": "tfg_models/B30/B30-FT-3/seed=1/checkpoints/B30_B30-FT-3_seed1_best_val_adv_acc.pth",
    "B30-FT-E": "tfg_models/B30/B30-FT-E/seed=1/checkpoints/B30_B30-FT-E_seed1_best_val_adv_acc.pth",
    "B30-FT-E-M": "tfg_models/B30/B30-FT-E-M/seed=1/checkpoints/B30_B30-FT-E_seed1_best_val_adv_acc.pth",
}


def load_torch_object(path: str | Path, device: torch.device):
    """Load checkpoints across PyTorch versions where weights_only may differ."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_b30_model(model_name: str, device: torch.device) -> nn.Module:
    if model_name == "B30":
        model = load_torch_object(B30_BASE_CKPT, device)
    elif model_name in B30_FT_CKPTS:
        model = load_torch_object(B30_BASE_CKPT, device)
        checkpoint = load_torch_object(B30_FT_CKPTS[model_name], device)
        state_dict = (
            checkpoint["model_state"]
            if isinstance(checkpoint, dict) and "model_state" in checkpoint
            else checkpoint
        )
        model.load_state_dict(state_dict)
    else:
        valid = ", ".join(["B30", *B30_FT_CKPTS.keys()])
        raise ValueError(f"Unknown model_name '{model_name}'. Valid values: {valid}")

    if not isinstance(model, nn.Module):
        raise TypeError(f"Checkpoint for {model_name} did not load as a torch.nn.Module.")

    return model.to(device).eval()


def load_mnist_sample(data_dir: str, split: str, sample_index: int):
    if split not in {"train", "test"}:
        raise ValueError("DATASET_SPLIT must be 'train' or 'test'.")

    dataset = datasets.MNIST(
        root=data_dir,
        train=(split == "train"),
        download=False,
        transform=transforms.ToTensor(),
    )

    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            f"sample_index={sample_index} is outside the {split} set "
            f"(valid range: 0..{len(dataset) - 1})."
        )

    image, label = dataset[sample_index]
    return image.unsqueeze(0), torch.tensor([label], dtype=torch.long)


def pgd_linf_attack(
    model: nn.Module,
    image: torch.Tensor,
    label: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    random_start: bool,
) -> torch.Tensor:
    original = image.detach()
    attacked = original.clone()

    if random_start:
        attacked = attacked + torch.empty_like(attacked).uniform_(-eps, eps)
        attacked = torch.clamp(attacked, 0.0, 1.0).detach()

    for _ in range(steps):
        attacked.requires_grad_(True)
        loss = F.cross_entropy(model(attacked), label)
        grad = torch.autograd.grad(loss, attacked)[0]

        attacked = attacked.detach() + alpha * grad.sign()
        delta = torch.clamp(attacked - original, min=-eps, max=eps)
        attacked = torch.clamp(original + delta, 0.0, 1.0).detach()

    return attacked


def reconstruct_with_decoder(model: nn.Module, image: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        encoded = model.encoder(image)
        reconstructed = model.decoder(encoded)
    return torch.clamp(reconstructed, 0.0, 1.0)


def predict(model: nn.Module, image: torch.Tensor) -> int:
    with torch.no_grad():
        return int(model(image).argmax(dim=1).item())


def to_plot_image(image: torch.Tensor):
    return image.detach().squeeze(0).squeeze(0).cpu().numpy()


def save_plot(
    original: torch.Tensor,
    attacked: torch.Tensor,
    reconstructed: torch.Tensor,
    label: int,
    pred_original: int,
    pred_attacked: int,
    pred_reconstructed: int,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panels = [
        ("Original", original, pred_original),
        ("PGD", attacked, pred_attacked),
        ("Reconstruida", reconstructed, pred_reconstructed),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, (title, image, pred) in zip(axes, panels):
        ax.imshow(to_plot_image(image), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(f"{title}\ny={label}, pred={pred}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", type=int, default=SAMPLE_INDEX)
    parser.add_argument("--model-name", type=str, default=MODEL_NAME)
    parser.add_argument("--dataset-split", type=str, default=DATASET_SPLIT, choices=["train", "test"])
    parser.add_argument("--data-dir", type=str, default=DATA_DIR)
    parser.add_argument("--output-path", type=str, default=OUTPUT_PATH)
    parser.add_argument("--eps", type=float, default=PGD_EPS)
    parser.add_argument("--alpha", type=float, default=PGD_ALPHA)
    parser.add_argument("--steps", type=int, default=PGD_STEPS)
    parser.add_argument("--no-random-start", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_b30_model(args.model_name, device)
    image, label = load_mnist_sample(args.data_dir, args.dataset_split, args.sample_index)
    image = image.to(device)
    label = label.to(device)

    attacked = pgd_linf_attack(
        model=model,
        image=image,
        label=label,
        eps=args.eps,
        alpha=args.alpha,
        steps=args.steps,
        random_start=not args.no_random_start,
    )
    reconstructed = reconstruct_with_decoder(model, attacked)

    pred_original = predict(model, image)
    pred_attacked = predict(model, attacked)
    pred_reconstructed = predict(model, reconstructed)

    save_plot(
        original=image,
        attacked=attacked,
        reconstructed=reconstructed,
        label=int(label.item()),
        pred_original=pred_original,
        pred_attacked=pred_attacked,
        pred_reconstructed=pred_reconstructed,
        output_path=args.output_path,
    )

    print(f"Model: {args.model_name}")
    print(f"Split/sample: {args.dataset_split}/{args.sample_index}")
    print(f"True label: {int(label.item())}")
    print(f"Pred original: {pred_original}")
    print(f"Pred attacked: {pred_attacked}")
    print(f"Pred reconstructed: {pred_reconstructed}")
    print(f"Saved plot to: {Path(args.output_path).resolve()}")


if __name__ == "__main__":
    main()
