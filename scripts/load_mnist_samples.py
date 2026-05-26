"""Load a small MNIST sample grid, ideally one image per class.

Examples
--------
    python scripts/load_mnist_samples.py
    python scripts/load_mnist_samples.py --split test --output results/mnist_samples.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms


REPO_ROOT = Path(__file__).resolve().parent.parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot MNIST samples from the dataset.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data",
        help="MNIST root directory.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "test"),
        default="test",
        help="Which MNIST split to load.",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=1,
        help="Number of images to show for each class.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "analysis" / "figures" / "mnist_samples.pdf",
        help="Output PDF path.",
    )
    return parser


def _load_dataset(data_dir: Path, split: str):
    return datasets.MNIST(
        root=data_dir,
        train=(split == "train"),
        download=True,
        transform=transforms.ToTensor(),
    )


def _collect_samples(dataset, per_class: int):
    images_by_class = {digit: [] for digit in range(10)}

    for image, label in dataset:
        label_int = int(label)
        if len(images_by_class[label_int]) < per_class:
            images_by_class[label_int].append(image.squeeze(0))
        if all(len(images) >= per_class for images in images_by_class.values()):
            break

    missing = [digit for digit, images in images_by_class.items() if len(images) < per_class]
    if missing:
        raise RuntimeError(f"Could not find enough samples for classes: {missing}")

    return images_by_class


def _plot_grid(images_by_class, output_path: Path, split: str) -> None:
    digits = list(range(10))
    per_class = len(next(iter(images_by_class.values())))
    n_cols = 5
    n_rows = 2

    fig, axes = plt.subplots(
        n_rows,
        n_cols * per_class,
        figsize=(2.0 * n_cols * per_class, 4.2),
        constrained_layout=True,
    )

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for index, digit in enumerate(digits):
        row = index // n_cols
        col_block = index % n_cols
        axis = axes[row][col_block * per_class]
        axis.imshow(images_by_class[digit][0].numpy(), cmap="gray")
        axis.axis("off")
        axis.set_title(str(digit), fontsize=11)

        for column in range(1, per_class):
            extra_axis = axes[row][col_block * per_class + column]
            extra_axis.imshow(images_by_class[digit][column].numpy(), cmap="gray")
            extra_axis.axis("off")

    fig.suptitle(f"MNIST {split} samples", fontsize=14)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.per_class < 1:
        raise ValueError("--per-class must be at least 1")

    dataset = _load_dataset(args.data_dir, args.split)
    images_by_class = _collect_samples(dataset, args.per_class)
    _plot_grid(images_by_class, args.output.expanduser().resolve(), args.split)

    print(f"Saved MNIST sample grid to: {args.output.expanduser().resolve()}")


if __name__ == "__main__":
    main()