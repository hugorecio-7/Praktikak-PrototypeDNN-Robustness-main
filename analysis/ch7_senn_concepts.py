"""
Create a Chapter 7 PDF comparing the concepts learned by multiple SENN models.

Edit MODELS below, or override them from the command line:

    python analysis/ch7_senn_concepts.py --models SENN_0_01 SENN-FT-1

    python analysis/ch7_senn_concepts.py --models SENN_0_01 SENN-FT-1 SENN-FT-2 --adv
    python analysis/ch7_senn_concepts.py --models SENN_0_01 SENN-FT-1 --adv --eps 0.4

The script saves the generated PDF under analysis/figures/ch7 by default.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms


# =============================================================================
# User-editable defaults
# =============================================================================

MODELS = ("SENN_0_01", "SENN-FT-1")
SAMPLE_INDEX = 0
DIGIT: int | None = None
ADV_EPS_DEFAULT = 0.3
PGD_ITERS_DEFAULT = 80
PGD_ALPHA_DEFAULT = 0.01


# =============================================================================
# Repository paths and imports
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_ROOT = PROJECT_ROOT / "analysis"

for path in (PROJECT_ROOT, ANALYSIS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import CKPT_PATHS, FIGURES_ROOT, RC_PARAMS, VARIANT_COLORS  # noqa: E402
from SENN.models.aggregators import SumAggregator  # noqa: E402
from SENN.models.conceptizers import ConvConceptizer  # noqa: E402
from SENN.models.parameterizers import ConvParameterizer  # noqa: E402
from SENN.models.senn import SENN  # noqa: E402


CONSTRUCTOR_REGISTRY = {
    "ConvConceptizer": ConvConceptizer,
    "ConvParameterizer": ConvParameterizer,
    "SumAggregator": SumAggregator,
}


@dataclass
class SennExplanation:
    name: str
    concepts: np.ndarray
    relevances: np.ndarray
    contributions: np.ndarray
    reconstruction: np.ndarray
    predicted_class: int
    confidence: float


def resolve_repo_path(path_or_name: str) -> Path:
    """Resolve a known model name or a filesystem path into an absolute path."""
    raw_path = CKPT_PATHS.get(path_or_name, path_or_name)
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")
    return path


def load_senn_config(device: torch.device) -> SimpleNamespace:
    """Load the base SENN architecture config used by all SENN variants."""
    cfg_path = PROJECT_ROOT / CKPT_PATHS["SENN_0_01_config"]
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["device"] = str(device)
    return SimpleNamespace(**cfg)


def instantiate_senn(device: torch.device) -> SENN:
    """Instantiate the SENN architecture from its JSON config."""
    cfg = load_senn_config(device)

    try:
        conceptizer_cls = CONSTRUCTOR_REGISTRY[cfg.conceptizer]
        parameterizer_cls = CONSTRUCTOR_REGISTRY[cfg.parameterizer]
        aggregator_cls = CONSTRUCTOR_REGISTRY[cfg.aggregator]
    except KeyError as exc:
        known = ", ".join(sorted(CONSTRUCTOR_REGISTRY))
        raise ValueError(f"Unknown SENN module '{exc.args[0]}'. Known modules: {known}") from exc

    conceptizer = conceptizer_cls(**cfg.__dict__)
    parameterizer = parameterizer_cls(**cfg.__dict__)
    aggregator = aggregator_cls(**cfg.__dict__)
    return SENN(conceptizer, parameterizer, aggregator).to(device)


def extract_model_state(checkpoint: object) -> dict[str, torch.Tensor]:
    """Support both raw state_dict checkpoints and dict checkpoints with model_state."""
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"]
    if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise ValueError(
        "Unsupported SENN checkpoint format. Expected a state_dict or a dict with key 'model_state'."
    )


def load_senn_model(model_name_or_path: str, device: torch.device) -> SENN:
    """Load a SENN model from one of the registered names or from an explicit path."""
    checkpoint_path = resolve_repo_path(model_name_or_path)
    model = instantiate_senn(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(extract_model_state(checkpoint))
    model.eval()
    return model


def load_mnist_sample(
    data_dir: Path,
    sample_index: int,
    digit: int | None,
    download: bool,
) -> tuple[torch.Tensor, int, int]:
    """Return one MNIST image tensor, its label, and the selected dataset index."""
    dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=download,
        transform=transforms.ToTensor(),
    )

    if digit is None:
        selected_index = sample_index
    else:
        if not 0 <= digit <= 9:
            raise ValueError("--digit must be between 0 and 9.")
        matching_indices = [idx for idx, (_, label) in enumerate(dataset) if int(label) == digit]
        if not matching_indices:
            raise ValueError(f"No MNIST test samples found for digit {digit}.")
        selected_index = matching_indices[sample_index % len(matching_indices)]

    image, label = dataset[selected_index]
    return image, int(label), int(selected_index)


def to_senn_input(image: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert an MNIST tensor in [0, 1] to the SENN input range [-1, 1]."""
    return image.unsqueeze(0).to(device) * 2.0 - 1.0


def batch_to_senn_input(batch_x: torch.Tensor) -> torch.Tensor:
    """Convert a raw image batch in [0, 1] to the SENN input range [-1, 1]."""
    return batch_x * 2.0 - 1.0


def image_for_display(image: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert a possibly [-1, 1] image into a clipped [0, 1] numpy array."""
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    image = np.asarray(image).squeeze()
    if image.min() < 0.0:
        image = (image + 1.0) / 2.0
    return np.clip(image, 0.0, 1.0)


@torch.no_grad()
def explain_model(name: str, model: SENN, image: torch.Tensor, device: torch.device) -> SennExplanation:
    """Extract concept values, relevances, and predicted-class contributions."""
    x = to_senn_input(image, device)
    concepts, reconstruction = model.conceptizer(x)
    relevances = model.parameterizer(x)
    log_probs = model.aggregator(concepts, relevances)

    probs = log_probs.exp().squeeze(0)
    predicted_class = int(probs.argmax().item())
    confidence = float(probs[predicted_class].item())

    concept_values = concepts.squeeze(0).squeeze(-1).detach().cpu().numpy()
    predicted_relevances = relevances.squeeze(0)[:, predicted_class].detach().cpu().numpy()
    contributions = concept_values * predicted_relevances

    return SennExplanation(
        name=name,
        concepts=concept_values,
        relevances=predicted_relevances,
        contributions=contributions,
        reconstruction=image_for_display(reconstruction),
        predicted_class=predicted_class,
        confidence=confidence,
    )


def make_pgd_adversarial_image(
    model: SENN,
    image: torch.Tensor,
    label: int,
    eps: float,
    iters: int,
    alpha: float,
    random_start: bool,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a PGD-Linf adversarial image in [0, 1].

    The repository PGD helper attacks raw MNIST inputs in [0, 1], while SENN
    internally expects [-1, 1]. The loss wrapper mirrors run_metrics.py by
    converting the attack batch before forwarding through SENN.
    """
    from adversarial_attacks import PGDLInf_attack

    model.eval()
    batch_x = image.unsqueeze(0).to(device)
    target = torch.tensor([label], dtype=torch.long, device=device)

    def loss_f(*, batch_x: torch.Tensor) -> torch.Tensor:
        log_probs = model(batch_to_senn_input(batch_x))
        return F.nll_loss(log_probs, target)

    adv_batch = PGDLInf_attack(
        batch_x=batch_x,
        loss_f=loss_f,
        iters=iters,
        eps=eps,
        alpha=alpha,
        random_start=random_start,
    )
    return adv_batch.squeeze(0).detach().cpu()


def symmetric_limit(values: list[np.ndarray], pad: float = 1.15) -> tuple[float, float]:
    """Return a centered x-limit that covers all values."""
    max_abs = max(float(np.max(np.abs(v))) for v in values)
    if max_abs == 0.0:
        max_abs = 1.0
    return -max_abs * pad, max_abs * pad


def draw_bar_panel(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    xlim: tuple[float, float],
    color: str | list[str],
) -> None:
    """Draw a horizontal bar panel with concept labels."""
    y = np.arange(len(values))
    ax.barh(y, values, color=color, edgecolor="none")
    ax.axvline(0.0, color="#222222", linewidth=0.8)
    ax.set_yticks(y, [f"C{i + 1}" for i in y])
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_title(title)
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, alpha=0.8)


def sanitize_filename(value: str) -> str:
    """Create a compact filesystem-safe token."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    model_names: list[str],
    sample_index: int,
    suffix: str,
) -> Path:
    """Save the figure as a PDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    models_token = "_vs_".join(sanitize_filename(name) for name in model_names)
    stem = f"SENN_concepts_{models_token}_sample{sample_index}{suffix}"
    path = output_dir / f"{stem}.pdf"
    fig.savefig(path, bbox_inches="tight")
    return path


def plot_comparison(
    row_images: list[torch.Tensor],
    label: int,
    dataset_index: int,
    explanations: list[SennExplanation],
    output_dir: Path,
    input_description: str,
    filename_suffix: str,
) -> Path:
    """Build and save the final side-by-side SENN concept comparison."""
    plt.rcParams.update(RC_PARAMS)

    concept_xlim = symmetric_limit([exp.concepts for exp in explanations])
    relevance_xlim = symmetric_limit([exp.relevances for exp in explanations])
    contribution_xlim = symmetric_limit([exp.contributions for exp in explanations])

    fig, axes = plt.subplots(
        nrows=len(explanations),
        ncols=5,
        figsize=(14.0, max(3.0, 2.7 * len(explanations))),
        gridspec_kw={"width_ratios": [0.9, 0.9, 1.45, 1.45, 1.45]},
        constrained_layout=True,
    )
    if len(explanations) == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, (exp, row_image) in enumerate(zip(explanations, row_images)):
        input_image = image_for_display(row_image)
        model_color = VARIANT_COLORS.get(exp.name, "#1f77b4")
        contribution_colors = ["#2ca02c" if value >= 0 else "#d62728" for value in exp.contributions]

        axes[row, 0].imshow(input_image, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title("Input")
        axes[row, 0].set_ylabel(exp.name, rotation=0, ha="right", va="center", labelpad=42, fontweight="bold")
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])

        axes[row, 1].imshow(exp.reconstruction, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title("Reconstruction")
        axes[row, 1].set_xticks([])
        axes[row, 1].set_yticks([])

        draw_bar_panel(axes[row, 2], exp.concepts, "Concept values", concept_xlim, model_color)
        draw_bar_panel(axes[row, 3], exp.relevances, f"Relevance for class {exp.predicted_class}", relevance_xlim, model_color)
        draw_bar_panel(axes[row, 4], exp.contributions, "Predicted-class contribution", contribution_xlim, contribution_colors)

        axes[row, 0].text(
            0.5,
            -0.12,
            f"true={label} | pred={exp.predicted_class} | p={exp.confidence:.2f}",
            transform=axes[row, 0].transAxes,
            ha="center",
            va="top",
            fontsize=8,
        )

    fig.suptitle(
        f"SENN concept comparison on MNIST test sample {dataset_index} ({input_description})",
        fontsize=13,
        fontweight="bold",
    )

    return save_figure(
        fig=fig,
        output_dir=output_dir,
        model_names=[exp.name for exp in explanations],
        sample_index=dataset_index,
        suffix=filename_suffix,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare concepts from multiple SENN models.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS),
        help="SENN model names or checkpoint paths to compare.",
    )
    parser.add_argument("--sample-index", type=int, default=SAMPLE_INDEX, help="MNIST test sample index.")
    parser.add_argument("--digit", type=int, default=DIGIT, help="Optional digit filter from 0 to 9.")
    parser.add_argument("--adv", action="store_true", help="Use a PGD-Linf adversarial version of the sample.")
    parser.add_argument("--eps", type=float, default=ADV_EPS_DEFAULT, help="PGD epsilon used when --adv is active.")
    parser.add_argument("--pgd-iters", type=int, default=PGD_ITERS_DEFAULT, help="PGD iterations used with --adv.")
    parser.add_argument("--pgd-alpha", type=float, default=PGD_ALPHA_DEFAULT, help="PGD step size used with --adv.")
    parser.add_argument(
        "--no-random-start",
        action="store_true",
        help="Disable random PGD initialization when --adv is active.",
    )
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data", help="MNIST data root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(FIGURES_ROOT) / "ch7",
        help="Directory where the figure will be saved.",
    )
    parser.add_argument("--download-mnist", action="store_true", help="Download MNIST if it is not already present.")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Device used to run the SENN models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_names = list(args.models)
    if len(model_names) < 2:
        raise ValueError("Pass at least two SENN models with --models to make a comparison.")

    device = torch.device(args.device)
    if args.adv and torch.cuda.is_available() and device.type != "cuda":
        # PGDLInf_attack in this repository moves attack batches to CUDA when it
        # is available, so keep the models on the same device in adversarial mode.
        device = torch.device("cuda")

    image, label, dataset_index = load_mnist_sample(
        data_dir=args.data_dir,
        sample_index=args.sample_index,
        digit=args.digit,
        download=args.download_mnist,
    )

    models = [load_senn_model(name, device) for name in model_names]

    input_description = "clean input"
    filename_suffix = "_clean"
    row_images = [image for _ in models]

    if args.adv:
        row_images = [
            make_pgd_adversarial_image(
                model=model,
                image=image,
                label=label,
                eps=args.eps,
                iters=args.pgd_iters,
                alpha=args.pgd_alpha,
                random_start=not args.no_random_start,
                device=device,
            )
            for model in models
        ]
        eps_token = str(args.eps).replace(".", "p")
        input_description = f"PGD-Linf adversarial input, eps={args.eps}, attacked per model"
        filename_suffix = f"_adv_eps{eps_token}_per_model"

    explanations = [
        explain_model(name, model, row_image, device)
        for name, model, row_image in zip(model_names, models, row_images)
    ]

    saved_path = plot_comparison(
        row_images=row_images,
        label=label,
        dataset_index=dataset_index,
        explanations=explanations,
        output_dir=args.output_dir,
        input_description=input_description,
        filename_suffix=filename_suffix,
    )

    print(f"Saved: {saved_path}")


if __name__ == "__main__":
    main()
