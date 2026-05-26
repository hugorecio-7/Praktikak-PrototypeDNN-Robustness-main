"""Export a PDF grid with the decoded prototypes of a B30 model.

Usage examples
--------------
    python scripts/plot_b30_prototypes.py --variant B30-FT-E
    python scripts/plot_b30_prototypes.py --model-dir tfg_models/B30/B30-FT-E/seed=1
    python scripts/plot_b30_prototypes.py --checkpoint tfg_models/B30/B30-FT-E/seed=1/checkpoints/B30_B30-FT-E_seed1_best_val_adv_acc.pth
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_ROOT = REPO_ROOT / "analysis"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from config import CKPT_PATHS, DEFAULT_SEED, VARIANTS
def _load_checkpoint_object(checkpoint_path: Path, device: torch.device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def _load_b30_base_model(device: torch.device) -> nn.Module:
    base_path = Path(CKPT_PATHS["B30"])
    model = _load_checkpoint_object(base_path, device)
    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Expected a full B30 model object at {base_path}, got {type(model)!r}."
        )
    return model.to(device).eval()


def _is_base_b30_checkpoint(checkpoint_path: Path) -> bool:
    base_path = Path(CKPT_PATHS["B30"])
    return checkpoint_path.resolve() == base_path.resolve()


def _resolve_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return Path(args.checkpoint).expanduser().resolve()

    if args.model_dir is not None:
        model_dir = Path(args.model_dir).expanduser().resolve()
        summary_path = model_dir / "summary.json"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            best_path = summary.get("best_checkpoint_path")
            if isinstance(best_path, str):
                candidate = model_dir / "checkpoints" / Path(best_path).name
                if candidate.exists():
                    return candidate.resolve()
                candidate = Path(best_path)
                if candidate.exists():
                    return candidate.resolve()

        checkpoint_dir = model_dir / "checkpoints"
        if checkpoint_dir.exists():
            preferred = sorted(checkpoint_dir.glob("*best_val_adv_acc.pth"))
            if preferred:
                return preferred[0].resolve()
            pth_files = sorted(checkpoint_dir.glob("*.pth"))
            if pth_files:
                return pth_files[-1].resolve()

        raise FileNotFoundError(f"No checkpoint found under {model_dir}")

    if args.variant not in CKPT_PATHS:
        valid_variants = ", ".join(sorted(v for v in CKPT_PATHS if v == "B30" or v.startswith("B30-")))
        raise ValueError(f"Unknown B30 variant '{args.variant}'. Available: {valid_variants}")

    return Path(CKPT_PATHS[args.variant]).expanduser().resolve()


def _infer_variant_from_checkpoint(checkpoint_path: Path) -> str | None:
    name = checkpoint_path.name
    if name == Path(CKPT_PATHS["B30"]).name:
        return "B30"

    for variant in VARIANTS["B30"]:
        if variant == "B30":
            continue
        if variant in name:
            return variant

    return None


def _load_b30_model(checkpoint_path: Path, device: torch.device, variant: str | None) -> nn.Module:
    model_obj = _load_checkpoint_object(checkpoint_path, device)
    if isinstance(model_obj, nn.Module):
        return model_obj.to(device).eval()

    if _is_base_b30_checkpoint(checkpoint_path):
        raise TypeError(
            f"Base checkpoint {checkpoint_path} did not contain a full model object."
        )

    base_model = _load_b30_base_model(device)
    if isinstance(model_obj, dict):
        model_state = model_obj.get("model_state", model_obj)
    else:
        model_state = model_obj
    base_model.load_state_dict(model_state)
    return base_model.to(device).eval()


def _decode_prototypes(model: nn.Module) -> torch.Tensor:
    if not hasattr(model, "prototype_layer") or not hasattr(model, "decoder"):
        raise AttributeError("The selected model does not expose prototype_layer/decoder.")

    prototype_vectors = model.prototype_layer.prototype_distances.detach().cpu()
    flat_dim = prototype_vectors.shape[1]

    if hasattr(model.decoder, "decoder") and len(model.decoder.decoder) > 0:
        decoder_in_channels = model.decoder.decoder[0].dconv.in_channels
    else:
        raise AttributeError("Could not infer decoder input channels from the selected model.")

    spatial_area = flat_dim / decoder_in_channels
    spatial_size = int(round(math.sqrt(spatial_area)))
    if decoder_in_channels * spatial_size * spatial_size != flat_dim:
        raise ValueError(
            "Cannot reshape prototype vectors into decoder input tensor: "
            f"flat_dim={flat_dim}, decoder_in_channels={decoder_in_channels}."
        )

    prototype_images = model.decoder(
        prototype_vectors.view(-1, decoder_in_channels, spatial_size, spatial_size)
    ).detach().cpu()

    if prototype_images.dim() == 4 and prototype_images.size(1) == 1:
        return prototype_images[:, 0]
    return prototype_images.squeeze(1)


def _plot_prototypes(prototype_images: torch.Tensor, title: str, output_path: Path, n_cols: int) -> None:
    num_prototypes = prototype_images.shape[0]
    num_rows = math.ceil(num_prototypes / n_cols)

    fig, axes = plt.subplots(
        num_rows,
        n_cols,
        figsize=(2.1 * n_cols, 2.1 * num_rows),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).ravel()

    for index, axis in enumerate(axes_array):
        axis.axis("off")
        if index >= num_prototypes:
            continue
        image = prototype_images[index].numpy()
        axis.imshow(image, cmap="gray", interpolation="nearest")
        axis.set_title(f"P{index}", fontsize=9)

    fig.suptitle(title, fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a PDF grid with the decoded prototypes of a B30 model."
    )
    parser.add_argument(
        "--variant",
        default="B30",
        help="B30 variant name, for example B30, B30-FT-0, B30-FT-E-M.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed used when resolving a model directory.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Path to a B30 run directory such as tfg_models/B30/B30-FT-E/seed=1.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Direct path to a checkpoint file. Overrides --variant and --model-dir.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PDF path. If omitted, a file is written under analysis/figures/.",
    )
    parser.add_argument(
        "--n-cols",
        type=int,
        default=5,
        help="Number of columns in the prototype grid.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = _resolve_checkpoint(args)
    variant = args.variant

    if args.model_dir is not None and args.variant == "B30":
        summary_path = Path(args.model_dir).expanduser().resolve() / "summary.json"
        if summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
            variant = summary.get("variant", variant)
            if isinstance(summary.get("seed"), int):
                args.seed = summary["seed"]
    elif args.variant == "B30":
        inferred_variant = _infer_variant_from_checkpoint(checkpoint_path)
        if inferred_variant is not None:
            variant = inferred_variant

    if args.output is None:
        output_dir = REPO_ROOT / "analysis" / "figures" / "b30_prototypes"
        output_name = f"{variant}_seed{args.seed}_prototypes.pdf"
        output_path = output_dir / output_name
    else:
        output_path = args.output.expanduser().resolve()

    model = _load_b30_model(checkpoint_path, device, variant)
    prototype_images = _decode_prototypes(model)
    title = f"{variant} prototypes"
    title += f"\n{checkpoint_path.name}"

    _plot_prototypes(prototype_images, title, output_path, max(1, args.n_cols))
    print(f"Saved prototype PDF to: {output_path}")


if __name__ == "__main__":
    main()
