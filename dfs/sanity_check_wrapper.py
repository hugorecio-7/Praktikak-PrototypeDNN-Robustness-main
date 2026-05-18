"""Sanity-check deterministic parity between B30 and RiskAwareB30Wrapper."""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules import CAEModel_Balanced  # noqa: E402
from dfs.risk_b30_wrapper import RiskAwareB30Wrapper  # noqa: E402


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="Check deterministic B30 wrapper forward-pass parity."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


def load_checkpoint_if_requested(
    model: torch.nn.Module, checkpoint_path: Optional[str], device: torch.device
) -> None:
    if checkpoint_path is None:
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    torch.manual_seed(0)

    base_model = CAEModel_Balanced().to(device)
    load_checkpoint_if_requested(base_model, args.checkpoint, device)
    wrapper = RiskAwareB30Wrapper(base_model, mode="deterministic").to(device)

    base_model.eval()
    wrapper.eval()

    x = torch.randn(args.batch_size, 1, 28, 28, device=device)

    with torch.no_grad():
        base_logits = base_model(x)
        wrapper_logits = wrapper(x)

    abs_diff = (base_logits - wrapper_logits).abs()
    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

    print(f"base logits shape: {tuple(base_logits.shape)}")
    print(f"wrapper logits shape: {tuple(wrapper_logits.shape)}")
    print(f"max absolute difference: {max_abs_diff:.12g}")
    print(f"mean absolute difference: {mean_abs_diff:.12g}")

    assert base_logits.shape == wrapper_logits.shape, (
        f"Shape mismatch: base={tuple(base_logits.shape)}, "
        f"wrapper={tuple(wrapper_logits.shape)}"
    )
    assert max_abs_diff < 1e-6, (
        f"Wrapper logits differ from base logits: max_abs_diff={max_abs_diff}"
    )


if __name__ == "__main__":
    main()
