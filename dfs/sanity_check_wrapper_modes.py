"""Sanity-check deterministic, mean, and risk wrapper modes."""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules import CAEModel_Balanced  # noqa: E402
from dfs.dfl_layer import DistributionalPrototypeDistanceLayer  # noqa: E402
from dfs.risk_b30_wrapper import RiskAwareB30Wrapper  # noqa: E402


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="Check RiskAwareB30Wrapper deterministic, mean, and risk modes."
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=default_device)
    return parser.parse_args()


def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item()


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all(), f"{name} contains NaN or inf values"


def has_finite_parameter_gradient(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    torch.manual_seed(0)

    base_model = CAEModel_Balanced().to(device)
    dfl_layer = DistributionalPrototypeDistanceLayer(
        latent_dim=base_model.in_channels_prototype,
        num_bins=32,
        d_max=100.0,
    ).to(device)

    det_wrapper = RiskAwareB30Wrapper(base_model, mode="deterministic").to(device)
    det_wrapper_with_dfl = RiskAwareB30Wrapper(
        base_model,
        mode="deterministic",
        dfl_layer=dfl_layer,
    ).to(device)
    mean_wrapper = RiskAwareB30Wrapper(
        base_model,
        mode="mean",
        dfl_layer=dfl_layer,
    ).to(device)
    risk_wrapper_beta0 = RiskAwareB30Wrapper(
        base_model,
        mode="risk",
        beta=0.0,
        dfl_layer=dfl_layer,
    ).to(device)
    risk_wrapper_beta1 = RiskAwareB30Wrapper(
        base_model,
        mode="risk",
        beta=1.0,
        dfl_layer=dfl_layer,
    ).to(device)

    wrappers = (
        det_wrapper,
        det_wrapper_with_dfl,
        mean_wrapper,
        risk_wrapper_beta0,
        risk_wrapper_beta1,
    )

    base_model.eval()
    for wrapper in wrappers:
        wrapper.eval()

    x = torch.randn(args.batch_size, 1, 28, 28, device=device)

    with torch.no_grad():
        base_logits = base_model(x)
        det_logits = det_wrapper(x)
        det_dfl_logits = det_wrapper_with_dfl(x)
        mean_outputs = mean_wrapper.compute_distance_outputs(x)
        risk0_outputs = risk_wrapper_beta0.compute_distance_outputs(x)
        risk1_outputs = risk_wrapper_beta1.compute_distance_outputs(x)

    det_diff = max_abs_diff(base_logits, det_logits)
    det_dfl_diff = max_abs_diff(base_logits, det_dfl_logits)
    risk0_mean_diff = max_abs_diff(risk0_outputs["logits"], mean_outputs["logits"])
    risk1_d_used_mu_diff = max_abs_diff(risk1_outputs["d_used"], mean_outputs["mu"])

    print(f"base vs deterministic max abs diff: {det_diff:.12g}")
    print(f"base vs deterministic-with-DFL max abs diff: {det_dfl_diff:.12g}")
    print(f"mean logits shape: {tuple(mean_outputs['logits'].shape)}")
    print(f"risk beta=0 vs mean logits max abs diff: {risk0_mean_diff:.12g}")
    print(f"risk beta=1 d_used vs mean mu max abs diff: {risk1_d_used_mu_diff:.12g}")

    assert det_diff < 1e-6, "deterministic wrapper without DFL changed B30 logits"
    assert det_dfl_diff < 1e-6, "deterministic wrapper with DFL changed B30 logits"

    assert mean_outputs["logits"].shape == (args.batch_size, 10)
    assert_finite("mean logits", mean_outputs["logits"])
    assert mean_outputs["mu"].shape == mean_outputs["d_det"].shape
    assert mean_outputs["sigma"].shape == mean_outputs["d_det"].shape

    assert risk0_mean_diff < 1e-6, "risk beta=0 should match mean logits exactly"

    assert risk1_outputs["logits"].shape == (args.batch_size, 10)
    assert_finite("risk beta=1 logits", risk1_outputs["logits"])
    assert risk1_outputs["d_used"].shape == risk1_outputs["d_det"].shape

    dfl_layer.zero_grad(set_to_none=True)
    mean_wrapper.train()
    logits = mean_wrapper(x)
    loss = logits.mean()
    loss.backward()

    assert has_finite_parameter_gradient(dfl_layer), (
        "no DFL layer parameter received a finite gradient in mean mode"
    )


if __name__ == "__main__":
    main()
