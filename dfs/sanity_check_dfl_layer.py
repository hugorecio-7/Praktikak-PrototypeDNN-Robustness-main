"""Sanity-check tensor shapes, probabilities, and gradients for the DFL layer."""

import argparse

import torch

from dfl_layer import DistributionalPrototypeDistanceLayer


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="Smoke-test DistributionalPrototypeDistanceLayer."
    )
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--latent-dim", type=int, default=40)
    parser.add_argument("--num-prototypes", type=int, default=30)
    parser.add_argument("--num-bins", type=int, default=32)
    parser.add_argument("--d-max", type=float, default=100.0)
    return parser.parse_args()


def assert_all_finite(name: str, tensor: torch.Tensor) -> None:
    assert torch.isfinite(tensor).all(), f"{name} contains NaN or inf values"


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    torch.manual_seed(0)

    z = torch.randn(4, args.latent_dim, device=device)
    prototypes = torch.randn(args.num_prototypes, args.latent_dim, device=device)

    layer = DistributionalPrototypeDistanceLayer(
        latent_dim=args.latent_dim,
        num_bins=args.num_bins,
        d_max=args.d_max,
    ).to(device)

    out = layer(z, prototypes)

    print(f"bin_logits shape: {tuple(out['bin_logits'].shape)}")
    print(f"prob shape: {tuple(out['prob'].shape)}")
    print(f"mu shape: {tuple(out['mu'].shape)}")
    print(f"sigma shape: {tuple(out['sigma'].shape)}")

    expected_logits_shape = (4, args.num_prototypes, args.num_bins)
    expected_summary_shape = (4, args.num_prototypes)

    assert out["bin_logits"].shape == expected_logits_shape
    assert out["prob"].shape == expected_logits_shape
    assert out["mu"].shape == expected_summary_shape
    assert out["sigma"].shape == expected_summary_shape

    prob_sums = out["prob"].sum(dim=-1)
    ones = torch.ones_like(prob_sums)
    assert torch.allclose(prob_sums, ones, atol=1e-5), (
        "probabilities do not sum to 1 within tolerance"
    )

    for name in ("mu", "sigma", "prob", "bin_logits"):
        assert_all_finite(name, out[name])

    assert (out["sigma"] >= 0).all(), "sigma contains negative values"

    loss = out["mu"].mean() + out["sigma"].mean()
    loss.backward()

    has_finite_grad = any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in layer.parameters()
    )
    assert has_finite_grad, "no DFL layer parameter received a finite gradient"


if __name__ == "__main__":
    main()
