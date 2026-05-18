"""Sanity-check DistributionFocalDistanceLoss."""

import argparse

import torch

from dfl_layer import DistributionalPrototypeDistanceLayer
from dfl_losses import DistributionFocalDistanceLoss


def parse_args() -> argparse.Namespace:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    parser = argparse.ArgumentParser(
        description="Smoke-test DistributionFocalDistanceLoss."
    )
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--latent-dim", type=int, default=40)
    parser.add_argument("--num-prototypes", type=int, default=30)
    parser.add_argument("--num-bins", type=int, default=32)
    parser.add_argument("--d-max", type=float, default=100.0)
    return parser.parse_args()


def has_finite_parameter_gradient(module: torch.nn.Module) -> bool:
    return any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    torch.manual_seed(0)

    z = torch.randn(4, args.latent_dim, device=device)
    prototypes = torch.randn(args.num_prototypes, args.latent_dim, device=device)

    dfl_layer = DistributionalPrototypeDistanceLayer(
        latent_dim=args.latent_dim,
        num_bins=args.num_bins,
        d_max=args.d_max,
    ).to(device)
    dfl_loss = DistributionFocalDistanceLoss(
        num_bins=args.num_bins,
        d_max=args.d_max,
        reduction="mean",
    )

    diff = z[:, None, :] - prototypes[None, :, :]
    d_target = diff.pow(2).sum(dim=-1)

    out = dfl_layer(z, prototypes)
    detached_target = d_target.detach()
    loss = dfl_loss(out["bin_logits"], detached_target)
    loss.backward()

    gradients_exist = has_finite_parameter_gradient(dfl_layer)

    print(f"loss value: {loss.item():.12g}")
    print(f"max target distance: {d_target.max().item():.12g}")
    print(f"min target distance: {d_target.min().item():.12g}")
    print(f"gradients exist: {gradients_exist}")

    assert loss.dim() == 0, f"mean reduction should return a scalar, got {loss.shape}"
    assert torch.isfinite(loss), "loss is not finite"
    assert gradients_exist, "no DFL layer parameter received a finite gradient"
    assert not detached_target.requires_grad, "detached target should not require gradients"

    dfl_loss_none = DistributionFocalDistanceLoss(
        num_bins=args.num_bins,
        d_max=args.d_max,
        reduction="none",
    )
    loss_none = dfl_loss_none(out["bin_logits"], detached_target)
    assert loss_none.shape == (4, args.num_prototypes), (
        f"none reduction should return shape (4, {args.num_prototypes}), "
        f"got {tuple(loss_none.shape)}"
    )

    zeros = torch.zeros_like(detached_target)
    maxes = torch.full_like(detached_target, args.d_max)
    zero_loss = dfl_loss(out["bin_logits"], zeros)
    max_loss = dfl_loss(out["bin_logits"], maxes)
    assert torch.isfinite(zero_loss), "zero boundary target produced non-finite loss"
    assert torch.isfinite(max_loss), "d_max boundary target produced non-finite loss"


if __name__ == "__main__":
    main()
