from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


# =============================================================================
# Imports from analysis/
# =============================================================================

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

sys.path.append(str(THIS_DIR))
sys.path.append(str(PROJECT_ROOT))

from ch6_pca_trajectories_autoattack import (
    _load_b30,
    get_test_subset,
    pgd_with_trajectory,
    _b30_loss_f,
    autoattack_reference_points,
    PGD_ALPHA,
)

from metric_extractor import extract_internals


# =============================================================================
# Basic helpers
# =============================================================================

def predict(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return model(x).argmax(dim=1)


def linf_per_sample(x_adv: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
    return (x_adv - x_clean).abs().view(x_clean.size(0), -1).max(dim=1).values


def l2_per_sample(x_adv: torch.Tensor, x_clean: torch.Tensor) -> torch.Tensor:
    return (x_adv - x_clean).view(x_clean.size(0), -1).norm(p=2, dim=1)


def latent_l2_per_sample(
    model: torch.nn.Module,
    x_adv: torch.Tensor,
    x_clean: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        z_clean = extract_internals(model, x_clean)["z"]
        z_adv = extract_internals(model, x_adv)["z"]

    return (z_adv - z_clean).view(z_clean.size(0), -1).norm(p=2, dim=1)


def print_bool_array(name: str, value: torch.Tensor) -> None:
    print(f"{name:<22}", value.detach().cpu().numpy())


def print_float_array(name: str, value: torch.Tensor) -> None:
    print(f"{name:<22}", value.detach().cpu().numpy())


# =============================================================================
# Strong Square Attack diagnostic
# =============================================================================

def autoattack_square_strong_reference_points(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    n_queries: int = 20000,
    n_restarts: int = 5,
    batch_size: int = 64,
):
    """
    Run only Square Attack with more queries/restarts.

    This is intended only for debugging AutoAttack warnings such as:

        "Square Attack has decreased the robust accuracy..."

    Do not use this as a final evaluation for only one model unless the same
    configuration is applied consistently to all models.
    """
    from autoattack import AutoAttack

    device = x.device
    model.eval()

    adversary = AutoAttack(
        model,
        norm="Linf",
        eps=float(eps),
        version="standard",
        device=device,
    )

    # Run only Square Attack.
    adversary.attacks_to_run = ["square"]

    # Strengthen Square Attack if these attributes exist in the installed version.
    if hasattr(adversary, "square"):
        if hasattr(adversary.square, "n_queries"):
            adversary.square.n_queries = int(n_queries)
        else:
            print("[WARNING] adversary.square has no attribute n_queries")

        if hasattr(adversary.square, "n_restarts"):
            adversary.square.n_restarts = int(n_restarts)
        else:
            print("[WARNING] adversary.square has no attribute n_restarts")

    x_adv = adversary.run_standard_evaluation(
        x,
        y,
        bs=min(int(batch_size), x.shape[0]),
    )

    with torch.no_grad():
        z_adv = extract_internals(model, x_adv)["z"].cpu().numpy()

    return x_adv, z_adv


# =============================================================================
# Main diagnostic
# =============================================================================

def run_debug(
    model_name: str,
    seed: int,
    n_samples: int,
    eps: float,
    pgd_iters: int,
    auto_version: str,
    strong_square: bool,
    square_n_queries: int,
    square_n_restarts: int,
    batch_size: int,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Seed: {seed}")
    print(f"n_samples: {n_samples}")
    print(f"eps: {eps}")
    print(f"pgd_iters: {pgd_iters}")
    print(f"AutoAttack version: {auto_version}")
    print(f"Strong Square: {strong_square}")
    print(f"Device: {device}")
    print("=" * 80)

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    model = _load_b30(model_name, device)

    x, y = get_test_subset(n_samples=n_samples, seed=seed)
    x = x.to(device)
    y = y.to(device)

    # -------------------------------------------------------------------------
    # Clean predictions
    # -------------------------------------------------------------------------
    with torch.no_grad():
        pred_clean = predict(model, x)
        clean_correct = pred_clean.eq(y)

    print("\nClean predictions")
    print("-----------------")
    print("y_true:              ", y.detach().cpu().numpy())
    print("pred_clean:          ", pred_clean.detach().cpu().numpy())
    print("clean_correct:       ", clean_correct.detach().cpu().numpy())
    print(f"clean_acc_subset = {clean_correct.float().mean().item():.4f}")

    # -------------------------------------------------------------------------
    # PGD diagnostic
    # -------------------------------------------------------------------------
    loss_f = _b30_loss_f(model, y)

    x_pgd, traj = pgd_with_trajectory(
        model=model,
        x=x,
        loss_f=loss_f,
        iters=int(pgd_iters),
        eps=float(eps),
        alpha=PGD_ALPHA,
        device=device,
        extract_fn=extract_internals,
    )

    with torch.no_grad():
        pred_pgd = predict(model, x_pgd)
        pgd_correct = pred_pgd.eq(y)
        pgd_success = clean_correct & (~pgd_correct)

    pgd_linf = linf_per_sample(x_pgd, x)
    pgd_l2 = l2_per_sample(x_pgd, x)
    pgd_z_l2 = latent_l2_per_sample(model, x_pgd, x)

    print("\nPGD diagnostic")
    print("--------------")
    print("pred_pgd:            ", pred_pgd.detach().cpu().numpy())
    print("pgd_correct:         ", pgd_correct.detach().cpu().numpy())
    print("pgd_success:         ", pgd_success.detach().cpu().numpy())
    print("pgd_linf:            ", pgd_linf.detach().cpu().numpy())
    print("pgd_input_l2:        ", pgd_l2.detach().cpu().numpy())
    print("pgd_latent_l2:       ", pgd_z_l2.detach().cpu().numpy())
    print(f"pgd_success_count = {int(pgd_success.sum().item())}/{n_samples}")
    print(f"pgd_zero_input_shift_count = {int((pgd_linf == 0).sum().item())}/{n_samples}")
    print(f"pgd_zero_latent_shift_count = {int((pgd_z_l2 == 0).sum().item())}/{n_samples}")

    # -------------------------------------------------------------------------
    # AutoAttack standard diagnostic
    # -------------------------------------------------------------------------
    x_aa, _ = autoattack_reference_points(
        model=model,
        x=x,
        y=y,
        eps=float(eps),
        version=auto_version,
    )

    with torch.no_grad():
        pred_aa = predict(model, x_aa)
        aa_correct = pred_aa.eq(y)
        aa_success = clean_correct & (~aa_correct)

    aa_linf = linf_per_sample(x_aa, x)
    aa_l2 = l2_per_sample(x_aa, x)
    aa_z_l2 = latent_l2_per_sample(model, x_aa, x)

    print("\nAutoAttack diagnostic")
    print("---------------------")
    print("pred_aa:             ", pred_aa.detach().cpu().numpy())
    print("aa_correct:          ", aa_correct.detach().cpu().numpy())
    print("aa_success:          ", aa_success.detach().cpu().numpy())
    print("aa_linf:             ", aa_linf.detach().cpu().numpy())
    print("aa_input_l2:         ", aa_l2.detach().cpu().numpy())
    print("aa_latent_l2:        ", aa_z_l2.detach().cpu().numpy())
    print(f"aa_success_count = {int(aa_success.sum().item())}/{n_samples}")
    print(f"aa_zero_input_shift_count = {int((aa_linf == 0).sum().item())}/{n_samples}")
    print(f"aa_zero_latent_shift_count = {int((aa_z_l2 == 0).sum().item())}/{n_samples}")

    # -------------------------------------------------------------------------
    # Strong Square Attack diagnostic
    # -------------------------------------------------------------------------
    if strong_square:
        x_sq, _ = autoattack_square_strong_reference_points(
            model=model,
            x=x,
            y=y,
            eps=float(eps),
            n_queries=int(square_n_queries),
            n_restarts=int(square_n_restarts),
            batch_size=int(batch_size),
        )

        with torch.no_grad():
            pred_sq = predict(model, x_sq)
            sq_correct = pred_sq.eq(y)
            sq_success = clean_correct & (~sq_correct)

        sq_linf = linf_per_sample(x_sq, x)
        sq_l2 = l2_per_sample(x_sq, x)
        sq_z_l2 = latent_l2_per_sample(model, x_sq, x)

        print("\nStrong Square Attack diagnostic")
        print("-------------------------------")
        print(f"square_n_queries:    {square_n_queries}")
        print(f"square_n_restarts:   {square_n_restarts}")
        print("pred_square:         ", pred_sq.detach().cpu().numpy())
        print("square_correct:      ", sq_correct.detach().cpu().numpy())
        print("square_success:      ", sq_success.detach().cpu().numpy())
        print("square_linf:         ", sq_linf.detach().cpu().numpy())
        print("square_input_l2:     ", sq_l2.detach().cpu().numpy())
        print("square_latent_l2:    ", sq_z_l2.detach().cpu().numpy())
        print(f"square_success_count = {int(sq_success.sum().item())}/{n_samples}")
        print(f"square_zero_input_shift_count = {int((sq_linf == 0).sum().item())}/{n_samples}")
        print(f"square_zero_latent_shift_count = {int((sq_z_l2 == 0).sum().item())}/{n_samples}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\nSummary")
    print("-------")
    print(f"PGD mean input Linf shift:        {pgd_linf.mean().item():.8f}")
    print(f"PGD mean latent L2 shift:         {pgd_z_l2.mean().item():.8f}")
    print(f"PGD success count:                {int(pgd_success.sum().item())}/{n_samples}")
    print("")
    print(f"AutoAttack mean input Linf shift: {aa_linf.mean().item():.8f}")
    print(f"AutoAttack mean latent L2 shift:  {aa_z_l2.mean().item():.8f}")
    print(f"AutoAttack success count:         {int(aa_success.sum().item())}/{n_samples}")

    if strong_square:
        print("")
        print(f"Square mean input Linf shift:     {sq_linf.mean().item():.8f}")
        print(f"Square mean latent L2 shift:      {sq_z_l2.mean().item():.8f}")
        print(f"Square success count:             {int(sq_success.sum().item())}/{n_samples}")

    print("\nInterpretation hints")
    print("--------------------")
    if torch.all(aa_linf == 0):
        print(
            "[INFO] AutoAttack returned exactly the clean inputs for all selected samples.\n"
            "       This usually means it did not find adversarial examples for this subset,\n"
            "       and the wrapper kept the original inputs as final outputs."
        )

    if int(pgd_success.sum().item()) > 0 and int(aa_success.sum().item()) == 0:
        print(
            "[CHECK] PGD succeeds on at least one clean-correct sample, but AutoAttack succeeds on none.\n"
            "        Check preprocessing, epsilon scale, model.eval(), and AutoAttack wrapper behaviour."
        )

    if strong_square:
        aa_s = int(aa_success.sum().item())
        sq_s = int(sq_success.sum().item())

        if sq_s > aa_s:
            print(
                "[INFO] Strong Square found more adversarial examples than standard AutoAttack output.\n"
                "       This supports the warning that Square Attack may need more queries/restarts."
            )
        elif sq_s == aa_s:
            print(
                "[INFO] Strong Square found the same number of adversarial examples as standard AutoAttack.\n"
                "       The warning does not appear to change the practical conclusion on this subset."
            )
        else:
            print(
                "[INFO] Strong Square found fewer adversarial examples than standard AutoAttack.\n"
                "       This can happen because standard AutoAttack combines several attacks."
            )


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="B30-FT-E-M")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--eps", type=float, default=0.3)
    parser.add_argument("--pgd_iters", type=int, default=80)
    parser.add_argument("--auto_version", type=str, default="standard")

    parser.add_argument("--strong_square", action="store_true")
    parser.add_argument("--square_n_queries", type=int, default=20000)
    parser.add_argument("--square_n_restarts", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()

    run_debug(
        model_name=args.model,
        seed=args.seed,
        n_samples=args.n_samples,
        eps=args.eps,
        pgd_iters=args.pgd_iters,
        auto_version=args.auto_version,
        strong_square=args.strong_square,
        square_n_queries=args.square_n_queries,
        square_n_restarts=args.square_n_restarts,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()