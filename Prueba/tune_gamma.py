"""Small gamma sweep for the Reconstruction-Risk Shield on B30."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_loader import get_test_loader
from metric_calculators import calc_artificial_prototype_match_suppression
from modules import CAEModel_Balanced
from reconstruction_risk_wrapper import ReconstructionRiskWrapper

try:
    from adversarial_attacks import PGDLInf_attack as repo_pgd_linf_attack
except Exception:  # pragma: no cover - fallback for minimal test environments.
    repo_pgd_linf_attack = None


EPSILONS = [0.0, 0.025, 0.05, 0.1]
GAMMA_VALUES = [0.0, 10.0, 100.0, 1000.0]
POWERS = [2, 3]
OUTPUT_DIR = Path(__file__).resolve().parent

B30_MODEL_PATHS = {
    "B30": (
        "saved_model/mnist_model/"
        "mnist_cae_balanced_clstsep_1500_0.002_250_True_0.0_20_1_1_1_"
        "1.0_0.0_30_4_32_1/mnist_cae00750.pth"
    ),
    "RB30": (
        "saved_model/mnist_model/"
        "mnist_cae_adversarial_balanced_clstsep_pdglinf_ce_20_0.3_0.02_"
        "True_800_0.002_250_True_0.0_20_1_1_1_1.0_0.0_1.0_30_4_32_1/"
        "mnist_cae_adv00750.pth"
    ),
    "FTB30n": (
        "saved_model/mnist_model/"
        "mnist_cae_FT_30_nothing_pdglinf_ce_20_0.3_0.02_True_20_0.002_"
        "250_20_1_1_1_1.0_0.0_1/mnist_cae_adv00020.pth"
    ),
    "FTB30p": (
        "saved_model/mnist_model/"
        "mnist_cae_FT_30_prototypes_pdglinf_ce_20_0.3_0.02_True_20_0.002_"
        "250_20_1_1_1_1.0_0.0_1/mnist_cae_adv00020.pth"
    ),
    "B30-FT-0": (
        "tfg_models/B30/B30-FT-0/seed=1/checkpoints/"
        "B30_B30-FT-0_seed1_best_val_adv_acc.pth"
    ),
    "B30-FT-1": (
        "tfg_models/B30/B30-FT-1/seed=1/checkpoints/"
        "B30_B30-FT-1_seed1_best_val_adv_acc.pth"
    ),
    "B30-FT-2": (
        "tfg_models/B30/B30-FT-2/seed=1/checkpoints/"
        "B30_B30-FT-2_seed1_best_val_adv_acc.pth"
    ),
    "B30-FT-3": (
        "tfg_models/B30/B30-FT-3/seed=1/checkpoints/"
        "B30_B30-FT-3_seed1_best_val_adv_acc.pth"
    ),
    "B30-FT-E": (
        "tfg_models/B30/B30-FT-E/seed=1/checkpoints/"
        "B30_B30-FT-E_seed1_best_val_adv_acc.pth"
    ),
    "B30-FT-E-M": (
        "tfg_models/B30/B30-FT-E-M/seed=1/checkpoints/"
        "B30_B30-FT-E_seed1_best_val_adv_acc.pth"
    ),
}


def torch_load(path: Path, device: torch.device) -> Any:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Power values must be >= 1.")
    return parsed


def extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor] | None:
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
        if all(torch.is_tensor(value) for value in checkpoint.values()):
            return checkpoint
    return None


def resolve_model_path(model_name: str) -> Path:
    try:
        relative_path = B30_MODEL_PATHS[model_name]
    except KeyError as exc:
        available = ", ".join(sorted(B30_MODEL_PATHS))
        raise ValueError(
            f"Unknown B30 model {model_name!r}. Available models: {available}."
        ) from exc
    return ROOT / relative_path


def load_b30_model_from_path(model_path: Path, device: torch.device) -> CAEModel_Balanced:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model path not found: {model_path}. Check the path registry in "
            "B30_MODEL_PATHS."
        )

    checkpoint = torch_load(model_path, device)
    if isinstance(checkpoint, CAEModel_Balanced):
        model = checkpoint
    else:
        model = CAEModel_Balanced().to(device)
        state_dict = extract_state_dict(checkpoint)
        if state_dict is None:
            raise ValueError(
                "Unsupported checkpoint format. Expected a full "
                "CAEModel_Balanced object or a dict with model_state/state_dict."
            )
        model.load_state_dict(state_dict)

    model = model.to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_b30_model(model_name: str, device: torch.device) -> CAEModel_Balanced:
    model_path = resolve_model_path(model_name)

    if model_name.startswith("B30-FT-"):
        base_model = load_b30_model_from_path(resolve_model_path("B30"), device)
        checkpoint = torch_load(model_path, device)
        state_dict = extract_state_dict(checkpoint)
        if state_dict is None:
            raise ValueError(
                f"Unsupported checkpoint format for {model_name}. Expected a "
                "dict with model_state/state_dict."
            )
        base_model.load_state_dict(state_dict)
        for param in base_model.parameters():
            param.requires_grad_(False)
        return base_model.to(device).eval()

    return load_b30_model_from_path(model_path, device)


def format_float_for_filename(value: float) -> str:
    formatted = f"{value:g}".replace(".", "e").replace("-", "m")
    return formatted


def output_csv_path(
    model_name: str,
    epsilons: list[float],
    powers: list[int],
    grey: bool,
) -> Path:
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    max_epsilon = format_float_for_filename(max(epsilons))
    max_power = max(powers)
    grey_tag = "grey" if grey else "nogrey"
    filename = (
        f"gamma_tuning_results_{safe_model_name}_"
        f"{max_epsilon}_{max_power}_{grey_tag}.csv"
    )
    return OUTPUT_DIR / filename


def build_proto_labels(
    base_model: CAEModel_Balanced,
    device: torch.device,
) -> torch.Tensor:
    n_prototypes = base_model.prototype_layer.prototype_distances.shape[0]
    n_classes = base_model.fc.linear.out_features
    if n_prototypes % n_classes != 0:
        raise ValueError(
            f"Expected balanced prototypes, got {n_prototypes} prototypes for "
            f"{n_classes} classes."
        )
    prototypes_by_class = n_prototypes // n_classes
    return torch.arange(n_classes, device=device).repeat_interleave(prototypes_by_class)


def collect_validation_batch(
    data_dir: Path,
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    test_loader = get_test_loader(
        str(data_dir),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    images, labels = next(iter(test_loader))
    return images.to(device), labels.to(device)


def simple_pgd_linf_attack(
    batch_x: torch.Tensor,
    loss_f,
    iters: int,
    eps: float,
    alpha: float,
    random_start: bool,
) -> torch.Tensor:
    original = batch_x.detach()
    perturbed = original.clone()

    if random_start and eps > 0:
        perturbed = perturbed + torch.empty_like(perturbed).uniform_(-eps, eps)
        perturbed = torch.clamp(perturbed, min=0.0, max=1.0).detach()

    for _ in range(iters):
        perturbed.requires_grad_(True)
        loss = loss_f(batch_x=perturbed)
        gradients = torch.autograd.grad(loss, perturbed)[0]

        perturbed = perturbed.detach() + alpha * gradients.sign()
        delta = torch.clamp(perturbed - original, min=-eps, max=eps)
        perturbed = torch.clamp(original + delta, min=0.0, max=1.0).detach()

    return perturbed


def run_pgd_attack(
    wrapper: ReconstructionRiskWrapper,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float,
    iters: int,
    alpha: float,
    random_start: bool,
    grey: bool,
) -> torch.Tensor:
    if epsilon == 0.0:
        return images

    def loss_f(batch_x: torch.Tensor) -> torch.Tensor:
        logits = wrapper.base_model(batch_x) if grey else wrapper(batch_x)
        return F.cross_entropy(logits, labels)

    attack = repo_pgd_linf_attack or simple_pgd_linf_attack
    return attack(
        batch_x=images,
        loss_f=loss_f,
        iters=iters,
        eps=epsilon,
        alpha=alpha,
        random_start=random_start,
    )


def evaluate_batch(
    wrapper: ReconstructionRiskWrapper,
    images: torch.Tensor,
    labels: torch.Tensor,
    proto_labels: torch.Tensor,
) -> tuple[float, float, float]:
    with torch.no_grad():
        logits = wrapper(images)
        predictions = logits.argmax(dim=1)
        accuracy = predictions.eq(labels).float().mean().item()

        metric_outputs = calc_artificial_prototype_match_suppression(
            wrapper.last_d_latente,
            wrapper.last_d_final,
            labels,
            proto_labels,
        )

        # Calcular el APMSR verdadero (supresiones sobre el total de fallos latentes)
        false_matches = metric_outputs["false_match_mu"].sum().item()
        suppressions = metric_outputs["suppressed_match"].sum().item()
        harm_matches = metric_outputs["risk_harm_match"].sum().item()

        if false_matches > 0:
            apm_sr = (suppressions / false_matches) * 100.0
        else:
            apm_sr = 0.0
        harm_rate = (harm_matches / labels.size(0)) * 100.0

    return accuracy, apm_sr, harm_rate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune gamma for the Reconstruction-Risk Shield on B30."
    )
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",
        default=["B30"],
        choices=sorted(B30_MODEL_PATHS),
        help="One or more B30 model names from the local path registry.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--pgd-iters", type=int, default=80)
    parser.add_argument("--pgd-alpha", type=float, default=0.01)
    parser.add_argument("--no-random-start", action="store_true")
    parser.add_argument(
        "--grey",
        action="store_true",
        help="Grey-box PGD: attack the matching non-Shield model instead of the wrapper.",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=EPSILONS,
        help="Epsilon values to evaluate, e.g. --epsilons 0 0.025 0.05 0.1.",
    )
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=GAMMA_VALUES,
        help="Gamma values to evaluate, e.g. --gammas 0 10 100 1000.",
    )
    parser.add_argument(
        "--powers",
        type=positive_int,
        nargs="+",
        default=POWERS,
        help="Structural penalty powers, e.g. --powers 2 3.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, labels = collect_validation_batch(args.data_dir, args.batch_size, device)

    print(
        "Sweep config | "
        f"models={args.model} | "
        f"epsilons={args.epsilons} | "
        f"gammas={args.gammas} | "
        f"powers={args.powers} | "
        f"grey={args.grey}"
    )

    for model_name in args.model:
        base_model = load_b30_model(model_name, device)
        proto_labels = build_proto_labels(base_model, device)

        print(f"Loaded model {model_name} from {resolve_model_path(model_name)}")

        rows = []
        for power in args.powers:
            for gamma in args.gammas:
                wrapper = ReconstructionRiskWrapper(
                    base_model,
                    gamma=gamma,
                    power=power,
                ).to(device).eval()

                for epsilon in args.epsilons:
                    eval_images = run_pgd_attack(
                        wrapper=wrapper,
                        images=images,
                        labels=labels,
                        epsilon=epsilon,
                        iters=args.pgd_iters,
                        alpha=args.pgd_alpha,
                        random_start=not args.no_random_start,
                        grey=args.grey,
                    )
                    accuracy, suppressed_matches, harm_rate = evaluate_batch(
                        wrapper=wrapper,
                        images=eval_images,
                        labels=labels,
                        proto_labels=proto_labels,
                    )

                    row = {
                        "model": model_name,
                        "power": power,
                        "gamma": gamma,
                        "epsilon": epsilon,
                        "accuracy": accuracy,
                        "suppressed_matches": suppressed_matches,
                        "harm_rate": harm_rate,
                    }
                    rows.append(row)
                    print(
                        f"Model={model_name} | Power={power} | "
                        f"Gamma={gamma:.2f} | "
                        f"Epsilon={epsilon:.3f} | Accuracy={accuracy:.4f} | "
                        f"suppressed_matches={suppressed_matches:.2f}% | "
                        f"harm_rate={harm_rate:.2f}%"
                    )

        model_output_csv = output_csv_path(
            model_name=model_name,
            epsilons=args.epsilons,
            powers=args.powers,
            grey=args.grey,
        )
        with model_output_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "model",
                    "power",
                    "gamma",
                    "epsilon",
                    "accuracy",
                    "suppressed_matches",
                    "harm_rate",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"Results for {model_name} saved to {model_output_csv}")


if __name__ == "__main__":
    main()
