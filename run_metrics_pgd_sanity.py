"""
run_metrics_pgd_sanity.py
=========================

Usage examples
--------------
python run_metrics_pgd_sanity.py \
  --models B30-FT-E \
  --attacks PGD_R10_attack PGD_Long_attack PGD_Scaled8_attack PGD_Scaled16_attack \
  --max_eps 0.4 \
  --step 0.025 \
  --seed 1 \
  --run_id fte_pgd_sanity

python run_metrics_pgd_sanity.py \
  --models B30-FT-E B30-FT-E-M B30-FT-0 \
  --attacks PGD_R10_attack PGD_Long_attack PGD_Scaled8_attack PGD_Scaled16_attack \
  --max_eps 0.4 \
  --step 0.025 \
  --seed 1 \
  --run_id fte_vs_fte_m_pgd_sanity
"""

import argparse
import json
import random
from datetime import datetime
from functools import partial
from types import SimpleNamespace

try:
    import numpy as np
    import torch
    import torch.nn as nn

    from data_loader import *
    from loss_functions import *

    from SENN.models.senn import SENN
    from SENN.models.conceptizers import ConvConceptizer
    from SENN.models.parameterizers import ConvParameterizer
    from SENN.models.aggregators import SumAggregator
    from ProtoVAE import model as model_protovae

    from model_testing_metrics import adversarial_metrics_eps_collect
except ModuleNotFoundError as exc:
    RUNTIME_IMPORT_ERROR = exc

    class _MissingNN:
        class Module:
            pass

    np = None
    torch = None
    nn = _MissingNN()
else:
    RUNTIME_IMPORT_ERROR = None


paths = {
    "B30": "saved_model/mnist_model/mnist_cae_balanced_clstsep_1500_0.002_250_True_0.0_20_1_1_1_1.0_0.0_30_4_32_1/mnist_cae00750.pth",
    "S30": "saved_model/mnist_model/mnist_cae_standard_default_1500_0.002_250_False_0.5_20_1_1_1_0.8_0.2_30_4_32_1/mnist_cae00750.pth",
    "RS30": "saved_model/mnist_model/mnist_cae_adversarial_standard_default_pdglinf_ce_20_0.3_0.02_True_1500_0.002_250_False_0.5_20_1_1_1_0.8_0.2_1.0_30_4_32_1/mnist_cae_adv00750.pth",
    "RB30": "saved_model/mnist_model/mnist_cae_adversarial_balanced_clstsep_pdglinf_ce_20_0.3_0.02_True_800_0.002_250_True_0.0_20_1_1_1_1.0_0.0_1.0_30_4_32_1/mnist_cae_adv00750.pth",
    "FTB30n": "saved_model/mnist_model/mnist_cae_FT_30_nothing_pdglinf_ce_20_0.3_0.02_True_20_0.002_250_20_1_1_1_1.0_0.0_1/mnist_cae_adv00020.pth",
    "FTB30p": "saved_model/mnist_model/mnist_cae_FT_30_prototypes_pdglinf_ce_20_0.3_0.02_True_20_0.002_250_20_1_1_1_1.0_0.0_1/mnist_cae_adv00020.pth",

    "B30-FT-0": "tfg_models/B30/B30-FT-0/seed=1/checkpoints/B30_B30-FT-0_seed1_best_val_adv_acc.pth",
    "B30-FT-1": "tfg_models/B30/B30-FT-1/seed=1/checkpoints/B30_B30-FT-1_seed1_best_val_adv_acc.pth",
    "B30-FT-2": "tfg_models/B30/B30-FT-2/seed=1/checkpoints/B30_B30-FT-2_seed1_best_val_adv_acc.pth",
    "B30-FT-3": "tfg_models/B30/B30-FT-3/seed=1/checkpoints/B30_B30-FT-3_seed1_best_val_adv_acc.pth",
    "B30-FT-E": "tfg_models/B30/B30-FT-E/seed=1/checkpoints/B30_B30-FT-E_seed1_best_val_adv_acc.pth",
    "B30-FT-E-M": "tfg_models/B30/B30-FT-E-M/seed=1/checkpoints/B30_B30-FT-E_seed1_best_val_adv_acc.pth",

    "SENN_0_01": "SENN/results/mnist_lambda1e-2_seed29/checkpoints/best_model.pt",
    "SENN-FT-0": "tfg_models/SENN_0_01/SENN-FT-0/seed=1/checkpoints/SENN_0_01_SENN-FT-0_seed1_best_val_adv_acc.pth",
    "SENN-FT-1": "tfg_models/SENN_0_01/SENN-FT-1/seed=1/checkpoints/SENN_0_01_SENN-FT-1_seed1_best_val_adv_acc.pth",
    "SENN-FT-2": "tfg_models/SENN_0_01/SENN-FT-2/seed=1/checkpoints/SENN_0_01_SENN-FT-2_seed1_best_val_adv_acc.pth",

    "ProtoVAE": "ProtoVAE/saved_models/mnist/model.pth",
    "ProtoVAE-FT-0": "tfg_models/ProtoVAE/ProtoVAE-FT-0/seed=1/checkpoints/ProtoVAE_ProtoVAE-FT-0_seed1_best_val_adv_acc.pth",
    "ProtoVAE-FT-1": "tfg_models/ProtoVAE/ProtoVAE-FT-1/seed=1/checkpoints/ProtoVAE_ProtoVAE-FT-1_seed1_best_val_adv_acc.pth",
    "ProtoVAE-FT-2": "tfg_models/ProtoVAE/ProtoVAE-FT-2/seed=1/checkpoints/ProtoVAE_ProtoVAE-FT-2_seed1_best_val_adv_acc.pth",
    "ProtoVAE-FT-3": "tfg_models/ProtoVAE/ProtoVAE-FT-3/seed=1/checkpoints/ProtoVAE_ProtoVAE-FT-3_seed1_best_val_adv_acc.pth",
    "ProtoVAE-FT-E": "tfg_models/ProtoVAE/ProtoVAE-FT-E/seed=1/checkpoints/ProtoVAE_ProtoVAE-FT-E_seed1_best_val_adv_acc.pth",
}

config_paths = {
    "SENN_0_01": "SENN/configs/mnist_lambda1e-2_seed29.json",
    "SENN-FT-0": "SENN/configs/mnist_lambda1e-2_seed29.json",
    "SENN-FT-1": "SENN/configs/mnist_lambda1e-2_seed29.json",
    "SENN-FT-2": "SENN/configs/mnist_lambda1e-2_seed29.json",
}


def pgd_linf_restarts(
    batch_x,
    loss_f,
    eps,
    iters,
    alpha=None,
    alpha_divisor=None,
    random_start=True,
    restarts=10,
    clamp_min=0.0,
    clamp_max=1.0,
):
    if eps == 0:
        return batch_x.clone().detach()

    if alpha_divisor is not None:
        if alpha_divisor == 0:
            raise ValueError("alpha_divisor must be non-zero.")
        alpha_value = eps / alpha_divisor
    else:
        alpha_value = alpha

    if alpha_value is None:
        raise ValueError("PGD step size is undefined: provide alpha or alpha_divisor.")
    if restarts < 1:
        raise ValueError("restarts must be >= 1.")

    ori_images = batch_x.clone().detach()
    best_perturbed_batch_x = None
    best_loss = -float("inf")

    for _ in range(restarts):
        perturbed = ori_images.clone().detach()

        if random_start:
            perturbed = perturbed + torch.empty_like(perturbed).uniform_(-eps, eps)
            perturbed = torch.clamp(perturbed, min=clamp_min, max=clamp_max).detach()

        for _ in range(iters):
            perturbed.requires_grad_(True)
            loss = loss_f(batch_x=perturbed)
            grad = torch.autograd.grad(loss, perturbed)[0]

            perturbed = perturbed.detach() + alpha_value * torch.sign(grad)
            delta = torch.clamp(perturbed - ori_images, min=-eps, max=eps)
            perturbed = torch.clamp(
                ori_images + delta, min=clamp_min, max=clamp_max
            ).detach()

        with torch.no_grad():
            final_loss = loss_f(batch_x=perturbed)
            if isinstance(final_loss, torch.Tensor):
                final_loss = final_loss.detach().mean().item()

        if final_loss > best_loss:
            best_loss = final_loss
            best_perturbed_batch_x = perturbed.clone().detach()

    return best_perturbed_batch_x.detach()


def PGD_R10_attack(batch_x, loss_f, eps, iters=80, alpha=0.01, random_start=True, restarts=10):
    return pgd_linf_restarts(
        batch_x=batch_x,
        loss_f=loss_f,
        eps=eps,
        iters=iters,
        alpha=alpha,
        alpha_divisor=None,
        random_start=random_start,
        restarts=restarts,
    )


def PGD_Long_attack(batch_x, loss_f, eps, iters=200, alpha=0.01, random_start=True, restarts=10):
    return pgd_linf_restarts(
        batch_x=batch_x,
        loss_f=loss_f,
        eps=eps,
        iters=iters,
        alpha=alpha,
        alpha_divisor=None,
        random_start=random_start,
        restarts=restarts,
    )


def PGD_Scaled8_attack(batch_x, loss_f, eps, iters=200, random_start=True, restarts=10):
    return pgd_linf_restarts(
        batch_x=batch_x,
        loss_f=loss_f,
        eps=eps,
        iters=iters,
        alpha=None,
        alpha_divisor=8,
        random_start=random_start,
        restarts=restarts,
    )


def PGD_Scaled16_attack(batch_x, loss_f, eps, iters=200, random_start=True, restarts=10):
    return pgd_linf_restarts(
        batch_x=batch_x,
        loss_f=loss_f,
        eps=eps,
        iters=iters,
        alpha=None,
        alpha_divisor=16,
        random_start=random_start,
        restarts=restarts,
    )


attack_params = {
    "PGD_R10_attack": {
        "iters": 80,
        "alpha": 0.01,
        "random_start": True,
        "restarts": 10,
    },
    "PGD_Long_attack": {
        "iters": 200,
        "alpha": 0.01,
        "random_start": True,
        "restarts": 10,
    },
    "PGD_Scaled8_attack": {
        "iters": 200,
        "random_start": True,
        "restarts": 10,
    },
    "PGD_Scaled16_attack": {
        "iters": 200,
        "random_start": True,
        "restarts": 10,
    },
}

foolbox_attacks = {
    "PGD_R10_attack": False,
    "PGD_Long_attack": False,
    "PGD_Scaled8_attack": False,
    "PGD_Scaled16_attack": False,
}


OUT_ROOT = "resultsTFG/pgd_sanity"
DATASET_NAME = "mnist"
THREAT_NAME = "Linf"


class ProtoVAEWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.input_bounds = (0, 1)

    def forward(self, x):
        x = x * 2 - 1
        logits, _ = self.base.pred_class(x)
        return logits


class SENNWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model
        self.input_bounds = (0, 1)

    def forward(self, x):
        x = x * 2 - 1
        return self.base(x)


def instantiate_senn_from_config(config_path, device):
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)
    cfg_dict["device"] = str(device)
    cfg = SimpleNamespace(**cfg_dict)

    conceptizer = eval(cfg.conceptizer)(**cfg.__dict__)
    parameterizer = eval(cfg.parameterizer)(**cfg.__dict__)
    aggregator = eval(cfg.aggregator)(**cfg.__dict__)
    return SENN(conceptizer, parameterizer, aggregator)


def load_models(model_names):
    models = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name in model_names:
        model_path = paths[name]

        if name in config_paths:
            cfg_path = config_paths[name]
            model = instantiate_senn_from_config(cfg_path, device)
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state["model_state"])
            model = SENNWrapper(model).to(device).eval()
            print(f"Model {name} (SENN) loaded")

        elif name == "ProtoVAE":
            model = model_protovae.ProtoVAE().to(device).eval()
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            model = ProtoVAEWrapper(model).to(device).eval()
            print(f"Model {name} (ProtoVAE base) loaded")

        elif name.startswith("ProtoVAE-FT-"):
            model = model_protovae.ProtoVAE().to(device)
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state["model_state"])
            model = ProtoVAEWrapper(model).to(device).eval()
            print(f"Model {name} loaded")

        elif name.startswith("B30-FT-"):
            B30_BASE = paths["B30"]
            model = torch.load(B30_BASE, map_location=device, weights_only=False)
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state["model_state"])
            model.eval()
            print(f"Model {name} loaded")

        else:
            model = torch.load(model_path, map_location=device, weights_only=False)
            model.eval()
            print(f"Model {name} loaded")

        models.append(model)

    return models, model_names


def get_function(function_name):
    try:
        return eval(function_name)
    except NameError:
        raise ValueError(
            f"Function '{function_name}' not found. "
            "Make sure it is defined or imported."
        )


def describe_attack(attack_name, params):
    if attack_name == "PGD_Scaled8_attack":
        alpha_desc = "alpha=eps/8"
    elif attack_name == "PGD_Scaled16_attack":
        alpha_desc = "alpha=eps/16"
    else:
        alpha_desc = f"alpha={params.get('alpha')}"

    print(
        f"Registered attack {attack_name}: "
        f"iters={params.get('iters')}, "
        f"restarts={params.get('restarts')}, "
        f"random_start={params.get('random_start')}, "
        f"{alpha_desc}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run PGD sanity-check adversarial evaluation and collect all internal metrics."
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Model names from the paths registry.",
    )
    parser.add_argument(
        "--attacks",
        nargs="+",
        required=True,
        help="Attack function names.",
    )

    parser.add_argument(
        "--max_eps",
        type=float,
        default=0.4,
        help="Maximum epsilon value (default: 0.4).",
    )
    parser.add_argument(
        "--step",
        type=float,
        default=0.025,
        help="Step size for epsilon grid (default: 0.025).",
    )

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--bootstrap_B", type=int, default=2000)
    parser.add_argument("--bootstrap_alpha", type=float, default=0.05)
    parser.add_argument(
        "--rpgd_binary_steps",
        type=int,
        default=10,
        help="Binary-search refinements for r-PGD inside each epsilon step.",
    )
    parser.add_argument(
        "--rpgd_binary_tol",
        type=float,
        default=None,
        help="Optional tolerance for stopping the r-PGD binary search early.",
    )

    args = parser.parse_args()

    if RUNTIME_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Cannot run PGD sanity metrics because a runtime dependency is missing. "
            "Activate the project environment with PyTorch and the project dependencies."
        ) from RUNTIME_IMPORT_ERROR

    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    set_seed(args.seed)

    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    models, model_names = load_models(args.models)

    loss_fn = CELoss

    attack_fns = []
    for attack_name in args.attacks:
        if attack_name not in attack_params:
            available = ", ".join(attack_params.keys())
            raise ValueError(
                f"Attack '{attack_name}' is not registered in this PGD sanity script. "
                f"Available attacks: {available}"
            )
        fn = get_function(attack_name)
        params = attack_params[attack_name].copy()
        describe_attack(attack_name, params)
        attack_fns.append(partial(fn, **params))

    data_folder = "data"
    batch_size = 250
    n_workers = 0

    test_loader = get_test_loader(
        data_folder,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    )

    foolbox = [foolbox_attacks.get(name, False) for name in args.attacks]

    adversarial_metrics_eps_collect(
        models=models,
        model_names=model_names,
        test_loader=test_loader,
        attacks=attack_fns,
        attack_names=args.attacks,
        loss=loss_fn,
        max_eps=args.max_eps,
        step=args.step,
        foolbox_uses=foolbox,
        out_root=OUT_ROOT,
        dataset=DATASET_NAME,
        threat=THREAT_NAME,
        seed=args.seed,
        run_id=args.run_id,
        attack_params_map=attack_params,
        bootstrap_B=args.bootstrap_B,
        bootstrap_alpha=args.bootstrap_alpha,
        rpgd_binary_steps=args.rpgd_binary_steps,
        rpgd_binary_tol=args.rpgd_binary_tol,
    )


if __name__ == "__main__":
    main()
