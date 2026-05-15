"""
run_metrics.py
==============

Usage examples
--------------
# B30 ablation sweep with PGD:
python run_metrics.py \
    --models B30 B30-FT-0 B30-FT-1 B30-FT-2 B30-FT-3 B30-FT-E \
    --attacks PGDLInf_attack \
    --max_eps 0.4 \
    --step 0.025 \
    --seed 1 \
    --run_id ablation_b30_v1

# SENN ablation:
python run_metrics.py \
    --models SENN_0_01 SENN-FT-0 SENN-FT-1 SENN-FT-2 \
    --attacks PGDLInf_attack \
    --max_eps 0.4 \
    --step 0.025 \
    --seed 1

# ProtoVAE ablation:
python run_metrics.py \
    --models ProtoVAE ProtoVAE-FT-0 ProtoVAE-FT-1 ProtoVAE-FT-2 \
    --attacks PGDLInf_attack \
    --max_eps 0.4 \
    --step 0.025 \
    --seed 1
"""

import argparse
import json
import random
from datetime import datetime
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from data_loader import *
from loss_functions import *
from adversarial_attacks import *
from foolbox.attacks import LinfAdditiveUniformNoiseAttack

from SENN.models.senn import SENN
from SENN.models.conceptizers   import ConvConceptizer
from SENN.models.parameterizers import ConvParameterizer
from SENN.models.aggregators    import SumAggregator
from ProtoVAE import model as model_protovae

from model_testing_metrics import adversarial_metrics_eps_collect


paths = {
    "B30":  "saved_model/mnist_model/mnist_cae_balanced_clstsep_1500_0.002_250_True_0.0_20_1_1_1_1.0_0.0_30_4_32_1/mnist_cae00750.pth",
    "S30":  "saved_model/mnist_model/mnist_cae_standard_default_1500_0.002_250_False_0.5_20_1_1_1_0.8_0.2_30_4_32_1/mnist_cae00750.pth",
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

    "ProtoVAE":      "ProtoVAE/saved_models/mnist/model.pth",
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

# ---------------------------------------------------------------------------
# Attack registry — identical to run_test.py
# ---------------------------------------------------------------------------
attack_params = {
    "PGDLInf_attack":                         {"iters": 80, "alpha": 0.01, "random_start": True},
    "FSGM_attack":                            {},
    "LinfDeepFool_attack":                    {"steps": 200, "candidates": 10, "overshoot": 0.005},
    "L2DeepFool_attack":                      {"steps": 200, "candidates": 10, "overshoot": 0.005},
    "LinfAdditiveUniformNoise_attack":        {},
    "LinfBasicIterative_attack":              {"steps": 10, "random_start": True},
    "LinfFMNA_attack":                        {"steps": 10000, "max_stepsize": 1, "min_stepsize": 1e-4, "gamma": 0.1, "binary_search_steps": 100},
    "L2FMNA_attack":                          {"steps": 10000, "max_stepsize": 1, "min_stepsize": 1e-4, "gamma": 0.1, "binary_search_steps": 100},
    "LinfMomentumIterativeFastGradient_attack": {"steps": 10},
    "LinfAdamProjectedGradientDescent_attack":  {"steps": 20, "random_start": True},
    "AutoAttack_adv":                          {"version": "standard"},
    "PatchPGD_attack":                         {"steps": 40, "step_size": 0.1, "random_start": True, "restarts": 1},
}

foolbox_attacks = {
    "PGDLInf_attack":                          False,
    "FSGM_attack":                             False,
    "LinfDeepFool_attack":                     True,
    "L2DeepFool_attack":                       True,
    "LinfAdditiveUniformNoise_attack":         True,
    "LinfBasicIterative_attack":               True,
    "LinfFMNA_attack":                         True,
    "L2FMNA_attack":                           True,
    "LinfMomentumIterativeFastGradient_attack": True,
    "LinfAdamProjectedGradientDescent_attack":  True,
    "AutoAttack_adv":                           True,
    "PatchPGD_attack":                          False,
}

# ---------------------------------------------------------------------------
# General settings
# ---------------------------------------------------------------------------
OUT_ROOT     = "resultsTFG/runs_v1"
DATASET_NAME = "mnist"
THREAT_NAME  = "Linf"


# ---------------------------------------------------------------------------
# Wrapper classes — identical to run_test.py
# ---------------------------------------------------------------------------
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

    conceptizer  = eval(cfg.conceptizer)(**cfg.__dict__)
    parameterizer = eval(cfg.parameterizer)(**cfg.__dict__)
    aggregator   = eval(cfg.aggregator)(**cfg.__dict__)
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

def main():
    parser = argparse.ArgumentParser(
        description="Run adversarial evaluation and collect all internal metrics."
    )

    # Required
    parser.add_argument("--models",  nargs="+", required=True,
                        help="Model names from the paths registry.")
    parser.add_argument("--attacks", nargs="+", required=True,
                        help="Attack function names.")

    # Epsilon grid
    parser.add_argument("--max_eps", type=float, default=0.4,
                        help="Maximum epsilon value (default: 0.4).")
    parser.add_argument("--step",    type=float, default=0.025,
                        help="Step size for epsilon grid (default: 0.025).")

    # Optional
    parser.add_argument("--seed",            type=int,   default=1)
    parser.add_argument("--run_id",          type=str,   default=None)
    parser.add_argument("--bootstrap_B",     type=int,   default=2000)
    parser.add_argument("--bootstrap_alpha", type=float, default=0.05)
    parser.add_argument("--rpgd_binary_steps", type=int, default=10,
                        help="Binary-search refinements for r-PGD inside each epsilon step.")
    parser.add_argument("--rpgd_binary_tol", type=float, default=None,
                        help="Optional tolerance for stopping the r-PGD binary search early.")

    args = parser.parse_args()

    # Reproducibility
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

    set_seed(args.seed)

    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load models
    models, model_names = load_models(args.models)

    # Loss function
    loss_fn = CELoss

    # Build attack callables (partial-applied with hyperparams)
    attack_fns = []
    for attack_name in args.attacks:
        fn     = get_function(attack_name)
        params = attack_params.get(attack_name, {}).copy()
        if attack_name == "PatchPGD_attack":
            params["base_seed"] = args.seed
        attack_fns.append(partial(fn, **params))

    # Data
    
    data_folder = 'data'
    batch_size = 250
    n_workers = 0
    
    test_loader = get_test_loader(
        data_folder, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True
    )

    foolbox = [foolbox_attacks.get(name, False) for name in args.attacks]

    # Run
    adversarial_metrics_eps_collect(
        models           = models,
        model_names      = model_names,
        test_loader      = test_loader,
        attacks          = attack_fns,
        attack_names     = args.attacks,
        loss             = loss_fn,
        max_eps          = args.max_eps,
        step             = args.step,
        foolbox_uses     = foolbox,
        out_root         = OUT_ROOT,
        dataset          = DATASET_NAME,
        threat           = THREAT_NAME,
        seed             = args.seed,
        run_id           = args.run_id,
        attack_params_map = attack_params,
        bootstrap_B      = args.bootstrap_B,
        bootstrap_alpha  = args.bootstrap_alpha,
        rpgd_binary_steps = args.rpgd_binary_steps,
        rpgd_binary_tol   = args.rpgd_binary_tol,
    )


if __name__ == "__main__":
    main()
