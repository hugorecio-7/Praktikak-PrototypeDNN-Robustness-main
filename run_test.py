import argparse
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import random
from datetime import datetime
from torch.autograd import Function
from functools import partial
from model_testing import adversarial_attacks_eps_plot_test
from data_loader import *
from loss_functions import *
from adversarial_attacks import *
from foolbox.attacks import LinfAdditiveUniformNoiseAttack
import json
from types import SimpleNamespace
from SENN.models.senn import SENN
from SENN.models.conceptizers   import ConvConceptizer    
from SENN.models.parameterizers import ConvParameterizer  
from SENN.models.aggregators    import SumAggregator       
from ProtoVAE import model as model_protovae

# Paths to model files
paths = {"B30": "saved_model/mnist_model/mnist_cae_balanced_clstsep_1500_0.002_250_True_0.0_20_1_1_1_1.0_0.0_30_4_32_1/mnist_cae00750.pth",
         "S30": "saved_model/mnist_model/mnist_cae_standard_default_1500_0.002_250_False_0.5_20_1_1_1_0.8_0.2_30_4_32_1/mnist_cae00750.pth",
         "RS30": "saved_model/mnist_model/mnist_cae_adversarial_standard_default_pdglinf_ce_20_0.3_0.02_True_1500_0.002_250_False_0.5_20_1_1_1_0.8_0.2_1.0_30_4_32_1/mnist_cae_adv00750.pth",
         "RB30": "saved_model/mnist_model/mnist_cae_adversarial_balanced_clstsep_pdglinf_ce_20_0.3_0.02_True_800_0.002_250_True_0.0_20_1_1_1_1.0_0.0_1.0_30_4_32_1/mnist_cae_adv00750.pth",
         "FTB30n": "saved_model/mnist_model/mnist_cae_FT_30_nothing_pdglinf_ce_20_0.3_0.02_True_20_0.002_250_20_1_1_1_1.0_0.0_1/mnist_cae_adv00020.pth",
         "FTB30p": "saved_model/mnist_model/mnist_cae_FT_30_prototypes_pdglinf_ce_20_0.3_0.02_True_20_0.002_250_20_1_1_1_1.0_0.0_1/mnist_cae_adv00020.pth",
         # SENN models
         "SENN_0_01": "SENN/results/mnist_lambda1e-2_seed29/checkpoints/best_model.pt",
         # ProtoVAE models
         "ProtoVAE": "ProtoVAE/saved_models/mnist/model.pth",
         }

# Paths to SENN config files
config_paths = {
        "SENN_0_01": "SENN/configs/mnist_lambda1e-2_seed29.json",
}

# Define attack parameters
attack_params = {
    "PGDLInf_attack": {"iters": 80, "alpha": 0.01, "random_start": True},
    "FSGM_attack" : {},
    "LinfDeepFool_attack": {"steps": 200, "candidates": 10, "overshoot": 0.005},
    "L2DeepFool_attack":  {"steps": 200, "candidates": 10, "overshoot": 0.005},
    "LinfAdditiveUniformNoise_attack": {}, 
    "LinfBasicIterative_attack": {"steps": 10, "random_start": True},
    "LinfFMNA_attack": {"steps": 10000, "max_stepsize": 1, "min_stepsize": 1e-4, "gamma": 0.1, "binary_search_steps": 100},
    "L2FMNA_attack": {"steps": 10000, "max_stepsize": 1, "min_stepsize": 1e-4, "gamma": 0.1, "binary_search_steps": 100},
    "LinfMomentumIterativeFastGradient_attack": {"steps": 10},
    "LinfAdamProjectedGradientDescent_attack": {"steps": 20, "random_start": True},
    "AutoAttack_adv": {"version": "standard"},
    "PatchPGD_attack": {"steps": 40, "step_size": 0.1, "random_start": True, "restarts": 1},
}

# Define which attacks are foolbox or AutoAttack 
foolbox_attacks = {
    "PGDLInf_attack": False,
    "FSGM_attack" : False,
    "LinfDeepFool_attack": True,
    "L2DeepFool_attack": True,
    "LinfAdditiveUniformNoise_attack": True,
    "LinfBasicIterative_attack": True,
    "LinfFMNA_attack": True,
    "L2FMNA_attack": True,
    "LinfMomentumIterativeFastGradient_attack": True,
    "LinfAdamProjectedGradientDescent_attack": True,
    "AutoAttack_adv": True,
    "PatchPGD_attack": False,
    
}

# General settings
OUT_ROOT = "resultsTFG/runs_v1"
DATASET_NAME = "mnist"
THREAT_NAME = "Linf"

# A subclass of ProtoVAE and SENN in order to use pred_class method as forward method and to set the input bounds to [0,1] instead of [-1,1] (since the test data is in [0,1]) without changing the original models
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

# A function to get the parts of the SENN model and load the full SENN model
def instantiate_senn_from_config(config_path, device):
    """
    Reads a SENN JSON config and returns an uninitialized model.
    """
    with open(config_path, 'r') as f:
        cfg_dict = json.load(f)
    cfg_dict['device'] = str(device)
    cfg = SimpleNamespace(**cfg_dict)

    conceptizer = eval(cfg.conceptizer)(**cfg.__dict__)
    parameterizer = eval(cfg.parameterizer)(**cfg.__dict__)
    aggregator = eval(cfg.aggregator)(**cfg.__dict__)
    model = SENN(conceptizer, parameterizer, aggregator)
    return model

# A function to load models and each weight based on their names (Each model has their own necessary parameters)
def load_models(model_names):
    """Load models based on their names."""
    models = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in model_names:
        model_path = paths[name]
        if name in config_paths: # If the model is a SENN model, use the config file to instantiate it
            cfg_path = config_paths[name]
            model = instantiate_senn_from_config(cfg_path, device)
            # Load weights
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state["model_state"])
            print("Model SENN loaded")
            model = SENNWrapper(model).to(device).eval()
        elif name == "ProtoVAE": # If the model is a ProtoVAE model, load it using the ProtoVAEWrapper class
            model = model_protovae.ProtoVAE().to(device).eval()
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            model = ProtoVAEWrapper(model).to(device).eval()
            print("Model ProtoVAE loaded")
        else: # For other models, load them directly
            model = torch.load(model_path, map_location=device)
            model.eval()
        models.append(model)
    return models, model_names

# A function to retrieve a function dynamically from available modules, in this case, the adversarial attack functions
def get_function(function_name):
    """Retrieve a function dynamically from available modules."""
    try:
        return eval(function_name)  # Evaluate the function name dynamically
    except NameError:
        raise ValueError(f"Function '{function_name}' not found. Make sure it's defined or imported.")

def main():
    
    parser = argparse.ArgumentParser(description="Run adversarial attack evaluation and save results.")

    # Required arguments
    parser.add_argument("--models", nargs="+", required=True, help="List of model names (without extension).")
    parser.add_argument("--attacks", nargs="+", required=True, help="Attack function name.")
    
    # Epsilon settings
    parser.add_argument("--max_eps", type=float, default=0.8, help="Maximum epsilon value.")
    parser.add_argument("--step", type=float, default=0.025, help="Step size for epsilon values.")
    
    # Optional arguments
    parser.add_argument("--seed", type=int, default=0, help="Global random seed for reproducibility.")
    parser.add_argument("--run_id", type=str, default=None, help="ID of the run (optional).")
    parser.add_argument("--bootstrap_B", type=int, default=2000, help="Bootstrap samples.")
    parser.add_argument("--bootstrap_alpha", type=float, default=0.05, help="Alpha bootstrap (0.05 => 95%).")

    args = parser.parse_args()

    # Set random seeds for reproducibility
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

    # Load models
    models, model_names = load_models(args.models)

    # Define the loss function internally
    loss_fn = CELoss

    # Get attack functions dynamically
    attack_fns = []
    for attack in args.attacks:
        attack_fn = get_function(attack)
        params = attack_params.get(attack, {}).copy()
        if attack == "PatchPGD_attack":
            params["base_seed"] = args.seed 
        attack_fns.append(partial(attack_fn, **params))

    data_folder = 'data'
    batch_size = 250
    n_workers = 0

    test_loader = get_test_loader(data_folder, batch_size, shuffle=False, num_workers=n_workers, pin_memory=True)

    # An array to indicate which attacks are foolbox or AutoAttack
    foolbox = [foolbox_attacks.get(attack, False) for attack in args.attacks]

    # Run adversarial attack experiment for each attack
    adversarial_attacks_eps_plot_test(
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
        bootstrap_alpha=args.bootstrap_alpha
    )

if __name__ == "__main__":
    main()
