import argparse
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
import torch.nn.functional as F
from torch.autograd import Function
from functools import partial
from model_testing import adversarial_attacks_eps_plot_test
from data_loader import *
from loss_functions import *
from adversarial_attacks import *
import json
from types import SimpleNamespace
from SENN.models.senn import SENN
from SENN.models.conceptizers   import ConvConceptizer    
from SENN.models.parameterizers import ConvParameterizer  
from SENN.models.aggregators    import SumAggregator       
from Prob_PSENN.ProbPSENN import ProbPSENN, ProbPSENN_VAE
from ProtoVAE import model as model_protovae
from autoattack import utils_tf2
import tf2onnx
import onnx
from onnx2pytorch import ConvertModel

paths = {"B30": "saved_model/mnist_model/mnist_cae_balanced_clstsep_1500_0.002_250_True_0.0_20_1_1_1_1.0_0.0_30_4_32_1/mnist_cae00750.pth",
         "S30": "saved_model/mnist_model/mnist_cae_standard_default_1500_0.002_250_False_0.5_20_1_1_1_0.8_0.2_30_4_32_1/mnist_cae00750.pth",
         "S15": "saved_model/mnist_model/mnist_cae_standard_default_1500_0.002_250_False_0.5_20_1_1_1_0.8_0.2_15_4_32_1/mnist_cae00750.pth",
         "RS30": "saved_model/mnist_model/mnist_cae_adversarial_standard_default_pdglinf_ce_20_0.3_0.02_True_1500_0.002_250_False_0.5_20_1_1_1_0.8_0.2_1.0_30_4_32_1/mnist_cae_adv00750.pth",
         "RB30": "saved_model/mnist_model/mnist_cae_adversarial_balanced_clstsep_pdglinf_ce_20_0.3_0.02_True_800_0.002_250_True_0.0_20_1_1_1_1.0_0.0_1.0_30_4_32_1/mnist_cae_adv00750.pth",
         "FTB30n": "saved_model/mnist_model/mnist_cae_FT_30_nothing_pdglinf_ce_20_0.3_0.02_True_20_0.002_250_20_1_1_1_1.0_0.0_1/mnist_cae_adv00020.pth",
         "FTB30a": "saved_model/mnist_model/mnist_cae_FT_30_autoencoder_pdglinf_ce_20_0.3_0.02_True_20_0.002_250_20_1_1_1_1.0_0.0_1/mnist_cae_adv00020.pth",
         "FTB30p": "saved_model/mnist_model/mnist_cae_FT_30_prototypes_pdglinf_ce_20_0.3_0.02_True_20_0.002_250_20_1_1_1_1.0_0.0_1/mnist_cae_adv00020.pth",
         # SENN models
         "SENN_0_1": "SENN/results/mnist_lambda1e-1_seed29/checkpoints/best_model.pt",
         "SENN_0_01": "SENN/results/mnist_lambda1e-2_seed29/checkpoints/best_model.pt",
         "SENN_0_001": "SENN/results/mnist_lambda1e-3_seed29/checkpoints/best_model.pt",
         # ProtoVAE models
         "ProtoVAE": "ProtoVAE/saved_models/mnist/model.pth",
         # Prob-PSENN models
         "Prob_PSENN": "Prob_PSENN/results/mnist_None_40_gauss_full_cnn_32_10_0.001_128_True_exp_0/model/",
         }

config_paths = {
        "SENN_0_1": "SENN/configs/mnist_lambda1e-1_seed29.json",
        "SENN_0_01": "SENN/configs/mnist_lambda1e-2_seed29.json",
        "SENN_0_001": "SENN/configs/mnist_lambda1e-3_seed29.json",
}

attack_params = {
    "PGDLInf_attack": {"iters": 2, "alpha": 1, "random_start": True},
    "LinfDeepFool_attack": {"steps": 100, "candidates": 3, "overshoot": 1.02},
    "LinfAdditiveUniformNoise_attack": {}, 
    "LinfBasicIterative_attack": {"steps": 10, "random_start": True},
    "LinfFMNA_attack": {"steps": 100, "max_stepsize": 2, "min_stepsize": 1e-3, "gamma": 0.1},
    "LinfMomentumIterativeFastGradient_attack": {"steps": 10},
    "LinfAdamProjectedGradientDescent_attack_foolbox": {"steps": 10, "random_start": True},
    "AutoAttack_adv": {"version": "standard", "is_tf": False},
}

foolbox_attacks = {
    "PGDLInf_attack": False,
    "LinfDeepFool_attack": True,
    "LinfAdditiveUniformNoise_attack": True,
    "LinfBasicIterative_attack": True,
    "LinfFMNA_attack": True,
    "LinfMomentumIterativeFastGradient_attack": True,
    "LinfAdamProjectedGradientDescent_attack_foolbox": True,
    "AutoAttack_adv": True,
}

class ProbPSENNAdapter(nn.Module):

    def __init__(self, tf_model):
        super().__init__()
        self.tf_model = tf_model

    def forward(self, x_tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # Convert PyTorch tensor to NumPy array (float32)
        x_np = x_tensor.detach().cpu().numpy()
        #x_np = x_tensor.cpu().permute(0,2,3,1).numpy()
        if x_np.ndim == 4 and x_np.shape[1] in (1, 3):
            x_np = x_np.transpose(0, 2, 3, 1)
        # Call the TF model's distribution function
        distrib = self.tf_model.get_pred_distrib(x_np, n_samples=30, reduce_samples=True)
        # Convert back to PyTorch tensor, preserving device if needed
        return torch.from_numpy(distrib.probs.numpy()).to(x_tensor.device)

    def __call__(self, x_tensor: torch.Tensor) -> torch.Tensor:
        return self.forward(x_tensor)

class ProbPSENNWrapper(nn.Module):
    def __init__(self, tf_model):
        super().__init__()
        self.tf_model = tf_model

    def forward(self, x, *args, **kwargs):
        # Detach and use TF model (non-differentiable)
        with torch.no_grad():
            x_np = x.permute(0, 2, 3, 1).cpu().numpy()  # [B, 32, 32, 1]
            pred_dist = self.tf_model.get_pred_distrib(x_np, n_samples=30, reduce_samples=True)
            probs_np = pred_dist.probs.numpy()

        # Reattach to computation graph
        probs = torch.tensor(probs_np, device=x.device, requires_grad=True)
        logits = torch.log(probs + 1e-12)
        #Trick AutoAttack into "seeing" gradients (passthrough)
        logits = logits + 0.0 * x.sum()  
        return logits

class ProtoVAEWrapper(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        logits, _ = self.base.pred_class(x)
        return logits

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

def load_models(model_names):
    """Load models based on their names."""
    models = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name in model_names:
        model_path = paths[name]
        if name in config_paths:
            cfg_path = config_paths[name]
            model = instantiate_senn_from_config(cfg_path, device)
            # Load weights
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state["model_state"])
            print("Model SENN loaded")
            model.eval()
        elif name == "Prob_PSENN":
            if True:
                model = ProbPSENN_VAE.load_model(model_path, "") #Load the model (VAE backbone)
                # onnx_model, _ = tf2onnx.convert.from_keras(model)
                # pytorch_model = ConvertModel(onnx_model)
                model = ProbPSENNWrapper(model).to(device).eval()
                #model = ProbPSENNAdapter(model).to(device).eval()
            else:
                model = ProbPSENN.load_model(model_path, "")     #Load the model (AE backbone)
            print("Model Prob_PSENN loaded")
        elif name == "ProtoVAE":
            model = model_protovae.ProtoVAE().to(device).eval()
            state = torch.load(model_path, map_location=device)
            model.load_state_dict(state)
            model = ProtoVAEWrapper(model).to(device).eval()
            print("Model ProtoVAE loaded")
        else:
            model = torch.load(model_path, map_location=device)
            model.eval()
        models.append(model)
    return models, model_names

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
    
    # # Output paths
    # parser.add_argument("--csv_path", type=str, default="results/accuracy_results.csv", help="Path to save the CSV.")
    # parser.add_argument("--jpg_path", type=str, default="results/accuracy_plot.jpg", help="Path to save the plot.")

    args = parser.parse_args()

    # Load models
    models, model_names = load_models(args.models)

    # Define the loss function internally
    loss_fn = CELoss

    # Get attack functions dynamically
    attack_fns = []
    for attack in args.attacks:
        attack_fn = get_function(attack)
        params = attack_params.get(attack, {}).copy()
        if attack == "AutoAttack_adv" and args.models[0] == "Prob_PSENN":
            params["version"] = "rand"
            #params["is_tf"] = True
        #params = attack_params.get(attack, {})  # Get attack parameters
        attack_fns.append(partial(attack_fn, **params))

    data_folder = 'data'
    batch_size = 250
    n_workers = 0
    random_seed = 0

    # download MNIST data
    mode = "basic"    

    if args.models[0] == "ProtoVAE":
        mode = "norm"
    elif args.models[0] == "Prob_PSENN":
        mode = "resize_norm"

    test_loader = get_test_loader(data_folder, batch_size, shuffle=True, num_workers=n_workers, pin_memory=True, mode=mode)

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
    )

if __name__ == "__main__":
    main()
