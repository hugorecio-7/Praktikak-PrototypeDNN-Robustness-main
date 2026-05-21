import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
from html import parser
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from data_loader import get_train_val_loader, get_test_loader
from adversarial_attacks import PGDLInf_attack
from ProtoVAE import model as model_protovae
from ProtoVAE import settings as protovae_settings

from SENN.models.senn import SENN
from SENN.models.conceptizers import *  # noqa: F401,F403
from SENN.models.parameterizers import *  # noqa: F401,F403
from SENN.models.aggregators import *  # noqa: F401,F403
from SENN.models.losses import mnist_robustness_loss, mse_l1_sparsity
from loss_functions import ClstSepLoss
import modules  # noqa: F401
from Prueba.reconstruction_risk_wrapper import ReconstructionRiskWrapper


# ============================================================
# Fixed repo paths
# ============================================================
PROTO_VAE_CKPT = "ProtoVAE/saved_models/mnist/model.pth"
SENN_CKPT = "SENN/results/mnist_lambda1e-2_seed29/checkpoints/best_model.pt"
SENN_CONFIG = "SENN/configs/mnist_lambda1e-2_seed29.json"
B30_CKPT = "saved_model/mnist_model/mnist_cae_balanced_clstsep_1500_0.002_250_True_0.0_20_1_1_1_1.0_0.0_30_4_32_1/mnist_cae00750.pth"

# Baseline defaults extracted from previous B30 / FTB30 configs in the repo.
B30_DEFAULT_EPOCHS = 50
B30_DEFAULT_BATCH_SIZE = 250
B30_DEFAULT_LR = 0.002
B30_DEFAULT_LAMBDA_CLASS = 20.0
B30_DEFAULT_LAMBDA_AE = 1.0
B30_DEFAULT_LAMBDA_1 = 1.0
B30_DEFAULT_LAMBDA_CLUS = 0.8
B30_DEFAULT_LAMBDA_SEP = 0.2

# Adversarial attack defaults (can be overridden by command-line args):
SHARED_ADV_EPS = 0.3
SHARED_ADV_ALPHA = 0.01
SHARED_ADV_STEPS = 40
SHARED_ADV_RANDOM_START = True


# ============================================================
# Final variants matrix
# ============================================================
PROTOVAE_VARIANTS = {
    "ProtoVAE-FT-0": {"freeze": []},
    "ProtoVAE-FT-1": {"freeze": ["autoencoder"]},
    "ProtoVAE-FT-2": {"freeze": ["prototypes"]},
    "ProtoVAE-FT-3": {"freeze": ["classifier"]},
    "ProtoVAE-FT-E": {"freeze": ["prototypes", "classifier"]},
}

SENN_VARIANTS = {
    "SENN-FT-0": {"freeze": []},
    "SENN-FT-1": {"freeze": ["conceptizer"]},
    "SENN-FT-2": {"freeze": ["parameterizer"]},
}

B30_VARIANTS = {
    "B30-FT-0": {"freeze": []},
    "B30-FT-1": {"freeze": ["autoencoder"]},
    "B30-FT-2": {"freeze": ["prototypes"]},
    "B30-FT-3": {"freeze": ["classifier"]},
    "B30-FT-E": {"freeze": ["prototypes", "classifier"]},
    "B30-FT-Shield": {"freeze": ["prototypes", "classifier", "decoder"]},
}


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "t", "1", "yes", "y", "si", "s"}:
        return True
    if normalized in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value {value!r}. Use true/false."
    )


# ============================================================
# Config
# ============================================================
@dataclass
class TrainConfig:
    model_name: str
    variant: str
    seed: int

    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    val_size: float
    early_stopping_patience: int
    save_root: str
    num_workers: int

    # Adversarial training parameters
    adv_eps: float
    adv_alpha: float
    adv_steps: int
    adv_random_start: bool

    # Global branch weights
    lambda_clean: float
    lambda_adv: float

    # Optional extra low-epsilon adversarial branch (disabled when adv_low_eps is None)
    adv_low_eps: float | None
    adv_low_alpha: float | None
    adv_low_steps: int | None
    lambda_adv_low: float

    # ProtoVAE coefficients (original clean loss)
    proto_ce_coef: float
    proto_recon_coef: float
    proto_kl_coef: float
    proto_ortho_coef: float

    # SENN coefficients (from config / losses.py)
    senn_concept_reg: float
    senn_robust_reg: float
    senn_sparsity_reg: float

    # B30 coefficients (balanced clstsep setup)
    b30_lambda_class: float
    b30_lambda_ae: float
    b30_lambda_1: float
    b30_lambda_clus: float
    b30_lambda_sep: float

    # Runtime
    data_dir: str = "data"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    run_tag: str = ""

    # Reconstruction-Risk Shield hyperparameters
    shield_gamma: float = 25.0
    shield_power: int = 3
    shield_use_shift: bool = True


# ============================================================
# Generic utils
# ============================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(path: str | Path, obj: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_history_csv(path: str | Path, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_config_txt(path: str | Path, cfg: TrainConfig, trainable_params: int, total_params: int) -> None:
    lines = [
        "Adversarial Fine-tuning Configuration",
        f"Model: {cfg.model_name}",
        f"Variant: {cfg.variant}",
        f"Random Seed: {cfg.seed}",
        "",
        "Optimization",
        f"Epochs: {cfg.epochs}",
        f"Learning Rate: {cfg.lr}",
        f"Batch Size: {cfg.batch_size}",
        f"Num Workers: {cfg.num_workers}",
        f"Weight Decay: {cfg.weight_decay}",
        f"Validation Split: {cfg.val_size}",
        "",
        "Adversarial Training (PGD)",
        "Adversarial Attack: pdglinf",
        f"Adversarial Iterations: {cfg.adv_steps}",
        f"Adversarial Epsilon: {cfg.adv_eps}",
        f"Adversarial Alpha: {cfg.adv_alpha}",
        f"Random Start: {cfg.adv_random_start}",
        f"Low-epsilon branch enabled: {cfg.adv_low_eps is not None}",
        f"Low-epsilon branch epsilon: {cfg.adv_low_eps}",
        f"Low-epsilon branch alpha: {cfg.adv_low_alpha}",
        f"Low-epsilon branch steps: {cfg.adv_low_steps}",
        f"Low-epsilon branch weight: {cfg.lambda_adv_low}",
        "",
        "ProtoVAE Coefficients",
        f"proto_ce_coef: {cfg.proto_ce_coef}",
        f"proto_recon_coef: {cfg.proto_recon_coef}",
        f"proto_kl_coef: {cfg.proto_kl_coef}",
        f"proto_ortho_coef: {cfg.proto_ortho_coef}",
        "",
        "SENN Coefficients",
        f"senn_concept_reg: {cfg.senn_concept_reg}",
        f"senn_robust_reg: {cfg.senn_robust_reg}",
        f"senn_sparsity_reg: {cfg.senn_sparsity_reg}",
        "",
        "B30 Coefficients",
        f"b30_lambda_class: {cfg.b30_lambda_class}",
        f"b30_lambda_ae: {cfg.b30_lambda_ae}",
        f"b30_lambda_1: {cfg.b30_lambda_1}",
        f"b30_lambda_clus: {cfg.b30_lambda_clus}",
        f"b30_lambda_sep: {cfg.b30_lambda_sep}",
        "",
        "Parameter Counts",
        f"Total Parameters: {total_params}",
        f"Trainable Parameters: {trainable_params}",
        "",
        "Save Root",
        f"{cfg.save_root}",
        f"Run Tag: {cfg.run_tag}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def freeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = False


def resolve_default_save_root() -> str:
    if Path("/kaggle/working").exists():
        return "/kaggle/working/adv_finetune_components"
    if Path("/content").exists():
        return "/content/resultsTFG/adv_finetune_components"
    return "resultsTFG/adv_finetune_components"


def resolve_default_num_workers() -> int:
    cpu_count = os.cpu_count() or 2
    if Path("/kaggle/working").exists():
        return max(1, min(4, cpu_count - 1))
    return max(0, min(2, cpu_count - 1))


def save_trainable_parameter_names(model: nn.Module, path: str | Path) -> None:
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    with open(path, "w", encoding="utf-8") as f:
        for name in names:
            f.write(name + "\n")


# ============================================================
# SENN config / model loading
# ============================================================
def load_senn_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def instantiate_senn_from_config(config_path: str, device: torch.device) -> nn.Module:
    cfg_dict = load_senn_config(config_path)
    cfg_dict["device"] = str(device)
    cfg = SimpleNamespace(**cfg_dict)

    conceptizer = eval(cfg.conceptizer)(**cfg.__dict__)
    parameterizer = eval(cfg.parameterizer)(**cfg.__dict__)
    aggregator = eval(cfg.aggregator)(**cfg.__dict__)
    model = SENN(conceptizer, parameterizer, aggregator)
    return model


def load_protovae(device: torch.device) -> nn.Module:
    model = model_protovae.ProtoVAE().to(device)
    state = torch.load(PROTO_VAE_CKPT, map_location=device)
    model.load_state_dict(state)
    return model


def load_senn(device: torch.device) -> nn.Module:
    model = instantiate_senn_from_config(SENN_CONFIG, device)
    state = torch.load(SENN_CKPT, map_location=device)
    model.load_state_dict(state["model_state"])
    return model.to(device)


def load_b30(device: torch.device) -> nn.Module:
    # B30 checkpoints were saved as full model objects in this repo.
    try:
        model = torch.load(B30_CKPT, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(B30_CKPT, map_location=device)

    if not isinstance(model, nn.Module):
        raise TypeError("B30 checkpoint is not a torch.nn.Module")

    return model.to(device)


def load_model(model_name: str, device: torch.device) -> nn.Module:
    if model_name == "ProtoVAE":
        return load_protovae(device)
    if model_name == "SENN_0_01":
        return load_senn(device)
    if model_name == "B30":
        return load_b30(device)
    raise ValueError(f"Unsupported model_name: {model_name}")


# ============================================================
# Freeze plans
# ============================================================
def apply_protovae_freeze(model: nn.Module, variant: str) -> None:
    for p in model.parameters():
        p.requires_grad = True

    freeze_list = PROTOVAE_VARIANTS[variant]["freeze"]

    if "autoencoder" in freeze_list:
        freeze_module(model.features)
        freeze_module(model.decoder_layers)

    if "prototypes" in freeze_list:
        model.prototype_vectors.requires_grad = False

    if "classifier" in freeze_list:
        freeze_module(model.last_layer)


def apply_senn_freeze(model: nn.Module, variant: str) -> None:
    for p in model.parameters():
        p.requires_grad = True

    freeze_list = SENN_VARIANTS[variant]["freeze"]

    if "conceptizer" in freeze_list:
        freeze_module(model.conceptizer)

    if "parameterizer" in freeze_list:
        freeze_module(model.parameterizer)

    # SumAggregator has no trainable parameters in this setup.


def apply_b30_freeze(model: nn.Module, variant: str) -> None:
    for p in model.parameters():
        p.requires_grad = True

    freeze_list = B30_VARIANTS[variant]["freeze"]

    if "autoencoder" in freeze_list:
        freeze_module(model.encoder)
        freeze_module(model.decoder)

    if "prototypes" in freeze_list:
        freeze_module(model.prototype_layer)

    if "classifier" in freeze_list:
        freeze_module(model.fc)

    if "decoder" in freeze_list:
        freeze_module(model.decoder)


def apply_freeze_plan(model: nn.Module, model_name: str, variant: str) -> None:
    if model_name == "ProtoVAE":
        apply_protovae_freeze(model, variant)
        return
    if model_name == "SENN_0_01":
        apply_senn_freeze(model, variant)
        return
    if model_name == "B30":
        apply_b30_freeze(model, variant)
        return
    raise ValueError(f"Unsupported model_name: {model_name}")


# ============================================================
# Input adapters
# ============================================================
def to_model_input(model_name: str, x_raw: torch.Tensor) -> torch.Tensor:
    """
    External dataset is always [0,1].
    ProtoVAE and SENN are fed internally in [-1,1].
    """
    if model_name in {"ProtoVAE", "SENN_0_01"}:
        return x_raw * 2.0 - 1.0
    return x_raw


# ============================================================
# ProtoVAE terms
# ============================================================
def protovae_terms(
    model: nn.Module,
    input_raw: torch.Tensor,
    recon_target_raw: torch.Tensor,
    y: torch.Tensor,
    cfg: TrainConfig,
):
    """
    ProtoVAE clean/adv branch:
    - classification CE
    - reconstruction MSE
    - KL
    - orthogonality

    For adversarial batches, recon_target_raw can still be the clean input.
    This mirrors the spirit of the previous PrototypeDL fine-tuning, where
    adversarial inputs should reconstruct the clean target rather than the noise.
    """
    x_in = to_model_input("ProtoVAE", input_raw)
    x_target = to_model_input("ProtoVAE", recon_target_raw)

    logits, decoded, kl_loss, ortho_loss = model(x_in, y=y, is_train=True)

    ce = F.cross_entropy(logits, y)
    recon = F.mse_loss(decoded, x_target, reduction="mean")

    # Clamp KL loss to prevent extreme values from destabilizing training, especially in early epochs.
    kl_loss = torch.clamp(kl_loss, max=10.0)

    full_loss = (
        cfg.proto_ce_coef * ce
        + cfg.proto_recon_coef * recon
        + cfg.proto_kl_coef * kl_loss
        + cfg.proto_ortho_coef * ortho_loss
    )

    return {
        "logits": logits,
        "full_loss": full_loss,
        "ce": ce,
        "recon": recon,
        "kl": kl_loss,
        "ortho": ortho_loss,
    }


def protovae_logits_only(model: nn.Module, input_raw: torch.Tensor) -> torch.Tensor:
    x_in = to_model_input("ProtoVAE", input_raw)
    logits, _ = model.pred_class(x_in)
    return logits


# ============================================================
# B30 terms
# ============================================================
def b30_terms(
    model: nn.Module,
    input_raw: torch.Tensor,
    recon_target_raw: torch.Tensor,
    y: torch.Tensor,
    cfg: TrainConfig,
):
    """
    B30 branch with the same core loss family used for balanced model training:
    ClstSepLoss = classification + prototype alignment + clst/sep + reconstruction.

    This mirrors finetune_model.py behavior for ClstSepLoss, where
    batch_x_ori is not provided and reconstruction is done against the
    current input branch (clean or adversarial).
    """
    logits = model(input_raw)

    full_loss, ce, e1, clst_l, sep_l, ae = ClstSepLoss(
        model,
        input_raw,
        y,
        logits,
        cfg.b30_lambda_class,
        cfg.b30_lambda_1,
        cfg.b30_lambda_clus,
        cfg.b30_lambda_sep,
        cfg.b30_lambda_ae,
    )

    return {
        "logits": logits,
        "full_loss": full_loss,
        "ce": ce,
        "e1": e1,
        "clst": clst_l,
        "sep": sep_l,
        "recon": ae,
    }


def b30_logits_only(model: nn.Module, input_raw: torch.Tensor) -> torch.Tensor:
    return model(input_raw)


# ============================================================
# SENN terms
# ============================================================
def senn_terms(
    model: nn.Module,
    input_raw: torch.Tensor,
    recon_target_raw: torch.Tensor,
    y: torch.Tensor,
    cfg: TrainConfig,
):
    """
    SENN clean/adv branch:
    - classification NLL (because SumAggregator returns log_softmax)
    - concept loss via mse_l1_sparsity
    - robustness loss via mnist_robustness_loss

    For adversarial batches, recon_target_raw can still be the clean input,
    again matching the idea of reconstructing the clean signal from perturbed inputs.
    """
    x_in = to_model_input("SENN_0_01", input_raw).detach().requires_grad_(True)
    x_target = to_model_input("SENN_0_01", recon_target_raw).detach()

    concepts, recon_x = model.conceptizer(x_in)
    relevances = model.parameterizer(x_in)

    aggregates = torch.bmm(relevances.permute(0, 2, 1), concepts)  # [B, num_classes, 1]
    log_probs = F.log_softmax(aggregates.squeeze(-1), dim=1)

    nll = F.nll_loss(log_probs, y)
    concept_loss = mse_l1_sparsity(x_target, recon_x, concepts, cfg.senn_sparsity_reg)
    robust_loss = mnist_robustness_loss(x_in, aggregates, concepts, relevances)

    full_loss = nll + cfg.senn_concept_reg * concept_loss + cfg.senn_robust_reg * robust_loss

    return {
        "logits": log_probs,
        "full_loss": full_loss,
        "nll": nll,
        "concept_loss": concept_loss,
        "robust_loss": robust_loss,
        "recon": F.mse_loss(recon_x, x_target, reduction="mean"),
    }


def senn_logits_only(model: nn.Module, input_raw: torch.Tensor) -> torch.Tensor:
    x_in = to_model_input("SENN_0_01", input_raw)
    return model(x_in)  # already log_softmax


# ============================================================
# Attack generation
# ============================================================
def attack_classification_loss(
    batch_x: torch.Tensor,
    model: nn.Module,
    model_name: str,
    batch_y: torch.Tensor,
) -> torch.Tensor:
    if model_name == "ProtoVAE":
        logits = protovae_logits_only(model, batch_x)
        return F.cross_entropy(logits, batch_y)

    if model_name == "SENN_0_01":
        log_probs = senn_logits_only(model, batch_x)
        return F.nll_loss(log_probs, batch_y)

    if model_name == "B30":
        logits = b30_logits_only(model, batch_x)
        return F.cross_entropy(logits, batch_y)

    raise ValueError(f"Unsupported model_name: {model_name}")


def generate_adv_batch(
    model: nn.Module,
    model_name: str,
    x_clean: torch.Tensor,
    y: torch.Tensor,
    cfg: TrainConfig,
    eps: float | None = None,
    alpha: float | None = None,
    steps: int | None = None,
) -> torch.Tensor:
    """
    Generate PGD adversarial examples in the external [0,1] space.
    """
    was_training = model.training
    model.eval()

    loss_f = lambda *, batch_x: attack_classification_loss(
        batch_x=batch_x,
        model=model,
        model_name=model_name,
        batch_y=y,
    )

    attack_eps = cfg.adv_eps if eps is None else eps
    attack_alpha = cfg.adv_alpha if alpha is None else alpha
    attack_steps = cfg.adv_steps if steps is None else steps

    x_adv = PGDLInf_attack(
        batch_x=x_clean,
        loss_f=loss_f,
        iters=attack_steps,
        eps=attack_eps,
        alpha=attack_alpha,
        random_start=cfg.adv_random_start,
    )

    model.train(was_training)
    return x_adv.detach()


# ============================================================
# Batch step
# ============================================================
def compute_model_terms(
    model: nn.Module,
    model_name: str,
    input_raw: torch.Tensor,
    recon_target_raw: torch.Tensor,
    y: torch.Tensor,
    cfg: TrainConfig,
):
    if model_name == "ProtoVAE":
        return protovae_terms(model, input_raw, recon_target_raw, y, cfg)
    if model_name == "SENN_0_01":
        return senn_terms(model, input_raw, recon_target_raw, y, cfg)
    if model_name == "B30":
        return b30_terms(model, input_raw, recon_target_raw, y, cfg)
    raise ValueError(f"Unsupported model_name: {model_name}")


def run_epoch(model: nn.Module, loader, cfg: TrainConfig, optimizer: optim.Optimizer | None) -> Dict[str, float]:
    """
    PrototypeDL-style alternating training:
    - generate adversarial batch
    - do one optimization step on adversarial data
    - do one optimization step on clean data

    In evaluation mode:
    - no updates
    - compute both clean and adversarial metrics
    """
    is_train = optimizer is not None
    model.train(is_train)

    total_seen = 0

    meters = {
        "total_loss": 0.0,
        "clean_full_loss": 0.0,
        "adv_full_loss": 0.0,
        "clean_acc": 0.0,
        "adv_acc": 0.0,

        "proto_clean_ce": 0.0,
        "proto_clean_recon": 0.0,
        "proto_clean_kl": 0.0,
        "proto_clean_ortho": 0.0,

        "proto_adv_ce": 0.0,
        "proto_adv_recon": 0.0,
        "proto_adv_kl": 0.0,
        "proto_adv_ortho": 0.0,

        "senn_clean_nll": 0.0,
        "senn_clean_concept_loss": 0.0,
        "senn_clean_robust_loss": 0.0,
        "senn_clean_recon": 0.0,

        "senn_adv_nll": 0.0,
        "senn_adv_concept_loss": 0.0,
        "senn_adv_robust_loss": 0.0,
        "senn_adv_recon": 0.0,

        "b30_clean_ce": 0.0,
        "b30_clean_e1": 0.0,
        "b30_clean_clst": 0.0,
        "b30_clean_sep": 0.0,
        "b30_clean_recon": 0.0,

        "b30_adv_ce": 0.0,
        "b30_adv_e1": 0.0,
        "b30_adv_clst": 0.0,
        "b30_adv_sep": 0.0,
        "b30_adv_recon": 0.0,
    }

    pbar = tqdm(loader, leave=False)

    for x_clean, y in pbar:
        x_clean = x_clean.to(cfg.device)
        y = y.to(cfg.device)

        # 1) Build main adversarial batch using the current model state
        x_adv = generate_adv_batch(model, cfg.model_name, x_clean, y, cfg)

        if is_train:
            # ----------------------------------------------------
            # Adversarial update
            # ----------------------------------------------------
            optimizer.zero_grad(set_to_none=True)
            adv_terms = compute_model_terms(
                model=model,
                model_name=cfg.model_name,
                input_raw=x_adv,
                recon_target_raw=x_clean,   # reconstruct clean target
                y=y,
                cfg=cfg,
            )
            (cfg.lambda_adv * adv_terms["full_loss"]).backward()
            optimizer.step()

            # ----------------------------------------------------
            # Optional extra low-epsilon adversarial update
            # ----------------------------------------------------
            if cfg.adv_low_eps is not None:
                x_adv_low = generate_adv_batch(
                    model,
                    cfg.model_name,
                    x_clean,
                    y,
                    cfg,
                    eps=cfg.adv_low_eps,
                    alpha=cfg.adv_low_alpha,
                    steps=cfg.adv_low_steps,
                )
                optimizer.zero_grad(set_to_none=True)
                adv_low_terms = compute_model_terms(
                    model=model,
                    model_name=cfg.model_name,
                    input_raw=x_adv_low,
                    recon_target_raw=x_clean,
                    y=y,
                    cfg=cfg,
                )
                (cfg.lambda_adv_low * adv_low_terms["full_loss"]).backward()
                optimizer.step()

            # ----------------------------------------------------
            # Clean update
            # ----------------------------------------------------
            optimizer.zero_grad(set_to_none=True)
            clean_terms = compute_model_terms(
                model=model,
                model_name=cfg.model_name,
                input_raw=x_clean,
                recon_target_raw=x_clean,
                y=y,
                cfg=cfg,
            )
            (cfg.lambda_clean * clean_terms["full_loss"]).backward()
            optimizer.step()

        else:
            # SENN robustness loss requires autograd Jacobians even in eval.
            if cfg.model_name == "SENN_0_01":
                adv_terms = compute_model_terms(
                    model=model,
                    model_name=cfg.model_name,
                    input_raw=x_adv,
                    recon_target_raw=x_clean,
                    y=y,
                    cfg=cfg,
                )
                clean_terms = compute_model_terms(
                    model=model,
                    model_name=cfg.model_name,
                    input_raw=x_clean,
                    recon_target_raw=x_clean,
                    y=y,
                    cfg=cfg,
                )
            else:
                with torch.no_grad():
                    adv_terms = compute_model_terms(
                        model=model,
                        model_name=cfg.model_name,
                        input_raw=x_adv,
                        recon_target_raw=x_clean,
                        y=y,
                        cfg=cfg,
                    )
                    clean_terms = compute_model_terms(
                        model=model,
                        model_name=cfg.model_name,
                        input_raw=x_clean,
                        recon_target_raw=x_clean,
                        y=y,
                        cfg=cfg,
                    )

        clean_preds = clean_terms["logits"].argmax(dim=1)
        adv_preds = adv_terms["logits"].argmax(dim=1)

        bs = y.size(0)
        total_seen += bs

        meters["clean_full_loss"] += float(clean_terms["full_loss"].detach().item()) * bs
        meters["adv_full_loss"] += float(adv_terms["full_loss"].detach().item()) * bs
        meters["total_loss"] += float(
            (cfg.lambda_clean * clean_terms["full_loss"] + cfg.lambda_adv * adv_terms["full_loss"]).detach().item()
        ) * bs

        meters["clean_acc"] += float((clean_preds == y).sum().item())
        meters["adv_acc"] += float((adv_preds == y).sum().item())

        if cfg.model_name == "ProtoVAE":
            meters["proto_clean_ce"] += float(clean_terms["ce"].detach().item()) * bs
            meters["proto_clean_recon"] += float(clean_terms["recon"].detach().item()) * bs
            meters["proto_clean_kl"] += float(clean_terms["kl"].detach().item()) * bs
            meters["proto_clean_ortho"] += float(clean_terms["ortho"].detach().item()) * bs

            meters["proto_adv_ce"] += float(adv_terms["ce"].detach().item()) * bs
            meters["proto_adv_recon"] += float(adv_terms["recon"].detach().item()) * bs
            meters["proto_adv_kl"] += float(adv_terms["kl"].detach().item()) * bs
            meters["proto_adv_ortho"] += float(adv_terms["ortho"].detach().item()) * bs
        elif cfg.model_name == "SENN_0_01":
            meters["senn_clean_nll"] += float(clean_terms["nll"].detach().item()) * bs
            meters["senn_clean_concept_loss"] += float(clean_terms["concept_loss"].detach().item()) * bs
            meters["senn_clean_robust_loss"] += float(clean_terms["robust_loss"].detach().item()) * bs
            meters["senn_clean_recon"] += float(clean_terms["recon"].detach().item()) * bs

            meters["senn_adv_nll"] += float(adv_terms["nll"].detach().item()) * bs
            meters["senn_adv_concept_loss"] += float(adv_terms["concept_loss"].detach().item()) * bs
            meters["senn_adv_robust_loss"] += float(adv_terms["robust_loss"].detach().item()) * bs
            meters["senn_adv_recon"] += float(adv_terms["recon"].detach().item()) * bs
        else:
            meters["b30_clean_ce"] += float(clean_terms["ce"].detach().item()) * bs
            meters["b30_clean_e1"] += float(clean_terms["e1"].detach().item()) * bs
            meters["b30_clean_clst"] += float(clean_terms["clst"].detach().item()) * bs
            meters["b30_clean_sep"] += float(clean_terms["sep"].detach().item()) * bs
            meters["b30_clean_recon"] += float(clean_terms["recon"].detach().item()) * bs

            meters["b30_adv_ce"] += float(adv_terms["ce"].detach().item()) * bs
            meters["b30_adv_e1"] += float(adv_terms["e1"].detach().item()) * bs
            meters["b30_adv_clst"] += float(adv_terms["clst"].detach().item()) * bs
            meters["b30_adv_sep"] += float(adv_terms["sep"].detach().item()) * bs
            meters["b30_adv_recon"] += float(adv_terms["recon"].detach().item()) * bs

        pbar.set_description(
            f"loss={meters['total_loss']/max(total_seen,1):.4f} | clean_acc={meters['clean_acc']/max(total_seen,1):.4f} | adv_acc={meters['adv_acc']/max(total_seen,1):.4f}"
        )

    out_metrics = {}
    for k, v in meters.items():
        if "acc" in k:
            out_metrics[k] = v / total_seen
        else:
            out_metrics[k] = v / total_seen

    return out_metrics


# ============================================================
# Config builder
# ============================================================
def validate_variant(model_name: str, variant: str) -> None:
    if model_name == "ProtoVAE" and variant not in PROTOVAE_VARIANTS:
        raise ValueError(f"Unknown ProtoVAE variant: {variant}")
    if model_name == "SENN_0_01" and variant not in SENN_VARIANTS:
        raise ValueError(f"Unknown SENN variant: {variant}")
    if model_name == "B30" and variant not in B30_VARIANTS:
        raise ValueError(f"Unknown B30 variant: {variant}")


def build_config(args: argparse.Namespace) -> TrainConfig:
    senn_cfg = load_senn_config(SENN_CONFIG)

    # Model-specific baseline defaults extracted from repo configs/settings:
    # - ProtoVAE: ProtoVAE/settings.py
    # - SENN: SENN/configs/mnist_lambda1e-2_seed29.json
    # - B30: saved_model/* B30/FTB30 config.txt
    if args.model_name == "ProtoVAE":
        default_epochs = int(protovae_settings.num_train_epochs)
        default_batch_size = int(protovae_settings.batch_size)
        default_lr = float(protovae_settings.lr)
    elif args.model_name == "SENN_0_01":
        default_epochs = int(senn_cfg["epochs"])
        default_batch_size = int(senn_cfg.get("batch_size", 200))
        default_lr = float(senn_cfg["lr"])
    elif args.model_name == "B30":
        default_epochs = B30_DEFAULT_EPOCHS
        default_batch_size = B30_DEFAULT_BATCH_SIZE
        default_lr = B30_DEFAULT_LR
    else:
        raise ValueError(f"Unsupported model_name: {args.model_name}")

    default_save_root = resolve_default_save_root()
    default_num_workers = resolve_default_num_workers()

    adv_random_start = not args.no_adv_random_start

    return TrainConfig(
        model_name=args.model_name,
        variant=args.variant,
        seed=args.seed,

        epochs=args.epochs if args.epochs is not None else default_epochs,
        batch_size=args.batch_size if args.batch_size is not None else default_batch_size,
        lr=args.lr if args.lr is not None else default_lr,
        weight_decay=args.weight_decay,
        val_size=args.val_size,
        save_root=args.save_root if args.save_root is not None else default_save_root,
        num_workers=args.num_workers if args.num_workers is not None else default_num_workers,

        adv_eps=args.adv_eps,
        adv_alpha=args.adv_alpha,
        adv_steps=args.adv_steps,
        adv_random_start=adv_random_start,

        # These are kept for completeness even though alternating updates
        # already separate clean and adversarial branches.
        lambda_clean=args.lambda_clean,
        lambda_adv=args.lambda_adv,

        # Optional extra low-epsilon adversarial branch
        adv_low_eps=args.adv_low_eps,
        adv_low_alpha=(args.adv_low_alpha if args.adv_low_eps is not None else None),
        adv_low_steps=(args.adv_low_steps if args.adv_low_eps is not None else None),
        lambda_adv_low=args.lambda_adv_low,

        # ProtoVAE clean loss coefficients
        proto_ce_coef=args.proto_ce_coef,
        proto_recon_coef=args.proto_recon_coef,
        proto_kl_coef=args.proto_kl_coef,
        proto_ortho_coef=args.proto_ortho_coef,

        # SENN coefficients
        senn_concept_reg=args.senn_concept_reg if args.senn_concept_reg is not None else float(senn_cfg["concept_reg"]),
        senn_robust_reg=args.senn_robust_reg if args.senn_robust_reg is not None else float(senn_cfg["robust_reg"]),
        senn_sparsity_reg=args.senn_sparsity_reg if args.senn_sparsity_reg is not None else float(senn_cfg["sparsity_reg"]),

        # B30 coefficients
        b30_lambda_class=args.b30_lambda_class if args.b30_lambda_class is not None else B30_DEFAULT_LAMBDA_CLASS,
        b30_lambda_ae=args.b30_lambda_ae if args.b30_lambda_ae is not None else B30_DEFAULT_LAMBDA_AE,
        b30_lambda_1=args.b30_lambda_1 if args.b30_lambda_1 is not None else B30_DEFAULT_LAMBDA_1,
        b30_lambda_clus=args.b30_lambda_clus if args.b30_lambda_clus is not None else B30_DEFAULT_LAMBDA_CLUS,
        b30_lambda_sep=args.b30_lambda_sep if args.b30_lambda_sep is not None else B30_DEFAULT_LAMBDA_SEP,
        
        # Early stopping parameters
        early_stopping_patience=args.early_stopping_patience,
        run_tag=args.run_tag,

        # Reconstruction-Risk Shield hyperparameters
        shield_gamma=args.shield_gamma,
        shield_power=args.shield_power,
        shield_use_shift=args.shield_use_shift,
    )


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, required=True, choices=["ProtoVAE", "SENN_0_01", "B30"])
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--early_stopping_patience", type=int, default=10)

    # Environment-aware default: Kaggle -> /kaggle/working, Colab -> /content, else local ./results
    parser.add_argument("--save_root", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    # Adversarial training parameters
    parser.add_argument("--adv_eps", type=float, default=SHARED_ADV_EPS)
    parser.add_argument("--adv_alpha", type=float, default=SHARED_ADV_ALPHA)
    parser.add_argument("--adv_steps", type=int, default=SHARED_ADV_STEPS)
    parser.add_argument("--no_adv_random_start", action="store_true")

    # Optional extra low-epsilon adversarial branch (disabled by default)
    parser.add_argument("--adv_low_eps", type=float, default=None)
    parser.add_argument("--adv_low_alpha", type=float, default=None)
    parser.add_argument("--adv_low_steps", type=int, default=None)

    # Global branch weights (left exposed for flexibility)
    parser.add_argument("--lambda_clean", type=float, default=1.0)
    parser.add_argument("--lambda_adv", type=float, default=1.0)
    parser.add_argument("--lambda_adv_low", type=float, default=1.0)

    # ProtoVAE coefficients (original defaults are 1.0)
    parser.add_argument("--proto_ce_coef", type=float, default=1.0)
    parser.add_argument("--proto_recon_coef", type=float, default=1.0)
    parser.add_argument("--proto_kl_coef", type=float, default=1.0)
    parser.add_argument("--proto_ortho_coef", type=float, default=1.0)

    # SENN coefficients (fallback to config if omitted)
    parser.add_argument("--senn_concept_reg", type=float, default=None)
    parser.add_argument("--senn_robust_reg", type=float, default=None)
    parser.add_argument("--senn_sparsity_reg", type=float, default=None)

    # B30 coefficients (fallback to B30 config defaults if omitted)
    parser.add_argument("--b30_lambda_class", type=float, default=None)
    parser.add_argument("--b30_lambda_ae", type=float, default=None)
    parser.add_argument("--b30_lambda_1", type=float, default=None)
    parser.add_argument("--b30_lambda_clus", type=float, default=None)
    parser.add_argument("--b30_lambda_sep", type=float, default=None)

    # Reconstruction-Risk Shield parameters
    parser.add_argument("--shield_gamma", type=float, default=25.0)
    parser.add_argument("--shield_power", type=int, default=3)
    parser.add_argument("--shield_use_shift", type=parse_bool, default=True)

    # Optional tag to avoid overwriting runs with the same variant/seed
    parser.add_argument("--run_tag", type=str, default="")

    parser.add_argument(
        "--save_n_frames", type=int, default=0,
        help="Save exactly N lightweight pca_frame checkpoints spaced between "
         "epoch 0 and best_epoch (determined after early stopping). 0 = disabled.",
    )

    args = parser.parse_args()

    validate_variant(args.model_name, args.variant)
    cfg = build_config(args)
    if cfg.adv_low_eps is not None:
        if cfg.adv_low_alpha is None:
            cfg.adv_low_alpha = cfg.adv_alpha
        if cfg.adv_low_steps is None:
            cfg.adv_low_steps = cfg.adv_steps
    set_seed(cfg.seed)

    run_leaf = f"seed={cfg.seed}" if not cfg.run_tag else f"seed={cfg.seed}_{cfg.run_tag}"
    run_dir = Path(cfg.save_root) / cfg.model_name / cfg.variant / run_leaf
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(ckpt_dir)

    device = torch.device(cfg.device)
    pin_memory = torch.cuda.is_available()

    train_loader, val_loader = get_train_val_loader(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        random_seed=cfg.seed,
        val_size=cfg.val_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    test_loader = get_test_loader(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    model = load_model(cfg.model_name, device)
    apply_freeze_plan(model, cfg.model_name, cfg.variant)
    if cfg.variant == "B30-FT-Shield":
        print(
            ">>> Wrapping model with ReconstructionRiskWrapper "
            f"(Gamma={cfg.shield_gamma}, Power={cfg.shield_power}, "
            f"Shift={cfg.shield_use_shift}) <<<"
        )
        model = ReconstructionRiskWrapper(
            model,
            gamma=cfg.shield_gamma,
            power=cfg.shield_power,
            use_shift=cfg.shield_use_shift,
        ).to(device)

    total_params, trainable_params = count_parameters(model)
    save_trainable_parameter_names(model, run_dir / "trainable_parameters.txt")
    save_json(run_dir / "config.json", asdict(cfg))
    save_config_txt(run_dir / "config.txt", cfg, trainable_params, total_params)

    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    print(f"Model:            {cfg.model_name}")
    print(f"Variant:          {cfg.variant}")
    print(f"Seed:             {cfg.seed}")
    print(f"Device:           {cfg.device}")
    print(f"Num workers:      {cfg.num_workers}")
    print(f"Total params:     {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    history: List[Dict] = []
    best_val_adv_acc = -1.0
    best_epoch = -1          
    patience_counter = 0

    best_ckpt_name = f"{cfg.model_name}_{cfg.variant}_seed{cfg.seed}_best_val_adv_acc.pth"
    best_ckpt_path = ckpt_dir / best_ckpt_name

    start_time = time.time()

    # Save base model as epoch 0 before any fine-tuning.
    if args.save_n_frames > 0:
        torch.save(
            {"epoch": 0, "model_state": model.state_dict()},
            ckpt_dir / "all_epoch_000.pth",
        )
    
    for epoch in range(1, cfg.epochs + 1):
        epochs_run = epoch
        train_metrics = run_epoch(model, train_loader, cfg, optimizer)
        val_metrics = run_epoch(model, val_loader, cfg, optimizer=None)

        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(row)

        if val_metrics["adv_acc"] > best_val_adv_acc:
            best_val_adv_acc = val_metrics["adv_acc"]
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_name": cfg.model_name,
                    "variant": cfg.variant,
                    "seed": cfg.seed,
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_adv_acc": best_val_adv_acc,
                    "config": asdict(cfg),
                },
                best_ckpt_path,
            )
            
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs without improvement.")
                break
        
        # Lightweight checkpoint for every epoch (PCA frame candidates).
        if args.save_n_frames > 0:
            torch.save(
                {"epoch": epoch, "model_state": model.state_dict()},
                ckpt_dir / f"all_epoch_{epoch:03d}.pth",
            )
    
    total_runtime_s = time.time() - start_time

    # --- PCA frame selection (post-hoc, based on actual best_epoch) ----------
    if args.save_n_frames > 0:

        all_epoch_ckpts = sorted(
            [
                p for p in ckpt_dir.glob("all_epoch_*.pth")
                if 1 <= int(p.stem.replace("all_epoch_", "")) <= best_epoch
            ],
            key=lambda p: int(p.stem.replace("all_epoch_", "")),
        )
        
        # Delete checkpoints beyond best_epoch (not useful for PCA).
        for p in ckpt_dir.glob("all_epoch_*.pth"):
            epoch_num = int(p.stem.replace("all_epoch_", ""))
            if epoch_num > best_epoch:
                p.unlink()

        # Epoch 0 is always served by CKPT_PATHS in load_pca_frame_paths.
        ep0 = ckpt_dir / "all_epoch_000.pth"
        if ep0.exists():
            ep0.unlink()

        if all_epoch_ckpts:
            # Select n_frames indices with linspace from 0 to best_epoch.
            # Epoch 0 = base model (CKPT_PATHS["B30"]), already available.
            # So here we select from epoch 1..best_epoch.
            n_available = len(all_epoch_ckpts)
            n_select    = min(args.save_n_frames, n_available)
            selected_idx = set(
                int(round(i)) for i in np.linspace(0, n_available - 1, n_select)
            )

            for i, p in enumerate(all_epoch_ckpts):
                if i in selected_idx:
                    epoch_num  = int(p.stem.replace("all_epoch_", ""))
                    frame_path = ckpt_dir / f"pca_frame_epoch_{epoch_num:03d}.pth"
                    p.rename(frame_path)
                    print(f"  PCA frame saved: {frame_path.name}")
                else:
                    p.unlink()   # delete non-selected checkpoint

            print(f"  PCA frames: {n_select} frames up to best_epoch={best_epoch}.")

    best_state = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_state["model_state"])

    test_metrics = run_epoch(model, test_loader, cfg, optimizer=None)

    summary = {
        "model_name": cfg.model_name,
        "variant": cfg.variant,
        "seed": cfg.seed,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "best_val_adv_acc": best_val_adv_acc,
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "runtime_s": total_runtime_s,
        "best_checkpoint_path": str(best_ckpt_path),
        "config": asdict(cfg),
    }

    for k, v in test_metrics.items():
        summary[f"test_{k}"] = v

    save_json(run_dir / "history.json", {"history": history})
    save_history_csv(run_dir / "history.csv", history)
    save_json(run_dir / "summary.json", summary)

    print("Done.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
