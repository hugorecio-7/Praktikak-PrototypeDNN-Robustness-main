import torch
import math
from autoattack.autoattack import AutoAttack 
import eagerpy as ep
import foolbox as fb
from ProtoVAE.model import ProtoVAE

def FSGM_attack(batch_x, loss_f, eps):
    """
    Performs the Fast Sign Gradient Method (FSGM) attack on the input batch_x.

    Args:
        batch_x (torch.Tensor): The input batch of images.
        loss_f (callable): The loss function used to compute the loss.
        eps (float): The magnitude of the perturbation.
        
    Returns:
        torch.Tensor: The perturbed batch of images.

    """
    batch_x = batch_x.clone().detach().requires_grad_(True)
    
    loss = loss_f(batch_x = batch_x)
    
    input_gradients = torch.autograd.grad(loss, batch_x)[0]
        
    input_gradient_sign = torch.sign(input_gradients)

    perturbed_batch_x = torch.clamp(batch_x + eps * input_gradient_sign, min=0, max=1).detach_()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define the device
    
    perturbed_batch_x = perturbed_batch_x.to(device)

    return perturbed_batch_x

# Adapted version of https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgd.py 
def PGDLInf_attack(batch_x, loss_f, iters, eps, alpha, random_start):
    """
    Performs the Projected Gradient Descent (PGD) attack with L-infinity norm on a batch of input images.

    Args:
        batch_x (torch.Tensor): The batch of input images.
        loss_f (callable): The loss function to maximize.
        iters (int): The number of iterations for the attack.
        eps (float): The maximum perturbation allowed for each pixel.
        alpha (float): The step size for each iteration of the attack.
        random_start (bool): Whether to start the attack from a random point.

    Returns:
        torch.Tensor: The perturbed batch of input images.

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define the device
    
    ori_images = batch_x.clone().detach().to(device).to(device)
    perturbed_batch_x = batch_x.clone().detach().to(device)
    
    if random_start:
        # Starting at a uniformly random point
        perturbed_batch_x = perturbed_batch_x + torch.empty_like(perturbed_batch_x).uniform_(
            -eps, eps
        )

        perturbed_batch_x = torch.clamp(perturbed_batch_x, min=0, max=1).detach()
    
    for _ in range(iters):
        perturbed_batch_x.requires_grad = True
        
        loss = loss_f(batch_x = perturbed_batch_x)
        
        input_gradients = torch.autograd.grad(loss, perturbed_batch_x)[0]
        input_gradient_sign = torch.sign(input_gradients)

        perturbed_batch_x = perturbed_batch_x.detach() + alpha * input_gradient_sign
        delta = torch.clamp(perturbed_batch_x - ori_images, min=-eps, max=eps)
        # if norm:
        #     perturbed_batch_x = torch.clamp(ori_images + delta, min=-1, max=1).detach() 
        # else:
        perturbed_batch_x = torch.clamp(ori_images + delta, min=0, max=1).detach()
        
    return perturbed_batch_x

#Adapted version of https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgdl2.py
def PGDL2_attack(batch_x, loss_f, iters, eps, alpha, random_start):
    """
    Performs the PGD with L2 norm adversarial attack on a batch of input images.

    Args:
        batch_x (torch.Tensor): The batch of input images.
        loss_f (callable): The loss function used to compute the loss.
        iters (int): The number of iterations for the attack.
        eps (float): The maximum perturbation allowed for each pixel.
        alpha (float): The step size for each iteration of the attack.
        random_start (bool): Whether to start the attack from a random point.

    Returns:
        torch.Tensor: The perturbed batch of input images.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    
    eps_for_division=1e-10
    batch_size = len(batch_x)
    
    ori_images = batch_x.clone().detach().to(device)
    perturbed_batch_x = batch_x.clone().detach().to(device)
    
    if random_start:
            # Starting at a uniformly random point
            delta = torch.empty_like(perturbed_batch_x).normal_()
            d_flat = delta.view(perturbed_batch_x.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(perturbed_batch_x.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * eps
            perturbed_batch_x = torch.clamp(perturbed_batch_x + delta, min=0, max=1).detach()
    
    for _ in range(iters):
        
        perturbed_batch_x.requires_grad = True
        
        loss = loss_f(batch_x=perturbed_batch_x)
        
        # Update adversarial images
        grad = torch.autograd.grad(
            loss, perturbed_batch_x, retain_graph=False, create_graph=False
        )[0]
        grad_norms = (
            torch.norm(grad.view(batch_size, -1), p=2, dim=1)
            + eps_for_division
        )  # nopep8
        grad = grad / grad_norms.view(batch_size, 1, 1, 1)
        perturbed_batch_x = perturbed_batch_x.detach() + alpha * grad

        delta = perturbed_batch_x - ori_images
        delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
        factor = eps / delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))
        delta = delta * factor.view(-1, 1, 1, 1)

        perturbed_batch_x = torch.clamp(ori_images + delta, min=0, max=1).detach()
            
    return perturbed_batch_x

def LinfDeepFool_attack(batch_x, batch_y, model, steps=50, candidates=10, overshoot=1.02, epsilon=0.3):
    """
    Applies the LinfDeepFool attack from Foolbox to a batch of images.

    Args:
        batch_x (torch.Tensor): Batch of input images.
        batch_y (torch.Tensor): True labels of the input batch.
        model (torch.nn.Module): PyTorch model to attack.
        steps (int): Maximum number of iterations for the attack.
        candidates (int): Number of candidates for the DeepFool attack.
        overshoot (float): Overshoot factor for the perturbation.
        epsilon (float): The epsilon value to scale the perturbations.

    Returns:
        torch.Tensor: Perturbed images.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model and data to CPU if they are not already
    model = model.to(device).eval() # Ensure model is on CPU and in eval mode
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    
    
    # Convert the PyTorch model to Foolbox
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    # Create the attack
    attack = fb.attacks.LinfDeepFoolAttack(steps=steps, candidates=candidates, overshoot=overshoot)
    try:
        raw, clipped, is_adv = attack(fmodel, batch_x, batch_y, epsilons=epsilon)
    except Exception as e:
        print(f"Attack failed: {e}")
        return batch_x  # Fallback to original input

    return  clipped

def L2DeepFool_attack(batch_x, batch_y, model, steps=50, candidates=10, overshoot=1.02, epsilon=0.3):
    """
    Applies the L2DeepFool_attack attack from Foolbox to a batch of images.

    Args:
        batch_x (torch.Tensor): Batch of input images.
        batch_y (torch.Tensor): True labels of the input batch.
        model (torch.nn.Module): PyTorch model to attack.
        steps (int): Maximum number of iterations for the attack.
        candidates (int): Number of candidates for the DeepFool attack.
        overshoot (float): Overshoot factor for the perturbation.
        epsilon (float): The epsilon value to scale the perturbations.

    Returns:
        torch.Tensor: Perturbed images.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model and data to CPU if they are not already
    model = model.to(device).eval() # Ensure model is on CPU and in eval mode
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    
    
    # Convert the PyTorch model to Foolbox
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    # Create the attack
    attack = fb.attacks.L2DeepFoolAttack(steps=steps, candidates=candidates, overshoot=overshoot)
    
    try:
        raw, clipped, is_adv = attack(fmodel, batch_x, batch_y, epsilons=epsilon)
    except Exception as e:
        print(f"Attack failed: {e}")
        return batch_x
    
    return  clipped

def LinfAdditiveUniformNoise_attack(batch_x, batch_y, model, epsilon=0.3):
    """
    Applies the LinfAdditiveUniformNoise attack from Foolbox to a batch of images.

    Args:
        batch_x (torch.Tensor): Batch of input images.
        batch_y (torch.Tensor): True labels of the input batch.
        model (torch.nn.Module): PyTorch model to attack.
        epsilon (float): The epsilon value to scale the perturbations.

    Returns:
        PyTorchTensor: Perturbed images.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_x.to(device)
    batch_y.to(device)
    model = model.to(device).eval()
    
    # Convert the PyTorch model to Foolbox
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    # Define attack
    attack = fb.attacks.LinfAdditiveUniformNoiseAttack()

    # Run attack
    try:
        raw, clipped, is_adv = attack(fmodel, batch_x, batch_y, epsilons=epsilon)
    except Exception as e:
        print(f"Attack failed: {e}")
        return batch_x  # Fallback to original input
    
    return  clipped

def LinfBasicIterative_attack(batch_x, batch_y, model, steps=50, epsilon=0.3, random_start=True):
    """
    Applies the LinfBasicIterative_attack attack from Foolbox to a batch of images.

    Args:
        batch_x (torch.Tensor): Batch of input images.
        batch_y (torch.Tensor): True labels of the input batch.
        model (torch.nn.Module): PyTorch model to attack.
        steps (int): Maximum number of iterations for the attack.
        epsilon (float): The epsilon value to scale the perturbations.
        random_start (bool): Whether to start the attack from a random point.

    Returns:
        torch.Tensor: Perturbed images.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move model and data to CPU if they are not already
    model = model.to(device).eval() # Ensure model is on CPU and in eval mode
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    
    # Convert the PyTorch model to Foolbox
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    # Create the attack
    attack = fb.attacks.LinfBasicIterativeAttack(steps=steps, random_start=random_start)
    
    try:
        raw, clipped, is_adv = attack(fmodel, batch_x, batch_y, epsilons=epsilon)
    except Exception as e:
        print(f"Attack failed: {e}")
        return batch_x  # Fallback to original input
    
    return  clipped

def LinfFMNA_attack(batch_x, batch_y, model, binary_search_steps, steps=100, max_stepsize=2, min_stepsize=1e-3, gamma=0.1, epsilon=0.3):
    """
    Applies the LinfFMNA_attack attack from Foolbox to a batch of images.

    Args:
        batch_x (torch.Tensor): Batch of input images.
        batch_y (torch.Tensor): True labels of the input batch.
        model (torch.nn.Module): PyTorch model to attack.
        binary_search_steps (int): Number of binary search steps for the attack.
        steps (int): Maximum number of iterations for the attack.
        max_stepsize (float): Maximum step size for the attack.
        min_stepsize (float): Minimum step size for the attack.
        gamma (float): Scaling factor for the perturbations.
        epsilon (float): The epsilon value to scale the perturbations.

    Returns:
        torch.Tensor: Perturbed images.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move model and data to CPU if they are not already
    model = model.to(device).eval() # Ensure model is on CPU and in eval mode
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    
    # Convert the PyTorch model to Foolbox
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    # Create the attack
    attack = fb.attacks.LInfFMNAttack(steps=steps, max_stepsize=max_stepsize, min_stepsize=min_stepsize, gamma=gamma, binary_search_steps=binary_search_steps)
    
    try:
        raw, clipped, is_adv = attack(fmodel, batch_x, batch_y, epsilons=epsilon)
    except Exception as e:
        print(f"Attack failed: {e}")
        return batch_x  # Fallback to original input
    
    return  clipped

def L2FMNA_attack(batch_x, batch_y, model, binary_search_steps, steps=100, max_stepsize=2, min_stepsize=1e-3, gamma=0.1, epsilon=0.3):
    """
    Applies the L2fFMNA_attack attack from Foolbox to a batch of images.

    Args:
        batch_x (torch.Tensor): Batch of input images.
        batch_y (torch.Tensor): True labels of the input batch.
        model (torch.nn.Module): PyTorch model to attack.
        binary_search_steps (int): Number of binary search steps for the attack.
        steps (int): Maximum number of iterations for the attack.
        max_stepsize (float): Maximum step size for the attack.
        min_stepsize (float): Minimum step size for the attack.
        gamma (float): Scaling factor for the perturbations.
        epsilon (float): The epsilon value to scale the perturbations.

    Returns:
        torch.Tensor: Perturbed images.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move model and data to CPU if they are not already
    model = model.to(device).eval() # Ensure model is on CPU and in eval mode
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    
    # Convert the PyTorch model to Foolbox
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    # Create the attack
    attack = fb.attacks.L2FMNAttack(steps=steps, max_stepsize=max_stepsize, min_stepsize=min_stepsize, gamma=gamma, binary_search_steps=binary_search_steps)
    
    try:
        raw, clipped, is_adv = attack(fmodel, batch_x, batch_y, epsilons=epsilon)
    except Exception as e:
        print(f"Attack failed: {e}")
        return batch_x  # Fallback to original input
    
    return  clipped

def LinfMomentumIterativeFastGradient_attack(batch_x, batch_y, model, steps=100, epsilon=0.3):
    """
    Applies the LinfMomentumIterativeFastGradient attack from Foolbox to a batch of images.

    Args:
        batch_x (torch.Tensor): Batch of input images.
        batch_y (torch.Tensor): True labels of the input batch.
        model (torch.nn.Module): PyTorch model to attack.
        steps (int): Maximum number of iterations for the attack.
        epsilon (float): The epsilon value to scale the perturbations.

    Returns:
        torch.Tensor: Perturbed images.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move model and data to CPU if they are not already
    model = model.to(device).eval() # Ensure model is on CPU and in eval mode
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    
    # Convert the PyTorch model to Foolbox
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    # Create the attack
    attack = fb.attacks.LinfMomentumIterativeFastGradientMethod(steps=steps)
    
    try:
        raw, clipped, is_adv = attack(fmodel, batch_x, batch_y, epsilons=epsilon)
    except Exception as e:
        print(f"Attack failed: {e}")
        return batch_x  # Fallback to original input
    
    return  clipped

def LinfAdamProjectedGradientDescent_attack(batch_x, batch_y, model, steps=100, epsilon=0.3, random_start=True):
    """
    Applies the LinfAdamProjectedGradientDescent attack from Foolbox to a batch of images.

    Args:
        batch_x (torch.Tensor): Batch of input images.
        batch_y (torch.Tensor): True labels of the input batch.
        model (torch.nn.Module): PyTorch model to attack.
        steps (int): Maximum number of iterations for the attack.
        epsilon (float): The epsilon value to scale the perturbations.
        random_start (bool): Whether to start the attack from a random point.

    Returns:
        torch.Tensor: Perturbed images.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Move model and data to CPU if they are not already
    model = model.to(device).eval() # Ensure model is on CPU and in eval mode
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    
    # Convert the PyTorch model to Foolbox
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    # Create the attack
    attack = fb.attacks.LinfAdamProjectedGradientDescentAttack(steps=steps, random_start=random_start)
    
    try:
        raw, clipped, is_adv = attack(fmodel, batch_x, batch_y, epsilons=epsilon)
    except Exception as e:
        print(f"Attack failed: {e}")
        return batch_x  # Fallback to original input
    
    return  clipped 

def AutoAttack_adv(batch_x, batch_y, model, epsilon=0.03, version='standard'):
    """
    Applies AutoAttack to a batch of images.
    Args:
        batch_x (torch.Tensor): Input images (shape [B, C, H, W]).
        batch_y (torch.Tensor): True labels.
        model (torch.nn.Module): PyTorch model (outputs logits).
        epsilon (float): Perturbation budget (Linf norm).
        version (str): Version of AutoAttack to use ('standard', 'random', etc.).
        is_tf (bool): If True, indicates the model is a TensorFlow model (for compatibility with utils_tf2).
        
    Returns:
        torch.Tensor: Adversarial examples.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Explicitly set device to CPU

    # Move model and data to CPU if they are not already
    model = model.to(device).eval() # Ensure model is on CPU and in eval mode
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)

    # Initialize AutoAttack
    adversary = AutoAttack(
        model = model,
        norm='Linf',
        eps=epsilon,
        version=version,
        is_tf_model=False,
        device=device, 
        verbose=False
    )

    try:
        # Run the attack
        x_adv = adversary.run_standard_evaluation(batch_x, batch_y)
    except Exception as e:
        print(f"AutoAttack failed: {e}")
        x_adv = batch_x  # Fallback to original inputs

    return x_adv

def PatchPGD_attack(batch_x, loss_f, eps, steps=40, step_size=0.1, random_start=True, restarts=1, base_seed=None, batch_idx=None):
    """
    Masked PGD patch attack (per-image, random location, untargeted, no EoT).
    
    Args:
        batch_x: (B,C,H,W) input batch
        loss_f: function that takes batch_x as keyword and returns a scalar loss to maximize
        eps: patch area ratio (e.g. 0.1 for 10% area patch)
        steps: PGD steps
        step_size: PGD step size (relative to [0,1] range)
        random_start: whether to start at a random point inside the patch
        restarts: number of random patch locations to try
        base_seed: optional seed for reproducibility (affects patch location and random start)
        batch_idx: optional batch index for reproducibility (if base_seed is given, it will be mixed with batch_idx to get different patch locations per batch)
    Returns:
        batch_x_adv: (B,C,H,W) adversarial batch with patch perturbation
    """
    if eps <= 0:
        return batch_x.detach()

    device = batch_x.device
    x = batch_x.detach()
    B, C, H, W = x.shape

    # Bounds 
    vmin, vmax = (0.0, 1.0)

    # Patch side length from area ratio
    ratio = float(eps)
    side = int(round(math.sqrt(max(ratio, 0.0) * H * W)))
    side = max(1, min(side, min(H, W)))

    # Reproducible RNG (optional)
    # Location RNG should be stable for a given (seed, ratio, batch_idx)
    g_loc = None
    if base_seed is not None:
        mix = int(base_seed) * 1_000_003 + int(ratio * 1000) * 1_009
        if batch_idx is not None:
            mix += int(batch_idx) * 9_973
        g_loc = torch.Generator(device=device)
        g_loc.manual_seed(mix % (2**31 - 1))

    # Random patch position per image
    max_top = H - side
    max_left = W - side
    if g_loc is None:
        tops = torch.randint(0, max_top + 1, (B,), device=device)
        lefts = torch.randint(0, max_left + 1, (B,), device=device)
    else:
        tops = torch.randint(0, max_top + 1, (B,), device=device, generator=g_loc)
        lefts = torch.randint(0, max_left + 1, (B,), device=device, generator=g_loc)

    # Build mask (B,C,H,W)
    mask = torch.zeros((B, 1, H, W), device=device, dtype=x.dtype)
    for i in range(B):
        t = int(tops[i].item())
        l = int(lefts[i].item())
        mask[i, :, t : t + side, l : l + side] = 1.0
    mask = mask.expand(B, C, H, W)

    best_x_adv = None
    best_loss = None

    for r in range(int(restarts)):
        # RNG for random start per restart (optional)
        g_init = None
        if base_seed is not None:
            mix_r = (int(base_seed) * 1_000_003 + int(ratio * 1000) * 1_009)
            if batch_idx is not None:
                mix_r += int(batch_idx) * 9_973
            mix_r += r * 12_345
            g_init = torch.Generator(device=device)
            g_init.manual_seed(mix_r % (2**31 - 1))

        x_adv = x.clone()

        # Random start only inside the patch
        if random_start:
            if g_init is None:
                rand = torch.rand(x_adv.shape, device=device, dtype=x.dtype)
            else:
                rand = torch.rand(x_adv.shape, device=device, dtype=x.dtype, generator=g_init)
            rand = rand * (vmax - vmin) + vmin
            x_adv = x_adv * (1.0 - mask) + rand * mask
            x_adv = torch.clamp(x_adv, vmin, vmax)
            x_adv = x + (x_adv - x) * mask  # keep outside patch identical

        for _ in range(int(steps)):
            x_adv = x_adv.detach().requires_grad_(True)

            # loss_f is expected to accept batch_x as keyword (matches your CELoss signature)
            loss_val = loss_f(batch_x=x_adv)
            grad = torch.autograd.grad(loss_val, x_adv, only_inputs=True)[0]

            # Update only inside patch
            x_adv = x_adv + float(step_size) * torch.sign(grad) * mask

            # Clamp to valid bounds
            x_adv = torch.clamp(x_adv, vmin, vmax)

            # Enforce: outside patch remains unchanged
            x_adv = x + (x_adv - x) * mask

        with torch.no_grad():
            final_loss = loss_f(batch_x=x_adv).detach()

        if best_loss is None or final_loss > best_loss:
            best_loss = final_loss
            best_x_adv = x_adv.detach()

    return best_x_adv
