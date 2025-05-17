import torch

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

import eagerpy as ep
import foolbox as fb

def LinfDeepFool_attack(batch_x, batch_y, model, steps=50, candidates=10, overshoot=1.02, epsilon=0.3):
    """
    Applies the LinfDeepFool attack from Foolbox to a batch of images.

    Args:
        batch_x (torch.Tensor): Batch of input images.
        model (torch.nn.Module): PyTorch model to attack.
        batch_y (torch.Tensor): True labels of the input batch.
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
    attack = fb.attacks.LinfDeepFoolAttack(steps=steps, candidates=candidates, overshoot=overshoot)
    
    try:
        raw, clipped, is_adv = attack(fmodel, batch_x, batch_y, epsilons=epsilon)
    except Exception as e:
        print(f"Attack failed: {e}")
        return batch_x  # Fallback to original input
    
    return  clipped

def LinfAdditiveUniformNoise_attack(batch_x, batch_y, model, epsilon=0.3):
    """
    Applies the LinfAdditiveUniformNoise attack from Foolbox to a batch of images.

    Args:
        batch_x (torch.Tensor): Batch of input images.
        model (torch.nn.Module): PyTorch model to attack.
        batch_y (torch.Tensor): True labels of the input batch.
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
        model (torch.nn.Module): PyTorch model to attack.
        batch_y (torch.Tensor): True labels of the input batch.
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
    attack = fb.attacks.LinfBasicIterativeAttack(steps=steps, random_start=random_start)
    
    try:
        raw, clipped, is_adv = attack(fmodel, batch_x, batch_y, epsilons=epsilon)
    except Exception as e:
        print(f"Attack failed: {e}")
        return batch_x  # Fallback to original input
    
    return  clipped

def LinfFMNA_attack(batch_x, batch_y, model, steps=100, max_stepsize=2, min_stepsize=1e-3, gamma=0.1, epsilon=0.3):
    """
    Applies the LinfFMNA_attack attack from Foolbox to a batch of images.

    Args:
        batch_x (torch.Tensor): Batch of input images.
        model (torch.nn.Module): PyTorch model to attack.
        batch_y (torch.Tensor): True labels of the input batch.
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
    attack = fb.attacks.LInfFMNAttack(steps=steps, max_stepsize=max_stepsize, min_stepsize=min_stepsize, gamma=gamma)
    
    try:
        raw, clipped, is_adv = attack(fmodel, batch_x, batch_y, epsilons=epsilon)
    except Exception as e:
        print(f"Attack failed: {e}")
        return batch_x  # Fallback to original input
    
    return  clipped

def LinfMomentumIterativeFastGradient_attack(batch_x, batch_y, model, steps=100, epsilon=0.3):
    """
    Applies the LinfBaseGradientDescent attack from Foolbox to a batch of images.

    Args:
        batch_x (torch.Tensor): Batch of input images.
        model (torch.nn.Module): PyTorch model to attack.
        batch_y (torch.Tensor): True labels of the input batch.
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

def LinfAdamProjectedGradientDescent_attack_foolbox(batch_x, batch_y, model, steps=100, epsilon=0.3, random_start=True):
    """
    Applies the LinfBaseGradientDescent attack from Foolbox to a batch of images.

    Args:
        batch_x (torch.Tensor): Batch of input images.
        model (torch.nn.Module): PyTorch model to attack.
        batch_y (torch.Tensor): True labels of the input batch.
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
    attack = fb.attacks.LinfAdamProjectedGradientDescentAttack(steps=steps, random_start=random_start)
    
    try:
        raw, clipped, is_adv = attack(fmodel, batch_x, batch_y, epsilons=epsilon)
    except Exception as e:
        print(f"Attack failed: {e}")
        return batch_x  # Fallback to original input
    
    return  clipped

from autoattack.autoattack import AutoAttack
from autoattack import utils_tf2    

def AutoAttack_adv(batch_x, batch_y, model, steps=50, epsilon=0.03, version='standard', is_tf=False):
    """
    Applies AutoAttack to a batch of images.
    Args:
        batch_x (torch.Tensor): Input images (shape [B, C, H, W]).
        batch_y (torch.Tensor): True labels.
        model (torch.nn.Module): PyTorch model (outputs logits).
        steps (int): Max iterations for APGD attacks.
        epsilon (float): Perturbation budget (Linf norm).
    Returns:
        torch.Tensor: Adversarial examples.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Explicitly set device to CPU

    # Move model and data to CPU if they are not already
    if not is_tf:
        model = model.to(device).eval() # Ensure model is on CPU and in eval mode
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
    # else:
        # model = utils_tf2.ModelAdapter(model)
        

    # Initialize AutoAttack
    adversary = AutoAttack(
        model = model,
        norm='Linf',
        eps=epsilon,
        version=version,
        is_tf_model=is_tf,
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
