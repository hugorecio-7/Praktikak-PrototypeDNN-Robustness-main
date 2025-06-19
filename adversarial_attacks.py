import torch
from autoattack.autoattack import AutoAttack
from autoattack import utils_tf2   
import eagerpy as ep
import foolbox as fb
from ProtoVAE.model import ProtoVAE
from Prob_PSENN.ProbPSENN import ProbPSENN, ProbPSENN_VAE

def FSGM_attack(batch_x, loss_f, eps, norm=False):
    """
    Performs the Fast Sign Gradient Method (FSGM) attack on the input batch_x.

    Args:
        batch_x (torch.Tensor): The input batch of images.
        loss_f (callable): The loss function used to compute the loss.
        eps (float): The magnitude of the perturbation.
        norm (bool): If True, the perturbation is clamped to [-1, 1]. If False, it is clamped to [0, 1].
        
    Returns:
        torch.Tensor: The perturbed batch of images.

    """
    batch_x = batch_x.clone().detach().requires_grad_(True)
    
    loss = loss_f(batch_x = batch_x)
    
    input_gradients = torch.autograd.grad(loss, batch_x)[0]
        
    input_gradient_sign = torch.sign(input_gradients)

    if norm:
        perturbed_batch_x = torch.clamp(batch_x + eps * input_gradient_sign, min=-1, max=1).detach_()
    else:
        perturbed_batch_x = torch.clamp(batch_x + eps * input_gradient_sign, min=0, max=1).detach_()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define the device
    
    perturbed_batch_x = perturbed_batch_x.to(device)

    return perturbed_batch_x

# Adapted version of https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgd.py 
def PGDLInf_attack(batch_x, loss_f, iters, eps, alpha, random_start, norm=False):
    """
    Performs the Projected Gradient Descent (PGD) attack with L-infinity norm on a batch of input images.

    Args:
        batch_x (torch.Tensor): The batch of input images.
        loss_f (callable): The loss function to maximize.
        iters (int): The number of iterations for the attack.
        eps (float): The maximum perturbation allowed for each pixel.
        alpha (float): The step size for each iteration of the attack.
        random_start (bool): Whether to start the attack from a random point.
        norm (bool): If True, the perturbation is clamped to [-1, 1]. If False, it is clamped to [0, 1].

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
        if norm:
            perturbed_batch_x = torch.clamp(perturbed_batch_x, min=-1, max=1).detach()
        else:
            perturbed_batch_x = torch.clamp(perturbed_batch_x, min=0, max=1).detach()
    
    for _ in range(iters):
        perturbed_batch_x.requires_grad = True
        
        loss = loss_f(batch_x = perturbed_batch_x)
        
        input_gradients = torch.autograd.grad(loss, perturbed_batch_x)[0]
        input_gradient_sign = torch.sign(input_gradients)

        perturbed_batch_x = perturbed_batch_x.detach() + alpha * input_gradient_sign
        delta = torch.clamp(perturbed_batch_x - ori_images, min=-eps, max=eps)
        if norm:
            perturbed_batch_x = torch.clamp(ori_images + delta, min=-1, max=1).detach() 
        else:
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
    if model.__class__.__name__ in ['ProbPSENN_VAE', 'ProbPSENN', 'ProtoVAEWrapper']:
        fmodel = fb.PyTorchModel(model, bounds=(-1, 1))
    else:
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
    if model.__class__.__name__ in ['ProbPSENN_VAE', 'ProbPSENN', 'ProtoVAEWrapper']:
        fmodel = fb.PyTorchModel(model, bounds=(-1, 1))
    else:
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
    if model.__class__.__name__ in ['ProbPSENN_VAE', 'ProbPSENN', 'ProtoVAEWrapper']:
        fmodel = fb.PyTorchModel(model, bounds=(-1, 1))
    else:
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
        batch_y (torch.Tensor): True labels of the input batch.
        model (torch.nn.Module): PyTorch model to attack.
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
    if model.__class__.__name__ in ['ProbPSENN_VAE', 'ProbPSENN', 'ProtoVAEWrapper']:
        fmodel = fb.PyTorchModel(model, bounds=(-1, 1))
    else:
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
    if model.__class__.__name__ in ['ProbPSENN_VAE', 'ProbPSENN', 'ProtoVAEWrapper']:
        fmodel = fb.PyTorchModel(model, bounds=(-1, 1))
    else:
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
    if model.__class__.__name__ in ['ProbPSENN_VAE', 'ProbPSENN', 'ProtoVAEWrapper']:
        fmodel = fb.PyTorchModel(model, bounds=(-1, 1))
    else:
        fmodel = fb.PyTorchModel(model, bounds=(0, 1))

    # Create the attack
    attack = fb.attacks.LinfAdamProjectedGradientDescentAttack(steps=steps, random_start=random_start)
    
    try:
        raw, clipped, is_adv = attack(fmodel, batch_x, batch_y, epsilons=epsilon)
    except Exception as e:
        print(f"Attack failed: {e}")
        return batch_x  # Fallback to original input
    
    return  clipped 

# RescaledModel to adapt the model input range from [0, 1] to [-1, 1] 
class RescaledModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # Your actual model expecting [-1, 1] input

    def forward(self, x):
        # Rescale from [0, 1] to [-1, 1]
        x = x * 2 - 1
        return self.model(x)

def AutoAttack_adv(batch_x, batch_y, model, epsilon=0.03, version='standard', is_tf=False):
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
    if not is_tf:
        model = model.to(device).eval() # Ensure model is on CPU and in eval mode
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
    else:
        model = utils_tf2.ModelAdapter(model)
        
    if model.__class__.__name__ in ['ProbPSENN_VAE', 'ProbPSENN', 'ProtoVAEWrapper']:
        # Rescale model to expect inputs in [-1, 1] range
        model = RescaledModel(model)
        batch_x = (batch_x + 1) / 2

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

    if model.__class__.__name__ in ['RescaledModel']:
        x_adv = x_adv * 2 - 1

    return x_adv
