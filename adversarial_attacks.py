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

