import torch
import numpy as np
from functools import partial
from modules import Softmax
#from metrics import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
import foolbox as fb
import eagerpy as ep
import pandas as pd
from ProtoVAE.model import ProtoVAE
from Prob_PSENN.ProbPSENN import ProbPSENN, ProbPSENN_VAE

softmax = Softmax()

def show_adversarial_examples(model, batch_x, perturbed_batch_x, batch_y, pred_y, pred_y_adv, conf_y, conf_y_adv, n_examples, examples_type):
    """
    Display adversarial examples along with their closest prototypes.

    Args:
        model (torch.nn.Module): The trained model.
        batch_x (torch.Tensor): The original input batch.
        perturbed_batch_x (torch.Tensor): The perturbed input batch.
        batch_y (torch.Tensor): The true labels for the input batch.
        pred_y (torch.Tensor): The predicted labels for the input batch.
        pred_y_adv (torch.Tensor): The predicted labels for the perturbed input batch.
        conf_y (torch.Tensor): The confidence scores for the true labels in the input batch.
        conf_y_adv (torch.Tensor): The confidence scores for the predicted labels in the perturbed input batch.
        n_examples (int): The maximum number of examples to display.
        examples_type (str): The type of examples to display. Can be one of the following:
            - "cdp": Correctly classified and the closest prototype is different.
            - "csp": Correctly classified and the closest prototype is the same.
            - "isp": Incorrectly classified and the closest prototype is the same.
            - "idp": Incorrectly classified and the closest prototype is different.

    Returns:
        int: The total number of examples shown.
    """
    prototypes_to_show = 15
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Define the device
    model.prototype_layer.prototype_distances = model.prototype_layer.prototype_distances.to(device)
    prototype_distances = model.prototype_layer.prototype_distances
    n_prototypes = prototype_distances.size(0)
    prototype_imgs = model.decoder(prototype_distances.reshape((-1,10,2,2))).detach().cpu()

    n_cols = 5
    total_examples = 0  # Counter for total number of examples shown

    # Distances for normal and adversarial examples 
    distances = model.prototype_layer(model.encoder(batch_x).view(batch_x.size(0), -1))  
    distances_adv = model.prototype_layer(model.encoder(perturbed_batch_x).view(perturbed_batch_x.size(0), -1)) 
    for idx in range(batch_x.size(0)): 

        if n_examples == total_examples:
            break

        dists = distances[idx].detach().cpu().numpy()  

        # Sort prototypes by distances and keep only the closest 10
        sorted_prototypes = sorted(zip(prototype_imgs, dists), key=lambda x: x[1])[:prototypes_to_show]
        sorted_prototype_imgs, sorted_dists = zip(*sorted_prototypes)

        input_img_adv = perturbed_batch_x[idx].detach().cpu().numpy().reshape(batch_x.size(-2), batch_x.size(-1))
        dists_adv = distances_adv[idx].detach().cpu().numpy()  

        # Sort prototypes by distances for adversarial examples and keep only the closest 10
        sorted_prototypes_adv = sorted(zip(prototype_imgs, dists_adv), key=lambda x: x[1])[:prototypes_to_show]
        sorted_prototype_imgs_adv, sorted_dists_adv = zip(*sorted_prototypes_adv)

        if examples_type == "cdp":
            cond = pred_y_adv[idx] == batch_y[idx] and not torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0])
        elif examples_type == "csp":
            cond = pred_y_adv[idx] == batch_y[idx] and torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0])
        elif examples_type == "isp":
            cond = pred_y_adv[idx] != batch_y[idx] and torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0])
        elif examples_type == "idp":
            cond = pred_y_adv[idx] != batch_y[idx] and not torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0])

        if cond: 

            input_img = batch_x[idx].detach().cpu().numpy().reshape(batch_x.size(-2), batch_x.size(-1))

            gs = gridspec.GridSpec(1, 2, width_ratios=[1, n_cols], wspace=0.1)  

            ax0 = plt.subplot(gs[0])
            ax0.imshow(input_img, cmap='gray', interpolation='none')
            ax0.set_title("Standard")
            ax0.axis('off')
            ax0.text(0.5, -0.1, 'Y: {:.2f}  y: {:.2f}'.format(batch_y[idx], pred_y[idx]), horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes, fontsize=10)
            ax0.text(0.5, -0.4, 'Conf: {:.2f}'.format(conf_y[idx]), horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes, fontsize=10)

            # Calculate the number of rows needed for this image
            n_rows = np.ceil(prototypes_to_show / n_cols).astype(int)

            gs1 = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs[1], wspace=0.1, hspace=0.1)

            for p_idx in range(prototypes_to_show):  # Only show the closest 10 prototypes
                row = p_idx // n_cols
                col = p_idx % n_cols
                ax = plt.subplot(gs1[row, col])
                prototype_img = sorted_prototype_imgs[p_idx].detach().cpu().numpy().reshape(batch_x.size(-2), batch_x.size(-1))
                ax.imshow(prototype_img, cmap='gray', interpolation='none')

                ax.text(0.5, -0.1, f'{sorted_dists[p_idx]:.2f}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
                ax.axis('off')

            plt.tight_layout()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Further adjust space between plots
            plt.show()

            gs = gridspec.GridSpec(1, 2, width_ratios=[1, n_cols], wspace=0.1)  

            ax0 = plt.subplot(gs[0])
            ax0.imshow(input_img_adv, cmap='gray', interpolation='none')
            ax0.set_title("Adversarial")
            ax0.axis('off')
            ax0.text(0.5, -0.1, 'Y: {:.2f}  y: {:.2f}'.format(batch_y[idx], pred_y_adv[idx]), horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes, fontsize=10)
            ax0.text(0.5, -0.4, 'Conf: {:.2f}'.format(conf_y_adv[idx]), horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes, fontsize=10)

            n_rows = np.ceil(prototypes_to_show / n_cols).astype(int)

            gs1 = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs[1], wspace=0.1, hspace=0.1)

            for p_idx in range(prototypes_to_show):  # Only show the closest 10 prototypes
                row = p_idx // n_cols
                col = p_idx % n_cols
                ax = plt.subplot(gs1[row, col])
                prototype_img = sorted_prototype_imgs_adv[p_idx].detach().cpu().numpy().reshape(batch_x.size(-2), batch_x.size(-1))
                ax.imshow(prototype_img, cmap='gray', interpolation='none')

                ax.text(0.5, -0.1, f'{sorted_dists_adv[p_idx]:.2f}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
                ax.axis('off')

            plt.tight_layout()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Further adjust space between plots
            plt.show()
                
            total_examples += 1  # Increment the total examples counter

        if total_examples >= n_examples:  # Stop when the number of examples have been achieved
            break

    return total_examples

    
def attack_effects_test(model, test_loader, loss, attack):
    """
    Evaluates the effects of an attack on a given model using a test dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (torch.utils.data.DataLoader): The test dataset loader.
        loss (torch.nn.Module): The loss function to use for evaluation.
        attack (function): The attack function to generate adversarial examples.

    Returns:
        tuple: A tuple containing the test accuracy, adversarial test accuracy, and metric percentage.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device
    test_ac = 0
    test_ac_adv = 0
    n_test_batch = len(test_loader)
    
    objective = loss.keywords['objective']
    force_class = loss.keywords['force_class']
    change_expl = loss.keywords['change_expl']
    alpha1 = loss.keywords['alpha1']
    alpha2 = loss.keywords['alpha2']
    
    metric_percentage = 0
    
    thresh = -1.8 # We want to find prototypes for all the classes
    prototype_classes = np.where((model.fc.linear.weight.cpu().detach().numpy().T <= np.min(model.fc.linear.weight.cpu().detach().numpy().T, axis=1, keepdims=True)) | (model.fc.linear.weight.cpu().detach().numpy().T < thresh), 1, 0)
    
    for i, batch in enumerate(test_loader):
        batch_x = batch[0]
        batch_y = batch[1]
        batch_x = batch_x.to(device)
        batch_x.requires_grad = True
        batch_y = batch_y.to(device)

        # Get the predictions for the non-adversarial examples
        pred_y = model.forward(batch_x)
        pred_y = softmax(pred_y)

        # Generate adversarial examples from batch_x
        loss_f = partial(loss, batch_y=batch_y)
        adv_attack = partial(attack, loss_f=loss_f)
        perturbed_batch_x = adv_attack(batch_x)

        # Get the predictions for the adversarial examples
        pred_y_adv = model.forward(perturbed_batch_x)
        pred_y_adv = softmax(pred_y_adv)

        # test accuracy
        conf_y, max_indices = torch.max(pred_y,1)
        n = max_indices.size(0)
        test_ac += (max_indices == batch_y).sum(dtype=torch.float32)/n

        #adversarial test accuracy
        conf_y_adv, max_indices_adv = torch.max(pred_y_adv,1)
        n = max_indices_adv.size(0)
        test_ac_adv += (max_indices_adv == batch_y).sum(dtype=torch.float32)/n

        # Distances to prototypes for normal and adversarial examples 
        distances = model.prototype_layer(model.encoder(batch_x).view(batch_x.size(0), -1))  
        distances_adv = model.prototype_layer(model.encoder(perturbed_batch_x).view(perturbed_batch_x.size(0), -1)) 

        # For each example in the batch, verify if the attack has been effective or not
        for idx in range(batch_x.size(0)): 

            dists = distances[idx].detach().cpu().numpy()  
            dists_adv = distances_adv[idx].detach().cpu().numpy()  

            sorted_prototype_classes = sorted(zip(prototype_classes, dists), key=lambda x: x[1])
            sorted_prototype_classes_adv = sorted(zip(prototype_classes, dists_adv), key=lambda x: x[1])
            
            #Class of the closest prototype
            explanation_class = np.where(sorted_prototype_classes[0][0] == 1)[0][0]
            explanation_class_adv = np.where(sorted_prototype_classes_adv[0][0] == 1)[0][0]
            
            #If objective is change class and dont care about explanation
            if objective == "cecc" or objective == "necc" and alpha1 == 1 and alpha2 == 0:
                #If it was correctly classified before and now badly
                if  batch_y[idx] == max_indices[idx] and batch_y[idx] != max_indices_adv[idx]:
                    
                    if force_class is None:
                        metric_percentage += 1/n
                        
                    elif force_class is not None and force_class == max_indices_adv[idx]:
                        metric_percentage += 1/n
                        
            #If objective is change explanation and dont care about class
            elif objective == "cecc" or objective == "cenc" and alpha1 == 0 and alpha2 == 1:
                #If the explanation class has changed
                if explanation_class != explanation_class_adv:
                    
                    if change_expl is None:
                        metric_percentaje += 1/n
                        
                    elif change_expl is not None and explanation_class_adv == change_expl:
                        metric_percentage += 1/n
                
            elif objective == "cecc" and alpha1 == 1 and alpha2 == 1:
                #If the explanation class has changed and it was correctly classified before and now badly
                if explanation_class != explanation_class_adv and batch_y[idx] == max_indices[idx] \
                    and batch_y[idx] != max_indices_adv[idx]:
                    
                    if change_expl is None and force_class is None:
                        metric_percentaje += 1/n
                    
                    elif change_expl is not None and force_class is None \
                        and force_class == max_indices_adv[idx] and explanation_class_adv == change_expl:
                        metric_percentage += 1/n
            else:
                raise ValueError("You are not using a logic configuration, check")
                
    test_ac /= n_test_batch
    test_ac_adv /= n_test_batch
    metric_percentage /= n_test_batch 
    
    return test_ac, test_ac_adv, metric_percentage
    
def run_tests(model, test_loader, attack, loss_generator, num_runs=5):
    """
    Run multiple tests on a model using a given test loader and attack method.

    Args:
        model (torch.nn.Module): The model to be tested.
        test_loader (torch.utils.data.DataLoader): The data loader for the test set.
        attack (function): The attack method to be used.
        loss_generator (function): A function that generates the loss for the attack.
        num_runs (int, optional): The number of times to run the tests. Defaults to 5.

    Returns:
        tuple: A tuple containing three sub-tuples:
            - (test_ac_mean, test_ac_std): Mean and standard deviation of accuracy on the default test set.
            - (test_ac_adv_mean, test_ac_adv_std): Mean and standard deviation of accuracy on the adversarial test set.
            - (metric_percentage_mean, metric_percentage_std): Mean and standard deviation of the effectiveness percentage for the adversarial test set.
    """
    test_ac_list = []
    test_ac_adv_list = []
    metric_percentage_list = []
    
    # Run the tests num_runs times
    for _ in range(num_runs):
        loss = loss_generator()
        test_ac, test_ac_adv, metric_percentage = attack_effects_test(model, test_loader, attack=attack, loss=loss)
        test_ac_list.append(test_ac.cpu().numpy())
        test_ac_adv_list.append(test_ac_adv.cpu().numpy())
        metric_percentage_list.append(metric_percentage)  
    
    # Calculate the mean and standard deviation for the default test set
    test_ac_mean = np.mean(test_ac_list)
    test_ac_std = np.std(test_ac_list)
    
    # Calculate the mean and standard deviation for the adversarial test set
    test_ac_adv_mean = np.mean(test_ac_adv_list)
    test_ac_adv_std = np.std(test_ac_adv_list)
    
    # Calculate the mean and standard deviation for the effectiveness percentage for the adversarial test set
    metric_percentage_mean = np.mean(metric_percentage_list)
    metric_percentage_std = np.std(metric_percentage_list)
    
    return (test_ac_mean, test_ac_std), (test_ac_adv_mean, test_ac_adv_std), (metric_percentage_mean, metric_percentage_std)

def adversarial_attacks_eps_plot(models, model_names, test_loader, attack, loss, max_eps, step=0.025, foolbox_use=False):
    """
    Plots the accuracy of models under adversarial attacks for different epsilon values.

    Args:
        models (list): A list of PyTorch models.
        model_names (list): A list of names corresponding to the models.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        attack (function): The adversarial attack function.
        loss (function): The loss function used for the attack.
        max_eps (int or float): The maximum epsilon value for the attack.
        step (float, optional): The step size for epsilon values. Defaults to 0.025.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device
    dim = max_eps+1 if isinstance(max_eps, int) else int(max_eps/step) + 1
    
    results = np.zeros((len(models), dim))

    i=0
    for i, batch in enumerate(test_loader):
        batch_x = batch[0]
        batch_y = batch[1]
        batch_x = batch_x.to(device)
        #batch_x.requires_grad = True
        batch_y = batch_y.to(device)

        for idm, model in enumerate(models):
            pred_y = model.forward(batch_x)
            pred_y = softmax(pred_y)
            
            loss_f = partial(loss, model=model, batch_y=batch_y)

            # Non-adversarial test set accuracy
            conf_y, max_indices = torch.max(pred_y,1)
            n = max_indices.size(0)
            results[idm, 0] += (max_indices == batch_y).sum(dtype=torch.float32)/n
            
            # Adversarial test set accuracy for different epsilon values
            maxx = max_eps if isinstance(max_eps, int) else 1000 * max_eps
            s = 1 if isinstance(max_eps, int) else step * 1000
            for ide, epss in enumerate(np.arange(s, maxx+s, s)):
                
                # Convert epsilon to the right scale
                eps = epss / 1000 if not isinstance(max_eps, int) else epss
                
                # Generate adversarial examples from batch_x for the corresponding epsilon
                
                if foolbox_use:
                    perturbed_batch_x = attack(batch_x, batch_y, model=model, epsilon=eps)
                else:
                    adv_attack = partial(attack, loss_f=loss_f, eps=eps)
                    perturbed_batch_x = adv_attack(batch_x)

                # Get the predictions for the adversarial examples
                pred_y_adv = model.forward(perturbed_batch_x)

                #adversarial test accuracy
                conf_y_adv, max_indices_adv = torch.max(pred_y_adv,1)
                n = max_indices_adv.size(0)
                results[idm, ide+1] += (max_indices_adv == batch_y).sum(dtype=torch.float32)/n

    results /= len(test_loader)

    x_axis = np.arange(0, max_eps+step, step) if isinstance(max_eps, float) else np.arange(0, max_eps+1, 1)

    for i in range(len(models)):
        plt.plot(x_axis, results[i], label=model_names[i])
        plt.scatter(x_axis, results[i])

    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

from autoattack import utils_tf2

def adversarial_attacks_eps_plot_test(models, model_names, test_loader, attacks, attack_names, loss, max_eps, step, foolbox_uses):
    """
    Plots and saves the accuracy of models under multiple adversarial attacks for different epsilon values.

    Args:
        models (list): List of PyTorch models.
        model_names (list): List of names corresponding to the models.
        test_loader (torch.utils.data.DataLoader): Data loader for the test dataset.
        attacks (list): List of adversarial attack functions.
        attack_names (list): List of attack names.
        loss (function): Loss function used for the attack.
        max_eps (float or int): Maximum epsilon value for the attack.
        step (float, optional): Step size for epsilon values. Defaults to 0.025.
        foolbox_use (bool, optional): Whether to use Foolbox for attacks.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = max_eps + 1 if isinstance(max_eps, int) else int(round(max_eps / step)) + 1
    x_axis = np.linspace(0, max_eps, dim) if isinstance(max_eps, float) else np.arange(0, max_eps + 1, 1)

    for attack, attack_name, foolbox_use in zip(attacks, attack_names, foolbox_uses):
        results = np.zeros((len(models), dim))
        printed = False ####################

        for batch in test_loader:
            batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if printed:
                break

            for idm, model in enumerate(models):
                
                #if isinstance(model, ProbPSENN_VAE) or isinstance(model, ProbPSENN):
                    #model = model.to(device)
                    #adapter = ProbPSENNAdapter(model)
                    #print("Adapted")
                    #batch_x_attack = batch_x.permute(0, 3, 1, 2)
                    #batch_x = batch_x.detach().cpu().numpy().transpose(0, 2, 3, 1)
                
                #if isinstance(model, ProtoVAE):
                #    pred_y, _ = model.pred_class(batch_x)
                #if isinstance(model, ProbPSENN) or isinstance(model, ProbPSENN_VAE):
                #    pred_y = model.get_pred_distrib(batch_x.cpu().numpy(), n_samples=30, reduce_samples=True)# What is used in the eval.py in github
                #    pred_y = torch.from_numpy(pred_y).to(device)
                #else:
                
                pred_y = model.forward(batch_x)
                #pred_y = adapter.forward(batch_x)
                #pred_y = model.get_pred_distrib(batch_x, n_samples=30, reduce_samples=True)
                #pred_y = torch.from_numpy(pred_y.probs.numpy()).to(device)
                pred_y_s = torch.softmax(pred_y, dim=1)

                if not foolbox_use:
                    loss_f = partial(loss, model=model, batch_y=batch_y)

                print(f"[AutoAttack_adv] wrapping model: should be adapter or nn.Module, got {type(model)}")

                # Non-adversarial test set accuracy
                _, max_indices = torch.max(pred_y_s, 1)
                n = max_indices.size(0)
                results[idm, 0] += (max_indices == batch_y).float().sum().item() / n
                
                #Printiing###################
                print(f"pred_y shape {pred_y.shape}")
                print(f"pred_y logits {pred_y[0]}")
                print(f"pred_y softmax {pred_y_s[0]}")
                print(f"pred_y max_indices {max_indices}")
                print(f"pred_y accuracy {results[idm, 0]}")
                perturbed_batch_x = attack(batch_x, batch_y, model=model, epsilon=0.1)
                #perturbed_batch_x = attack(batch_x, batch_y, model=adapter, epsilon=0.1)
                #perturbed_batch_x = (perturbed_batch_x - 0.5) / 0.5
                pred_y_adv_01 = model.forward(perturbed_batch_x)
                #pred_y_adv_01 = adapter.forward(perturbed_batch_x)
                #pred_y_adv_01 = model.get_pred_distrib(perturbed_batch_x, n_samples=30, reduce_samples=True)
                #pred_y_adv_01 = torch.from_numpy(pred_y_adv_01.probs.numpy()).to(device)
                _, max_indices_adv_01 = torch.max(pred_y_adv_01, 1)
                pred_y_s_adv_01= torch.softmax(pred_y_adv_01, dim=1)
                _, max_indices_adv_01_s = torch.max(pred_y_s_adv_01, 1)

                print(f"pred_y_adv_01 shape {pred_y_adv_01.shape}")
                print(f"pred_y_adv_01 logits {pred_y_adv_01[0]}")
                print(f"pred_y_adv_01 softmax {pred_y_s_adv_01[0]}")
                print(f"pred_y_adv_01 max_indices {max_indices_adv_01}")
                print(f"pred_y_adv_01_s max indices {max_indices_adv_01_s}")

                perturbed_batch_x = attack(batch_x, batch_y, model=model, epsilon=0.6)
                #perturbed_batch_x = attack(batch_x, batch_y, model=adapter, epsilon=0.6)
                #perturbed_batch_x = (perturbed_batch_x - 0.5) / 0.5
                pred_y_adv_06 = model.forward(perturbed_batch_x)
                #pred_y_adv_06 = adapter.forward(perturbed_batch_x)
                #pred_y_adv_06 = model.get_pred_distrib(batch_x, n_samples=30, reduce_samples=True)
                #pred_y_adv_06 = torch.from_numpy(pred_y_adv_06.probs.numpy()).to(device)
                _, max_indices_adv_06 = torch.max(pred_y_adv_06, 1)
                pred_y_s_adv_06= torch.softmax(pred_y_adv_06, dim=1)
                _, max_indices_adv_06_s = torch.max(pred_y_s_adv_06, 1)

                print(f"pred_y_adv_06 shape {pred_y_adv_06.shape}")
                print(f"pred_y_adv_06 logits {pred_y_adv_06[0]}")
                print(f"pred_y_adv_06 softmax {pred_y_s_adv_06[0]}")
                print(f"pred_y_adv_06 max_indices {max_indices_adv_06}")
                print(f"pred_y_adv_06_s max indices {max_indices_adv_06_s}")

                printed = True
                break

                # Adversarial test set accuracy for different epsilon values
                maxx = int(round(max_eps / step)) if isinstance(max_eps, float) else max_eps
                s = int(round(step / step)) if isinstance(max_eps, float) else 1

                for ide, epss in enumerate(np.arange(s, maxx + s, s)):
                    eps = epss * step if isinstance(max_eps, float) else epss

                    # Generate adversarial examples
                    if foolbox_use:
                        perturbed_batch_x = attack(batch_x, batch_y, model=model, epsilon=eps)
                        if isinstance(model, ProtoVAE) or isinstance(model, ProbPSENN) or isinstance(model, ProbPSENN_VAE):
                            perturbed_batch_x = (perturbed_batch_x - 0.5) / 0.5    
                    else:
                        adv_attack = partial(attack, loss_f=loss_f, eps=eps)
                        perturbed_batch_x = adv_attack(batch_x)

                    # Get predictions for adversarial examples
                    
                    #if isinstance(model, ProtoVAE):
                    #    pred_y_adv, _ = model.pred_class(perturbed_batch_x)
                    #if isinstance(model, ProbPSENN_VAE) or isinstance(model, ProbPSENN):
                    #    pred_y_adv = model.get_pred_distrib(perturbed_batch_x.cpu().numpy(), n_samples=30, reduce_samples=True)# What is used in the eval.py in github
                    #    pred_y_adv = torch.from_numpy(pred_y_adv).to(device)
                    #else:
                    pred_y_adv = model.forward(perturbed_batch_x)

                    # Adversarial test accuracy
                    _, max_indices_adv = torch.max(pred_y_adv, 1)
                    results[idm, ide + 1] += (max_indices_adv == batch_y).float().sum().item() / n
        
        results /= len(test_loader)

        for idm, model_name in enumerate(model_names):
            model_result = results[idm]
            
            ############
            break

            # Save results to CSV
            csv_path = f"results/Accuracy/{model_name}_{attack_name}_maxeps({max_eps})_step({step})_results.csv"
            df = pd.DataFrame({model_name: model_result}, index=x_axis)
            df.index.name = "Epsilon"
            df.to_csv(csv_path)

            # Plot results
            plt.figure(figsize=(8, 6))
            plt.plot(x_axis, model_result, label=model_name)
            plt.scatter(x_axis, model_result)

            plt.xlabel('Epsilon')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.title(f"Adversarial Attack Accuracy vs. Epsilon\nModel: {model_name} | Attack: {attack_name}")

            # Save plot
            jpg_path = f"results/Plot/{model_name}_{attack_name}_maxeps({max_eps})_step({step})_plot.jpg"
            plt.savefig(jpg_path, dpi=300)
            plt.close()

            print(f"Accuracy results saved to {csv_path}")
            print(f"Plot saved to {jpg_path}")

def test_model(model, test_loader):
    """
    Test the given model on the test dataset and calculate the accuracy.

    Args:
        model (torch.nn.Module): The model to be tested.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device
    test_ac = 0.0

    for i, batch in enumerate(test_loader):
        batch_x = batch[0]
        batch_y = batch[1]
        batch_x = batch_x.to(device)
        batch_x.requires_grad = True
        batch_y = batch_y.to(device)


        pred_y = model.forward(batch_x)

        # test accuracy
        conf_y, max_indices = torch.max(pred_y,1)
        n = max_indices.size(0)
        test_ac += (max_indices == batch_y).sum(dtype=torch.float32)/n

    test_ac /= len(test_loader)

    print("test set accuracy: {:.4f}".format(test_ac))


def test_adversarial(model, test_loader, loss, attack, n_examples, examples_type):
    """
    Test the model's performance on adversarial examples.

    Args:
        model (torch.nn.Module): The trained model.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        loss (callable): The loss function used for training the model.
        attack (callable): The attack function used to generate adversarial examples.
        n_examples (int): The maximum number of adversarial examples to show.
        examples_type (str): The type of adversarial examples to show. Can be one of the following:
            - "cdp": Correctly classified and the closest prototype is different.
            - "csp": Correctly classified and the closest prototype is the same.
            - "isp": Incorrectly classified and the closest prototype is the same.
            - "idp": Incorrectly classified and the closest prototype is different.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device
    test_ac = 0
    test_ac_adv = 0
    n_test_batch = len(test_loader)

    corr_dist_proto = 0
    corr_same_proto = 0
    incorr_same_proto = 0
    incorr_dist_proto = 0


    prototype_distances = model.prototype_layer.prototype_distances
    prototype_imgs = model.decoder(prototype_distances.reshape((-1,10,2,2))).detach().cpu()
    total_examples = 0

    for i, batch in enumerate(test_loader):
        batch_x = batch[0]
        batch_y = batch[1]
        batch_x = batch_x.to(device)
        batch_x.requires_grad = True
        batch_y = batch_y.to(device)

        pred_y = model.forward(batch_x)
        pred_y = softmax(pred_y)

        loss_f = partial(loss, batch_y=batch_y)
        adv_attack = partial(attack, loss_f=loss_f)

        perturbed_batch_x = adv_attack(batch_x)

        pred_y_adv = model.forward(perturbed_batch_x)
        pred_y_adv = softmax(pred_y_adv)

        # test accuracy
        conf_y, max_indices = torch.max(pred_y,1)
        n = max_indices.size(0)
        test_ac += (max_indices == batch_y).sum(dtype=torch.float32)/n

        #adversarial test accuracy
        conf_y_adv, max_indices_adv = torch.max(pred_y_adv,1)
        n = max_indices_adv.size(0)
        test_ac_adv += (max_indices_adv == batch_y).sum(dtype=torch.float32)/n

        total_examples += show_adversarial_examples(model, batch_x, perturbed_batch_x, batch_y, max_indices, max_indices_adv, conf_y, conf_y_adv, n_examples-total_examples, examples_type)

        # Distances for normal and adversarial examples 
        distances = model.prototype_layer(model.encoder(batch_x).view(batch_x.size(0), -1))  
        distances_adv = model.prototype_layer(model.encoder(perturbed_batch_x).view(perturbed_batch_x.size(0), -1)) 

        for idx in range(batch_x.size(0)): 

            dists = distances[idx].detach().cpu().numpy()  
            dists_adv = distances_adv[idx].detach().cpu().numpy()  

            # Sort prototypes by distances
            sorted_prototypes = sorted(zip(prototype_imgs, dists), key=lambda x: x[1])
            sorted_prototype_imgs, sorted_dists = zip(*sorted_prototypes)

            # Sort prototypes by distances for adversarial examples
            sorted_prototypes_adv = sorted(zip(prototype_imgs, dists_adv), key=lambda x: x[1])
            sorted_prototype_imgs_adv, sorted_dists_adv = zip(*sorted_prototypes_adv)

            # If correctly classified and the closest prototype is different
            if batch_y[idx] == max_indices_adv[idx] and not torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0]):
                corr_dist_proto += 1/n

            # If incorrectly classified and the closest prototype is the same
            elif batch_y[idx] != max_indices_adv[idx] and torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0]):
                incorr_same_proto += 1/n

            # If correctly classified and the closest prototype is the same
            elif batch_y[idx] == max_indices_adv[idx] and torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0]):
                corr_same_proto += 1/n

            #If incorrectly classified and the closest prototype is different
            elif batch_y[idx] != max_indices_adv[idx] and not torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0]):
                incorr_dist_proto += 1/n

    test_ac /= n_test_batch
    test_ac_adv /= n_test_batch
    corr_dist_proto /= n_test_batch 
    corr_same_proto /= n_test_batch
    incorr_same_proto /= n_test_batch
    incorr_dist_proto /= n_test_batch

    print("test set:")
    print("\taccuracy: {:.4f}".format(test_ac))

    print("adversarial test set:")
    print("\taccuracy: {:.4f}".format(test_ac_adv))
    print("\tCorrectly classified and the closest prototype is different: {:.4f}".format(corr_dist_proto))
    print("\tCorrectly classified and the closest prototype is the same: {:.4f}".format(corr_same_proto))
    print("\tIncorrectly classified and the closest prototype is the same: {:.4f}".format(incorr_same_proto))
    print("\tIncorrectly classified and the closest prototype is different: {:.4f}".format(incorr_dist_proto))
    
def get_encoded_test_data_and_fit_pca(test_loader, model, device):
    """
    Encodes test data using the model's encoder and fits a PCA transformation on the encoded data.

    Args:
    - test_loader (DataLoader): DataLoader for the test dataset.
    - model (nn.Module): The model containing the encoder.
    - device (torch.device): The device to run the model on (CPU or GPU).

    Returns:
    - reduced_test_data (np.ndarray): 2D PCA projection of the encoded test data.
    - labels (np.ndarray): Labels of the test data.
    - pca (PCA): Fitted PCA object.
    """
    encoded_data = []
    labels = []
    
    # Encode the test data
    for batch in test_loader:
        batch_x, batch_y = batch
        batch_x = batch_x.to(device)
        encoded_batch = model.encoder(batch_x).detach().cpu().numpy().reshape(batch_x.size(0), -1)
        encoded_data.append(encoded_batch)
        labels.append(batch_y.numpy())
    
    encoded_data = np.concatenate(encoded_data, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # Fit PCA on the encoded data
    pca = PCA(n_components=2)
    reduced_test_data = pca.fit_transform(encoded_data)
    
    return reduced_test_data, labels, pca

def get_prototype_projection(model_path, device, pca):
    """
    Projects the prototypes of the model into a 2D PCA space.

    Args:
    - model_path (str): Path to the model file.
    - device (torch.device): The device to run the model on (CPU or GPU).
    - pca (PCA): Fitted PCA object.

    Returns:
    - reduced_prototypes (np.ndarray): 2D PCA projection of the prototypes.
    - prototype_imgs (torch.Tensor): Decoded prototype images.
    """
    model = torch.load(model_path)
    model.to(device)
    model.prototype_layer.prototype_distances = model.prototype_layer.prototype_distances.to(device)
    model.eval()
    
    prototype_distances = model.prototype_layer.prototype_distances
    prototype_imgs = model.decoder(prototype_distances.reshape((-1, 10, 2, 2))).detach().cpu()
    
    # Project prototypes using the fitted PCA
    reduced_prototypes = pca.transform(prototype_distances.detach().cpu().numpy().reshape(-1, 40))
    
    return reduced_prototypes, prototype_imgs

def plot_prototype_projection_with_data(reduced_prototypes, prototype_imgs, reduced_test_data, test_labels, title, xlim, ylim, save_path):
    """
    Plots the 2D PCA projection of prototypes and test data.

    Args:
    - reduced_prototypes (np.ndarray): 2D PCA projection of the prototypes.
    - prototype_imgs (torch.Tensor): Decoded prototype images.
    - reduced_test_data (np.ndarray): 2D PCA projection of the test data.
    - test_labels (np.ndarray): Labels of the test data.
    - title (str): Title of the plot.
    - xlim (tuple): x-axis limits for the plot.
    - ylim (tuple): y-axis limits for the plot.
    - save_path (str): Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.set_title(title)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    
    image_size = 0.2  # Adjust this value to make images smaller
    
    # Plot the encoded test data with different colors for each class using 'tab20' colormap
    scatter = ax.scatter(reduced_test_data[:, 0], reduced_test_data[:, 1], c=test_labels, cmap='tab20', alpha=0.5)
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)
    
    # Overlay prototype images
    for i, (x, y) in enumerate(reduced_prototypes):
        img = prototype_imgs[i].squeeze().numpy()  # Remove single-dimensional entries
        ax.imshow(img, cmap='gray', extent=(x - image_size / 2, x + image_size / 2, y - image_size / 2, y + image_size / 2), aspect='auto', zorder=10)
        ax.scatter(x, y, c='red', s=1, zorder=11)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
        
def plot_adversarial_examples(model, model_name, test_loader, loss, attack, attack_name, num_examples, eps=0.3, foolbox_use=True):
    """
    Visualize real images with their respective after-attack images.

    Args:
        model (torch.nn.Module): The trained model.
        model_name (str): Name of the model (for title/display purposes).
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        loss (callable): Loss function.
        attack (callable): The attack function used to generate adversarial examples.
        num_examples (int): Number of examples to display.
        eps (float): Perturbation budget for the attack.
        foolbox_use (bool): Whether to use Foolbox for attacks.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    plotted = 0  # Counter for total images plotted
    images = []
    perturbed_images = []
    real_labels = []
    pred_labels = []
    pred_adv_labels = []

    for i, batch in enumerate(test_loader):
        if plotted >= num_examples:
            break  # Stop once we reach the desired number

        batch_x, batch_y = batch[0].to(device), batch[1].to(device)

        # Generate adversarial examples
        loss_f = partial(loss, batch_y=batch_y)
        if foolbox_use:
            perturbed_batch_x = attack(batch_x, batch_y, model=model, epsilon=eps)
        else:
            adv_attack = partial(attack, loss_f=loss_f, eps=eps)
            perturbed_batch_x = adv_attack(batch_x)

        # Get predictions
        pred_y = torch.nn.functional.softmax(model(batch_x), dim=1)
        pred_y_adv = torch.nn.functional.softmax(model(perturbed_batch_x), dim=1)

        # Collect examples
        for j in range(len(batch_x)):
            if plotted >= num_examples:
                break  # Stop collecting once we reach the limit

            images.append(batch_x[j].cpu().detach().numpy())
            perturbed_images.append(perturbed_batch_x[j].cpu().detach().numpy())
            real_labels.append(batch_y[j].cpu().item())
            pred_labels.append(torch.argmax(pred_y[j]).cpu().item())
            pred_adv_labels.append(torch.argmax(pred_y_adv[j]).cpu().item())

            plotted += 1  # Increment counter

    # Plot collected images
    fig, axes = plt.subplots(num_examples, 2, figsize=(8, num_examples * 3))
    fig.suptitle(f"Model: {model_name} | Attack: {attack_name} | Epsilon: {eps}", fontsize=14, fontweight="bold")  
        
    for i in range(num_examples):
        # Original Image
        img = np.transpose(images[i], (1, 2, 0))  # Convert CHW -> HWC
        plt.subplot(num_examples, 2, 2 * i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Orig: {real_labels[i]}\nPred: {pred_labels[i]}")

        # Adversarial Image
        img_adv = np.transpose(perturbed_images[i], (1, 2, 0))
        plt.subplot(num_examples, 2, 2 * i + 2)
        plt.imshow(img_adv)
        plt.axis("off")
        plt.title(f"Adv Pred: {pred_adv_labels[i]}")

    plt.tight_layout()
    plt.show()
    
