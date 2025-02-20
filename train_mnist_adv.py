import torch
print('pytorch version ', torch.__version__)

from data_preprocessing import batch_elastic_transform
from data_loader import get_train_val_loader, get_test_loader
from loss_functions import generalLoss, ClstSepLoss, CELoss
from adversarial_attacks import FSGM_attack, PGDLInf_attack, PGDL2_attack
from modules import *
from autoencoder_helpers import *


import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from functools import partial

import argparse

if __name__ == '__main__':
    print("__main__")
    parser = argparse.ArgumentParser()
    parser.add_argument("-mt",  "--mt",  dest="modeltype",  type=str, default="standard",  help="model type to use")
    parser.add_argument("-tl",  "--tl",  dest="trainloss",  type=str, default="default",  help="loss function to train the model")
    parser.add_argument("-advatt",  "--advatt",  dest="adversarialattack",  type=str, default="pdglinf",  help="adversarial attack to use")
    parser.add_argument("-advl",  "--advl",  dest="adversarialloss",  type=str, default="ce",  help="loss function to generate adv examples")
    parser.add_argument("-eps",  "--eps",  dest="eps",  type=float, default=0.3,  help="epsilon value for adversarial attack")
    parser.add_argument("-iters",  "--iters",  dest="iters",  type=int, default=20,  help="number of iterations for adversarial attack")
    parser.add_argument("-alpha",  "--alpha",  dest="alpha",  type=float, default=0.02,  help="alpha value for adversarial attack")
    parser.add_argument("-rs", "--rs",  dest="randstart",  type=str, default="True",  help="random start for adversarial attack")
                        
    parser.add_argument("-e",  "--e",  dest="epochs",  type=int, default=1500,  help="training epochs")
    parser.add_argument("-lr",  "--lr",  dest="lr",  type=float, default=0.002,  help="learning rate")
    parser.add_argument("-bs",  "--bs",  dest="batchsize",  type=int, default=250,  help="batch size")
    
    parser.add_argument("-freezeW",  "--freezeW",  dest="freezeW",  type=str, default="False",  help="freeze weights layer fc")
    parser.add_argument("-fillW",  "--fillW",  dest="fillW",  type=float, default=0.5,  help="fill weights layer fc value")
    
    parser.add_argument("-lc",  "--lc",  dest="lambdac",  type=float, default=20,  help="lambda class hyperparameter")
    parser.add_argument("-lae",  "--lae",  dest="lambdae",  type=float, default=1,  help="lambda encoder hyperparameter")
    parser.add_argument("-l1",  "--l1",  dest="lambda1",  type=float, default=1,  help="lambda 1 hyperparameter")
    parser.add_argument("-l2",  "--l2",  dest="lambda2",  type=float, default=1,  help="lambda 2 hyperparameter")
    parser.add_argument("-lcl",  "--lcl",  dest="lambdaclus",  type=float, default=0.8,  help="lambda clustering hyperparameter")
    parser.add_argument("-ls",  "--ls",  dest="lambdasep",  type=float, default=0.2,  help="lambda separation hyperparameter")
    parser.add_argument("-div", "--div",  dest="div",  type=float, default=1.0,  help="divisor for cls and sep hyperparameters")
    
    parser.add_argument("-np",  "--np",  dest="nprototypes",  type=int, default=15,  help="number of prototypes")
    parser.add_argument("-nl",  "--nl",  dest="nlayers",  type=int, default=4,  help="number of layers encoder/decoder")
    parser.add_argument("-nm",  "--nm",  dest="nmaps",  type=int, default=32,  help="number of maps encoder/decoder")
    parser.add_argument("-npc",  "--npc",  dest="nprototypesclass",  type=int, default=3,  help="number of prototypes by class")
    
    parser.add_argument("-s",  "--s",  dest="seed",  type=int, default=1,  help="random seed for reproducibility")
    
    args = parser.parse_args()


#In case model type in balanced, make sure no adjust the number of prototypes according to the number of prototypes per class
if args.modeltype == "balanced":
    n_prototypes = 10 * args.nprototypesclass
elif args.modeltype == "standard":
    n_prototypes = args.nprototypes
        
# train the model for MNIST handwritten digit dataset
# the directory to save the model
name = f"mnist_cae_adversarial_{args.modeltype}_{args.trainloss}_{args.adversarialattack}_{args.adversarialloss}_{args.iters}_{args.eps}_{args.alpha}_{args.randstart}_{args.epochs}_{args.lr}_{args.batchsize}_{args.freezeW}_{args.fillW}_{args.lambdac}_{args.lambdae}_{args.lambda1}_{args.lambda2}_{args.lambdaclus}_{args.lambdasep}_{args.div}_{n_prototypes}_{args.nlayers}_{args.nmaps}_{args.seed}"
model_folder = os.path.join(os.getcwd(), "saved_model", "mnist_model", name)
makedirs(model_folder)
img_folder = os.path.join(model_folder, "img")
makedirs(img_folder)

#Save the configuration clearly

model_filename = "mnist_cae_adv"
optimizer_filename = "optimizer_cae_adv"

# console_log is the handle to a text file that records the console output
log_folder=os.path.join(model_folder, "log")
makedirs(log_folder)
console_log = open(os.path.join(log_folder, "console_log.txt"), "w+")

# GPU/CUDA setup
if torch.cuda.is_available():
    print("CUDA enabled!")
    device = torch.device('cuda:0')  # You can specify the index of the CUDA device you want to use
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Specify the index of the GPU(s) you want to use
else:
    print("CUDA not available. Using CPU.")
    device = torch.device('cpu')

# training parameters
learning_rate = args.lr
training_epochs = args.epochs
batch_size = args.batchsize           # the size of a minibatch

validation_display_step = 100      # how many epochs we do evaluate on the test set once
save_step = 50               # how frequently do we save the model to disk

# elastic deformation parameters
sigma = 4
alpha =20

# lambda's are the ratios between the four error terms
lambda_class = args.lambdac
lambda_ae = args.lambdae
# 1 and 2 here corresponds to the notation we used in the paper
lambda_1 = args.lambda1 
lambda_2 = args.lambda2

#if clstsep loss is used
lambda_clus = args.lambdaclus
lambda_sep = args.lambdasep

# input data parameters
input_height = 28         # MNIST data input shape
input_width = input_height
n_input_channel = 1       # the number of color channels; for MNIST is 1.
input_size = input_height * input_width * n_input_channel   # the number of pixels in one input image
input_shape = (1, n_input_channel, input_height, input_height) # input shape to pass in the model
n_classes = 10

# Network Parameters
n_layers = args.nlayers                 # the number of layers in the encoder/decoder
n_maps = args.nmaps                    # the number of maps in the encoder/decoder

# For Balanced prototypes
prototypes_by_class = args.nprototypesclass # the number of prototypes per class
fill_with = args.fillW # the value to fill the fc layer with

# data load and split parameters
random_seed = args.seed
data_folder = 'data'

# download MNIST data
train_loader, val_loader = get_train_val_loader(data_folder, batch_size, random_seed, augment=False, val_size=0.2,
                           shuffle=True, show_sample=False, num_workers=0, pin_memory=True)

#test_loader = get_test_loader(data_folder, batch_size, shuffle=True, num_workers=0, pin_memory=True)

# construct the model
if args.modeltype == "standard":
    model = CAEModel(input_shape=input_shape, n_maps=n_maps, n_prototypes=n_prototypes, 
                    n_layers=n_layers, n_classes=n_classes).to(device)
if args.modeltype == "balanced":
    model = CAEModel_Balanced(input_shape=input_shape, n_maps=n_maps, prototypes_by_class=prototypes_by_class, 
                    n_layers=n_layers, n_classes=n_classes, fill_with=fill_with).to(device)

#training loss
if args.trainloss == "default":
    training_loss = generalLoss
elif args.trainloss == "clstsep":
    training_loss = ClstSepLoss

#adversarial attack
eps = args.eps
iters = args.iters
alpha = args.alpha
random_start = True if args.randstart == "True" else False

if args.adversarialattack == "fsgm":
    adversarial_attack = partial(FSGM_attack, eps=eps, random_start=random_start) 
elif args.adversarialattack == "pdglinf":
    adversarial_attack = partial(PGDLInf_attack, eps=eps, alpha=alpha, iters=iters, random_start=random_start)
elif args.adversarialattack == "pdgl2":
    adversarial_attack = partial(PGDL2_attack, eps=eps, alpha=alpha, iters=iters, random_start=random_start)
    
#adversarial loss
if args.adversarialloss == "ce":
    adversarial_loss = CELoss
    
#Freeze the identity w matrix
if args.freezeW == "True":
    for param in model.fc.parameters():
        param.requires_grad = False

# optimizer setup
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start_time= time.time()

config_file_path = os.path.join(model_folder, "config.txt")
with open(config_file_path, "w") as config_file:
    config_file.write("Model Configuration:\n")
    config_file.write(f"Model Type: {args.modeltype}\n")
    config_file.write(f"Training Loss Function: {args.trainloss}\n")
    config_file.write(f"Adversarial Attack: {args.adversarialattack}\n")
    config_file.write(f"Adversarial Loss Function: {args.adversarialloss}\n")
    config_file.write(f"Adversarial Iterations: {args.iters}\n")
    config_file.write(f"Adversarial Epsilon: {args.eps}\n")
    config_file.write(f"Adversarial Alpha: {args.alpha}\n")
    config_file.write(f"Epochs: {args.epochs}\n")
    config_file.write(f"Learning Rate: {args.lr}\n")
    config_file.write(f"Batch Size: {args.batchsize}\n")
    config_file.write(f"Freeze Weights Layer fc: {args.freezeW}\n")
    config_file.write(f"Weights Layer fc Value: {args.fillW}\n")
    config_file.write(f"Lambda Class Hyperparameter: {args.lambdac}\n")
    config_file.write(f"Lambda Encoder Hyperparameter: {args.lambdae}\n")
    config_file.write(f"Lambda 1 Hyperparameter: {args.lambda1}\n")
    config_file.write(f"Lambda 2 Hyperparameter: {args.lambda2}\n")
    config_file.write(f"Lambda Clustering Hyperparameter: {args.lambdaclus}\n")
    config_file.write(f"Lambda Separation Hyperparameter: {args.lambdasep}\n")
    config_file.write(f"Divisor for Clustering and Separation Hyperparameters: {args.div}\n")
    config_file.write(f"Number of Prototypes by Class: {args.nprototypesclass}\n")
    config_file.write(f"Number of Prototypes: {n_prototypes}\n")
    config_file.write(f"Number of Layers Encoder/Decoder: {args.nlayers}\n")
    config_file.write(f"Number of Maps Encoder/Decoder: {args.nmaps}\n")
    config_file.write(f"Random Seed: {args.seed}\n")
    
# train the model
for epoch in range(0, training_epochs):
    print_and_write("#"*80, console_log)
    print_and_write("Epoch: %04d" % (epoch+1), console_log)
    n_train_batch = len(train_loader)
    n_val_batch = len(val_loader)
    start = time.time()

    train_ce, train_ae, train_e1, train_e2, train_te, train_ac, train_clst_l, train_sep_l = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    train_ac_adv = 0.0
    
    model.train()
    with tqdm(total=len(train_loader), file=sys.stdout) as pbar:
        
        for i, batch in enumerate(train_loader):
            batch_x = batch[0]
            batch_y = batch[1]
            batch_y = batch_y.to(device)
            # store original batch shape to put it back into this shape after transformation
            batch_shape = batch_x.shape
            
            # generate adversarial batch
            optimizer.zero_grad()
            loss_f = partial(adversarial_loss, model=model, batch_y=batch_y)
            batch_x_adv = adversarial_attack(loss_f=loss_f, batch_x=batch_x)
            batch_x_adv = batch_x_adv.to('cpu')
            
            for idx, b_x in enumerate([batch_x_adv, batch_x]):
                
                elastic_batch_x = b_x.to(device)
                
                optimizer.zero_grad()        

                pred_y = model.forward(elastic_batch_x)
                
                #If example is adversarial, we use the alternative autoencoder loss
                if idx == 0:
                    if args.trainloss == "default":
                        train_te, train_ce, train_e1, train_e2, train_ae = training_loss(model, elastic_batch_x, batch_y, pred_y, lambda_class, lambda_1 * args.div, lambda_2 * args.div, lambda_ae, batch_x.to(device))
                    elif args.trainloss == "clstsep":
                        train_te, train_ce, train_e1, train_clst_l, train_sep_l, train_ae = training_loss(model, elastic_batch_x, batch_y, pred_y, lambda_class, lambda_1 * args.div, lambda_clus * args.div, lambda_sep * args.div, lambda_ae, batch_x.to(device))    
                else:
                    if args.trainloss == "default":
                        train_te, train_ce, train_e1, train_e2, train_ae = training_loss(model, elastic_batch_x, batch_y, pred_y, lambda_class, lambda_1, lambda_2, lambda_ae)
                    elif args.trainloss == "clstsep":
                        train_te, train_ce, train_e1, train_clst_l, train_sep_l, train_ae = training_loss(model, elastic_batch_x, batch_y, pred_y, lambda_class, lambda_1, lambda_clus, lambda_sep, lambda_ae)
                
                train_te.backward()
                    
                optimizer.step()
                
                if idx == 0:
                    # adversarial train accuracy
                    max_vals, max_indices = torch.max(pred_y,1)
                    n = max_indices.size(0)
                    train_ac_adv += (max_indices == batch_y).sum(dtype=torch.float32)/n
                        
                else:
                    # non-adversarial train accuracy
                    max_vals, max_indices = torch.max(pred_y,1)
                    n = max_indices.size(0)
                    train_ac += (max_indices == batch_y).sum(dtype=torch.float32)/n

            pbar.set_description('batch: %03d' % (1 + i))
            pbar.update(1)            
    
    train_ac /= n_train_batch
    train_ac_adv /= n_train_batch
    
    # after every epoch, check the error terms on the entire training set
    if args.trainloss == "default":
        print_and_write("training set errors:"+"\tclassification error: {:.6f}".format(train_ce)+
                        "\tautoencoder error: {:.6f}".format(train_ae)+
                        "\terror_1: {:.6f}".format(train_e1)+
                        "\terror_2: {:.6f}".format(train_e2)+
                        "\ttotal error: {:.6f}".format(train_te)+
                        "\taccuracy: {:.6f}".format(train_ac)+
                        "\tadv accuracy: {:.6f}".format(train_ac_adv), console_log)
        
    elif args.trainloss == "clstsep":
        print_and_write("training set errors:"+"\tclassification error: {:.6f}".format(train_ce)+
                        "\tautoencoder error: {:.6f}".format(train_ae)+
                        "\terror_1: {:.6f}".format(train_e1)+
                        "\tclst loss: {:.6f}".format(train_clst_l)+
                        "\tsep loss: {:.6f}".format(train_sep_l)+
                        "\ttotal error: {:.6f}".format(train_te)+
                        "\taccuracy: {:.6f}".format(train_ac)+
                        "\tadv accuracy: {:.6f}".format(train_ac_adv), console_log)
    print_and_write('training takes {0:.2f} seconds.'.format((time.time() - start)), console_log)
     
    # validation set error terms evaluation
    if (epoch+1) % validation_display_step == 0 or epoch == training_epochs - 1:
        
        val_ce, val_ae, val_e1, val_e2, val_te, val_ac, val_clst_l, val_sep_l = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        val_ac_adv = 0.0
        
        for i, batch in enumerate(val_loader):
            batch_x = batch[0]
            batch_y = batch[1]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # generate adversarial batch
            optimizer.zero_grad()
            loss_f = partial(adversarial_loss, model=model, batch_y=batch_y)
            batch_x_adv = adversarial_attack(loss_f=loss_f, batch_x=batch_x)
            batch_x_adv = batch_x_adv.to(device)
            
            model.eval()
            with torch.no_grad():
        
                for idx, b_x in enumerate([batch_x_adv, batch_x]):
                    
                    pred_y = model.forward(b_x)
                    
                    if idx == 0:
                        # validation adversarial accuracy
                        max_vals_adv, max_indices_adv = torch.max(pred_y,1)
                        n_adv = max_indices_adv.size(0)
                        val_ac_adv += (max_indices_adv == batch_y).sum(dtype=torch.float32)/n_adv
                              
                    else:
                        #loss standard
                        if args.trainloss == "default":
                            val_te, val_ce, val_e1, val_e2, val_ae = training_loss(model, b_x, batch_y, pred_y, lambda_class, lambda_1, lambda_2, lambda_ae)
                        elif args.trainloss == "clstsep":
                            val_te, val_ce, val_e1, val_clst_l, val_sep_l, val_ae = training_loss(model, b_x, batch_y, pred_y, lambda_class, lambda_1, lambda_clus, lambda_sep, lambda_ae)
                            
                        # validation accuracy
                        max_vals, max_indices = torch.max(pred_y,1)
                        n = max_indices.size(0)
                        val_ac += (max_indices == batch_y).sum(dtype=torch.float32)/n
                    
        val_ac /= n_val_batch
        val_ac_adv /= n_val_batch
        
        # after every epoch, check the error terms on the entire training set
        if args.trainloss == "default":
            print_and_write("validation set errors:"+"\t classification error: {:.6f}".format(val_ce)+
                            "\tautoencoder error: {:.6f}".format(val_ae)+
                            "\terror_1: {:.6f}".format(val_e1)+
                            "\terror_2: {:.6f}".format(val_e2)+
                            "\ttotal error: {:.6f}".format(val_te)+
                            "\taccuracy: {:.6f}".format(val_ac)+
                            "\tadv accuracy {:.6f}".format(val_ac_adv), console_log)
        elif args.trainloss == "clstsep":
            print_and_write("validation set errors:"+"\tclassification error: {:.6f}".format(val_ce)+
                            "\tautoencoder error: {:.6f}".format(val_ae)+
                            "\terror_1: {:.6f}".format(val_e1)+
                            "\tclst loss: {:.6f}".format(val_clst_l)+
                            "\tsep loss: {:.6f}".format(val_sep_l)+
                            "\ttotal error: {:.6f}".format(val_te)+
                            "\taccuracy: {:.6f}".format(val_ac)+
                            "\tadv accuracy {:.6f}".format(val_ac_adv), console_log)
    
    if (epoch+1) % save_step == 0 or epoch == training_epochs - 1:
        # save model states
        torch.save(model, os.path.join(model_folder, model_filename+'%05d.pth' % (epoch+1)))
        torch.save(optimizer, os.path.join(model_folder, optimizer_filename+'%05d.pth' % (epoch+1)))

        model.eval()
        with torch.no_grad():
            
            # save outputs as images
            # decode prototype vectors
            prototype_distances = model.prototype_layer.prototype_distances
            prototype_imgs = model.decoder(prototype_distances.reshape((-1,10,2,2))).detach().cpu()

            # visualize the prototype images
            n_cols = 5
            n_rows = n_prototypes // n_cols + 1 if n_prototypes % n_cols != 0 else n_prototypes // n_cols
            g, b = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
            for i in range(n_rows):
                for j in range(n_cols):
                    if i*n_cols + j < n_prototypes:
                        b[i][j].imshow(prototype_imgs[i*n_cols + j].reshape(input_height, input_width),
                                        cmap='gray',
                                        interpolation='none')
                        b[i][j].axis('off')
                        
            plt.savefig(os.path.join(img_folder, 'prototype_result-' + str(epoch+1) + '.png'),
                        transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close()

            # apply encoding and decoding over a small subset of the training set
            batch_x = []
            for batch in train_loader:
                batch_x = batch[0].to(device)
                break

            examples_to_show = 10
            
            encoded = model.encoder.forward(batch_x[:examples_to_show])
            decoded = model.decoder.forward(encoded)

            decoded = decoded.detach().cpu()
            imgs = batch_x.detach().cpu()

            # compare original images to their reconstructions
            f, a = plt.subplots(2, examples_to_show, figsize=(examples_to_show, 2))
            for i in range(examples_to_show):
                a[0][i].imshow(imgs[i].reshape(input_height, input_width),
                                cmap='gray',
                                interpolation='none')
                a[0][i].axis('off')
                a[1][i].imshow(decoded[i].reshape(input_height, input_width), 
                                cmap='gray',
                                interpolation='none')
                a[1][i].axis('off')
                
            plt.savefig(os.path.join(img_folder, 'decoding_result-' + str(epoch+1) + '.png'),
                        transparent=True,
                        bbox_inches='tight',
                        pad_inches=0)
            plt.close()

print_and_write('Total taken time {0:.2f} seconds.'.format((time.time() - start_time)), console_log)
print_and_write("Optimization Finished!", console_log)
console_log.close()
