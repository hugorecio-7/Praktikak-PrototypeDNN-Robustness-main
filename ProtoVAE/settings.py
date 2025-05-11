# Hyperparameter and training settings for all the datasets


data_name = "mnist"
mode = "test"
model_file = "ProtoVAE/saved_models/mnist/model.pth"
expl = False


data_path = 'Data/'

coefs = {
        'crs_ent': 1,
        'recon': 1,
        'kl': 1,
        'ortho': 1,
    }


if (data_name == "mnist"):
    img_size = 28
    latent = 256
    num_prototypes = 50
    num_classes = 10
    batch_size = 128
    lr = 1e-3
    num_train_epochs = 10



elif (data_name == "fmnist"):
    img_size = 28
    latent = 256
    num_prototypes = 100
    num_classes = 10
    batch_size = 128
    lr = 1e-3
    num_train_epochs = 10



if (data_name == "cifar10"):
    img_size = 32
    latent = 512
    num_prototypes = 100
    num_classes = 10
    batch_size = 128
    lr = 1e-3
    num_train_epochs = 36



if (data_name == "svhn"):
    img_size = 32
    latent = 512
    num_prototypes = 50
    num_classes = 10
    batch_size = 64
    lr = 1e-3
    num_train_epochs = 36



if (data_name == "quickdraw"):
    img_size = 28
    latent = 512
    num_prototypes = 100
    num_classes = 10
    batch_size = 128
    lr = 1e-3
    num_train_epochs = 10
    data_path = data_path + 'quickdraw/'


