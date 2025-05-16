import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

# function to load and return train and val multi-process iterator over the MNIST dataset.

def get_train_val_loader(data_dir, batch_size, random_seed, augment=False, val_size=0.2, 
                         shuffle=True, show_sample=False, num_workers=0, pin_memory=True):

    # load the dataset
    train_dataset = datasets.MNIST(root=data_dir, train=True, 
                download=True, transform=transforms.ToTensor())
    val_dataset = datasets.MNIST(root=data_dir, train=True, 
                download=True, transform=transforms.ToTensor())

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))

    if shuffle == True:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # create data iterator
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, 
                                               num_workers=num_workers, pin_memory=pin_memory)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, 
                                             num_workers=num_workers, pin_memory=pin_memory)
    return (train_loader, val_loader)

# function to load and return a multi-process test iterator over the MNIST dataset.
def get_test_loader(data_dir, 
                    batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    mode="basic"):

    """
    mode:
      - 'basic'      : ToTensor() only → [0,1], 28×28
      - 'norm'       : ToTensor() + Normalize(0.5,0.5) → [–1,1], 28×28
      - 'resize_norm': Resize(32×32) + ToTensor() + Normalize(0.5,0.5) → [–1,1], 32×32
    """

    if mode == 'basic':
        transform = transforms.Compose([
            transforms.ToTensor(),  # [0,1], 28×28
        ])
        print("Mode: basic (ToTensor only)")

    elif mode == 'norm':
        transform = transforms.Compose([
            transforms.ToTensor(),                         # [0,1], 28×28
            transforms.Normalize((0.5,), (0.5,)),          # → [–1,1]
        ])
        print("Mode: norm (ToTensor + Normalize)")

    elif mode == 'resize_norm':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),                   # 28×28 → 32×32
            transforms.ToTensor(),                         # [0,1]
            transforms.Normalize((0.5,), (0.5,)),          # → [–1,1]
        ])
        print("Mode: resize_norm (Resize + ToTensor + Normalize)")

    else:
        raise ValueError(f"Unknown mode '{mode}'. Choose from 'basic', 'norm', 'resize_norm'.")

    dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                                              num_workers=num_workers, pin_memory=pin_memory)
    return data_loader
