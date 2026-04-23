import torch
import numpy as np
import random
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, random_split
# If Non-IID
# from util.data_split import dirichlet_split 

def get_mnist_dataloaders(num_clients=10, batch_size=32, iid=True, seed=42):
    """
    Download and split the MNIST dataset
    """
    # 1. Set the random seed to ensure consistent results each time it runs.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 2. Data Preprocessing 
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 3. Download the dataset
    data_root = './data'
    # download=True 
    train_dataset = datasets.MNIST(root=data_root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_root, train=False, download=True, transform=transform)

    # 4. Data partitioning
    if iid:
        # --- IID  (Simple average) ---
        total_len = len(train_dataset)
        len_per_client = total_len // num_clients
        lengths = [len_per_client] * num_clients
        # When dealing with remainders that cannot be divided evenly, distribute the excess to the preceding elements.
        for i in range(total_len % num_clients):
            lengths[i] += 1
            
        client_datasets = random_split(train_dataset, lengths)
    else:
        # --- Non-IID (Advanced) ---
        # If use data_split.py, it can be invoked here
        # labels = train_dataset.targets.tolist()
        # client_datasets = dirichlet_split(train_dataset, labels, num_clients, alpha=0.5, random_seed=seed)
        raise NotImplementedError("The Non-IID mode is currently inactive. Please first ensure the IID mode is fully operational.")

    # 5. Create DataLoader
    # pin_memory=True  It can accelerate the transfer of data from the CPU to the GPU.
    client_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True, pin_memory=True) 
        for ds in client_datasets
    ]
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, pin_memory=True)

    return client_loaders, test_loader



def get_cifar10_dataloaders(num_clients=10, batch_size=32, iid=True, seed=42):
    """
    Download and split the CIFAR-10 dataset
    """
    # 1. Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 2. Data preprocessing (CIFAR-10 specific)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616)
        )
    ])

    # 3. Download dataset
    data_root = './data'
    train_dataset = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    # 4. Data partitioning
    if iid:
        total_len = len(train_dataset)
        len_per_client = total_len // num_clients
        lengths = [len_per_client] * num_clients

        for i in range(total_len % num_clients):
            lengths[i] += 1

        client_datasets = random_split(train_dataset, lengths)
    else:
        raise NotImplementedError("Non-IID mode not implemented yet.")

    # 5. DataLoader
    client_loaders = [
        DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True
        )
        for ds in client_datasets
    ]

    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
        pin_memory=True
    )

    return client_loaders, test_loader


def get_femnist_dataloaders(num_clients, batch_size):
    print(" Downloading/Loading EMNIST (62 Classes)...")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,), (0.229,))
    ])
    
    train_dataset = torchvision.datasets.EMNIST(
        root='./data', split='byclass', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.EMNIST(
        root='./data', split='byclass', train=False, download=True, transform=transform
    )
    
    subset_indices = random.sample(range(len(test_dataset)), 2000)
    fast_test_dataset = Subset(test_dataset, subset_indices)
    
    test_loader = DataLoader(fast_test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    num_items = int(len(train_dataset) / num_clients)
    lengths = [num_items] * num_clients
    lengths[-1] += len(train_dataset) - sum(lengths)
    
    client_datasets = random_split(
        train_dataset, lengths, generator=torch.Generator().manual_seed(42)
    )
    
    client_loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True) for ds in client_datasets
    ]
    
    return client_loaders, test_loader