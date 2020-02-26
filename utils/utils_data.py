import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models

def prepare_data(dataset, batch_size):
    if dataset == 'SVHN':
        train_dataset = dsets.SVHN(root='~/data', split='train', transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.SVHN(root='~/data', split='test', transform=transforms.ToTensor(), download=True)
        num_classes = len(np.unique(train_dataset.labels))
    elif dataset == 'LSUN':
        train_dataset = dsets.LSUN(root='~/data', classes='train', transform=transforms.ToTensor())
        test_dataset = dsets.LSUN(root='~/data', classes='test', transform=transforms.ToTensor())
    elif dataset == 'CIFAR10':
        # normalizing values from https://github.com/kuangliu/pytorch-cifar/blob/master/main.py
        transform_tr_te = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_dataset = dsets.CIFAR10(root='~/data', train=True, transform=transform_tr_te, download=True)
        test_dataset = dsets.CIFAR10(root='~/data', train=False, transform=transform_tr_te, download=True)
        num_classes = len(train_dataset.classes)
    elif dataset == 'CIFAR100':
        # values from https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151
        transform_tr_te = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
        train_dataset = dsets.CIFAR100(root='~/data', train=True, transform=transform_tr_te, download=True)
        test_dataset = dsets.CIFAR100(root='~/data', train=False, transform=transform_tr_te, download=True)
        num_classes = len(train_dataset.classes)        
    else:
        train_dataset = dsets.__dict__[dataset](root='~/data', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.__dict__[dataset](root='~/data', train=False, transform=transforms.ToTensor())
        num_classes = len(train_dataset.classes)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, num_classes

def prepare_tvt_data(dataset, batch_size, training_proportion):
    if dataset == 'SVHN':
        trainvali_dataset = dsets.SVHN(root='~/data', split='train', transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.SVHN(root='~/data', split='test', transform=transforms.ToTensor(), download=True)
        num_classes = len(np.unique(trainvali_dataset.labels))
    elif dataset == 'LSUN':
        trainvali_dataset = dsets.LSUN(root='~/data', classes='train', transform=transforms.ToTensor())
        test_dataset = dsets.LSUN(root='~/data', classes='test', transform=transforms.ToTensor())
    else:
        trainvali_dataset = dsets.__dict__[dataset](root='~/data', train=True, transform=transforms.ToTensor(), download=True)
        test_dataset = dsets.__dict__[dataset](root='~/data', train=False, transform=transforms.ToTensor())
        num_classes = len(trainvali_dataset.classes)
    n_samples = len(trainvali_dataset)
    train_size = int(len(trainvali_dataset)*training_proportion)
    val_size = n_samples - train_size
    train_dataset, vali_dataset = torch.utils.data.random_split(trainvali_dataset, [train_size, val_size])
    if dataset == 'SVHN':
        drop_last = True
    else:
        drop_last = False
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=True)
    vali_loader = torch.utils.data.DataLoader(dataset=vali_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, drop_last=drop_last, shuffle=False)
    return train_loader, vali_loader, test_loader, num_classes
