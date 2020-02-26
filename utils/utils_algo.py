import sys
import numpy as np
import math
import time
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dsets
import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F


class MyDataset(torch.utils.data.Dataset):

    def __init__(self, data, label, conf, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()])):
        self.transform = transform
        self.data = data
        self.conf = conf
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.transform(self.data)[0][idx]
        out_label = self.label[idx]
        out_conf = self.conf[idx]
        return out_data, out_label, out_conf
    

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def acc_check(loader, model, device):
    total_acc, total_loss, num_samples = 0, 0, 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.eval()
    with torch.no_grad():
        for (images, labels, *empty) in loader:
            images, labels =images.to(device), labels.to(device)
            outputs = model(images)
            sm_outputs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(sm_outputs.data, 1)
            total_acc += (predicted == labels).sum().item()
            total_loss += criterion(outputs, labels.long())
            num_samples += labels.size(0)
    return 100 * total_acc / num_samples, total_loss / num_samples

def get_grads(loader, model, device):
    num_samples = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    model.eval()

    # Initialize sum_grads with zeros of sizes compatible with parameters
    sum_grads = []
    model.zero_grad()
    for param in model.parameters():
        if not param.requires_grad:
            continue
        sum_grads.append(None)

    for (images, labels, *empty) in loader:
        images, labels =images.to(device), labels.to(device)
        outputs = model(images)
        # total_loss += criterion(outputs, labels.long())
        loss = criterion(outputs, labels.long())
        model.zero_grad()
        loss.backward()
        it_params = (p for p in model.parameters() if p.requires_grad)
        for i, param in enumerate(it_params):
            if sum_grads[i] is None:
                sum_grads[i] = param.grad
            else:
                sum_grads[i] += param.grad
        num_samples += labels.size(0)

    avg_grads = []
    for sum_grad in sum_grads:
        avg_grads.append(sum_grad.view(-1) / num_samples)

    return avg_grads

def get_grad_norm(loader, model, device, grads=None):
    if grads is None:
        grads = get_grads(loader, model, device)

    return torch.cat([grad.view(-1) for grad in grads]).norm()

def get_fngrads(loader, model, device, grads=None):
    """ Get filter normalized gradients.
    """
    if grads is None:
        grads = get_grads(loader, model, device)

    it_params = (p for p in model.parameters() if p.requires_grad)

    fngrads = []
    for param, grad in zip(it_params, grads):
        fngrads.append(grad * param.data.norm())

    return fngrads

def get_fngrad_norm(loader, model, device, fngrads=None, grads=None):
    if fngrads is None:
        fngrads = get_fngrads(loader, model, device, grads=grads)

    return torch.cat([fngrad.view(-1) for fngrad in fngrads]).norm()

def bayes_acc_check(loader):
    correct_samples, num_samples = 0, 0
    for (_, labels, conf) in loader:
        correct_samples += torch.sum(torch.max(conf, 1)[1] == labels).item()
        num_samples += labels.size(0)
    return 100 * correct_samples / num_samples

def surrogate_ber_check(loader):
    y_posterior_list = []
    for (_, labels, conf) in loader:
        temp_list = [c[l].item() for c,l in zip(conf, labels)]
        y_posterior_list.extend(temp_list)
    return np.mean(-np.log(np.array(y_posterior_list)))

def kl_check(loader, model, device):
    total, num_samples = 0, 0
    criterion = nn.KLDivLoss(reduction='sum')
    sm = nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        for (images, labels, confs) in loader:
            images, labels, confs = images.to(device), labels.to(device), confs.to(device)
            outputs = model(images)
            total += criterion(torch.log(sm(outputs)), confs)
            num_samples += len(labels)
    return total / num_samples

def softmax_check(loader, model, K, device):
    save_sm = torch.empty((0, K))
    sm = nn.Softmax(dim=1)
    model.eval()
    with torch.no_grad():
        for (images, _, confs) in loader:
            images, confs = images.to(device), confs.to(device)
            outputs = model(images)
            save_sm = torch.cat((save_sm, sm(outputs)))
    return save_sm.numpy()

def var_check(loader, model, device):
    ce_nosum = nn.CrossEntropyLoss(reduction='none')
    loss_vector = np.array([])
    model.eval()
    with torch.no_grad():
        for (images, labels, _) in loader:
            images, labels =images.to(device), labels.to(device)
            outputs = model(images)
            loss_vector = np.append(loss_vector, ce_nosum(outputs, labels).cpu().numpy())
    return np.var(loss_vector)


'''
def acc_check(loader, model, eceBool=False):
    total, num_samples, ece = 0, 0, None
    if eceBool:
        all_max_confs, all_predictions, all_labels = np.empty(0), np.empty(0), np.empty(0)
        bins_acc, bins_confs, bins_count = [], [], []
    for (images, labels, *empty) in loader:
        images, labels =images.to(device), labels.to(device)
        outputs = model(images)
        sm_outputs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(sm_outputs.data, 1)
        total += (predicted == labels).sum().item()
        num_samples += labels.size(0)
        if eceBool:
            max_confs = np.max(sm_outputs.detach().cpu().numpy(), 1)
            predictions = np.argmax(sm_outputs.detach().cpu().numpy(), 1)
            all_max_confs = np.concatenate([all_max_confs, max_confs])
            all_predictions = np.concatenate([all_predictions, predictions])
            all_labels = np.concatenate([all_labels, labels.detach().cpu().numpy()])
    if eceBool:
        num_bins=20
        bins = np.arange(num_bins+1.) / num_bins
        for i in range(num_bins-1):
            narrowed_down = (all_max_confs > bins[i]) & (all_max_confs < bins[i+1])
            if(sum(narrowed_down)!=0):
                bins_acc.append(sum(all_predictions[narrowed_down] == all_labels[narrowed_down]) / sum(narrowed_down))
                bins_confs.append(np.mean(all_predictions[narrowed_down]))
                bins_count.append(sum(narrowed_down))
            else:
                bins_acc.append(0), bins_confs.append(0), bins_count.append(0)
        ece = float(np.sum(np.absolute(np.array(bins_acc) - np.array(bins_confs)) * np.array(bins_count)) / num_samples)
    return 100 * total / num_samples, ece
'''

# def loss_check(loader, model):
#     total, num_samples = 0, 0
#     criterion = nn.CrossEntropyLoss(reduction='sum')
#     model.eval()
#     with torch.no_grad():
#         for (images, labels, *empty) in loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             total += criterion(outputs, labels.long())
#             num_samples += len(labels)
#     return total / num_samples
