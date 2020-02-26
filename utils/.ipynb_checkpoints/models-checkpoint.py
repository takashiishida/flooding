import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import torch.nn.functional as F

    
class mlp_model(nn.Module):
    def __init__(self, input_dim, output_dim, middle_dim=500):
        super(mlp_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, middle_dim)
        self.bn1 = nn.BatchNorm1d(middle_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(middle_dim, middle_dim)
        self.bn2 = nn.BatchNorm1d(middle_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(middle_dim, middle_dim)
        self.bn3 = nn.BatchNorm1d(middle_dim)
        self.relu3 = nn.ReLU()        
        self.fc4 = nn.Linear(middle_dim, middle_dim)
        self.bn4 = nn.BatchNorm1d(middle_dim)
        self.relu4 = nn.ReLU()                
        self.fc5 = nn.Linear(middle_dim, output_dim)

    def forward(self, x):
        if len(x.shape) == 4:
            # for MNIST/Fashion/Kuzushiji
            x = x.view(-1, self.num_flat_features(x))
        out = x
        out = self.relu1(self.bn1(self.fc1(out)))
        out = self.relu2(self.bn2(self.fc2(out)))
        out = self.relu3(self.bn3(self.fc3(out)))
        out = self.relu4(self.bn4(self.fc4(out)))
        out = self.fc5(out)
        return out

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features