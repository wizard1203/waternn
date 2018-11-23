import os
from collections import namedtuple
import time
from torch.nn import functional as F
import logging
from torch import nn
import torch as t
# from utils import array_tool as at
# from utils.vis_tool import Visualizer

from config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter


class WaterNet(nn.Module):
    def __init__(self):
        super(WaterNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=1)
        self.fc1 = nn.Linear(768, 200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 17)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 768)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        # print('=======after fc ======{}===='.format(x))
        return F.log_softmax(x, dim=1)

    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer 

class WaterNetSmallFC(nn.Module):
    def __init__(self):
        super(WaterNet, self).__init__()
        self.fc1 = nn.Linear(384, 100)
        self.fc2 = nn.Linear(100, 40)
        self.fc3 = nn.Linear(40, 17)

    def forward(self, x):
        x = x.view(-1, 384)
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = self.fc3(x)
        # print('=======after fc ======{}===='.format(x))
        return F.log_softmax(x, dim=1)

    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer 

class WaterNet(nn.Module):
    def __init__(self):
        super(WaterNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=1)
        self.fc1 = nn.Linear(768, 200)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 17)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(-1, 768)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        # print('=======after fc ======{}===='.format(x))
        return F.log_softmax(x, dim=1)

    def get_optimizer(self):
        """
        return optimizer, It could be overwriten if you want to specify 
        special optimizer
        """
        lr = opt.lr
        params = []
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': opt.weight_decay}]
        if opt.use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer 