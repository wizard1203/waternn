from __future__ import  absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F

from torch import nn
import torch as t
# from utils import array_tool as at
# from utils.vis_tool import Visualizer

from config import opt



def get_optimizer(model):
    """
    return optimizer, It could be overwriten if you want to specify 
    special optimizer
    """
    if opt.use_adam:
        optimizer = t.optim.Adam(model.parameters(), 
            lr = opt.lr, weight_decay = opt.weight_decay)
    else:
        optimizer = t.optim.SGD(model.parameters(), 
            lr = opt.lr, weight_decay = opt.weight_decay, momentum=0.9)
    return optimizer 
