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
from waternet import WaterNet
from myoptimizer import get_optimizer

class WaterNetTrainer(nn.Module):
    """
    Args:
        
            
    """

    def __init__(self, water_net):
        super(WaterNetTrainer, self).__init__()

        self.water_net = water_net

        # optimizer
        self.optimizer = get_optimizer(self.water_net)

        # visdom wrapper
        # self.vis = Visualizer(env=opt.env)

    def forward(self, labels, datas):
        """

        Args:

        Returns:
            
        """
        pred = self.water_net(datas)
        return pred

    def train_step(self, label, datas):
        # switch to train mode
        self.water_net.train()

        self.optimizer.zero_grad()
        pred = self.forward(label, datas)
        loss = F.nll_loss(pred, label)
        loss.backward()
        self.optimizer.step()


        # self.update_meters(losses)
        return loss, pred

    def save(self, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.water_net.state_dict()
        save_dict['config'] = opt._state_dict()
        # save_dict['vis_info'] = self.vis.state_dict()
        save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/waternn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        # self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self
