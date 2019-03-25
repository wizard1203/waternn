import os
from collections import namedtuple
import time
from torch.nn import functional as F
import logging
from torch import nn
import torch as t

from config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter
from myoptimizer import get_optimizer

class WaterNetTrainer(nn.Module):

    def __init__(self, water_net):
        super(WaterNetTrainer, self).__init__()

        self.water_net = water_net

        # optimizer
        self.optimizer = get_optimizer(self.water_net)

    def forward(self, datas):
        pred = self.water_net(datas)
        return pred

    def train_step(self, label, datas):
        # switch to train mode
        self.water_net.train()

        self.optimizer.zero_grad()
        pred = self.forward(datas)
        loss = F.nll_loss(pred, label)
        loss.backward()
        self.optimizer.step()

        return loss, pred

    def scale_lr(self):
        lastlr = opt.lr
        opt.lr *= opt.lr_decay
        print("=========*** lr{} change to lr{}==========\n".format(lastlr, opt.lr))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = opt.lr
        return self.optimizer

    def save(self, save_optimizer=True, better = False, save_path=None):
        save_dict = dict()

        save_dict['model'] = self.water_net.state_dict()
        save_dict['config'] = opt._state_dict()
        # save_dict['vis_info'] = self.vis.state_dict()
        save_dict['optimizer'] = self.optimizer.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()
        
        if better:
            save_path = 'cur_best_params'
        else:
            # save_path = opt.save_path
            if opt.customize:
                save_name = 'model' + '_self_' + opt.arch + '_' + opt.optim + opt.kind + 'params.tar'
            else:
                save_name = 'model' + '_default_' + opt.arch + '_' + opt.optim + opt.kind + 'params.tar'
            save_path = os.path.join(opt.save_path, save_name)
            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
                
        print(save_path)
        t.save(save_dict, save_path)
        # self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False):

        state_dict = t.load(path)
        if 'model' in state_dict:
            self.water_net.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.water_net.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self
