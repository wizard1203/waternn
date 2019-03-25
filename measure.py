import argparse
import os
import random
import shutil
import time
import warnings
import sys
import logging
import GPUtil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from datasets import DatasetHDF5  
from networks.alexnet import AlexNet
from threading import Thread

class Measure:
    # io time
    io_time = 0

    # pretrained
    pretrained = None

    # architecture of network
    customize = True
    arch = 'waternet'

    train_num_workers = 8
    test_num_workers = 8

    # optimizers
    optim = 'SGD'
    use_adam = False

    # param for optimizer
    lr = 0.0000875
    weight_decay = 0.00001
    lr_decay = 0.5  #

    # record i-th log
    kind = '0'

    # set gpu :
    # gpu = True

    # visualization
    env = 'water-nn'  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    data = 'water'

    # training
    epoch = 14

    # if eval
    evaluate = False

    # debug
    # debug_file = '/tmp/debugf'

    test_num = 10000
    # model
    load_path = None
    save_path = '~/water/modelparams'

    
    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')
        if opt.customize:
            logging_name = 'log' + '_self_' + opt.arch + '_'+ opt.optim + opt.kind + '.txt' 
        else:
            logging_name = 'log' + '_default_' + opt.arch  + '_' + opt.optim + opt.kind + '.txt'
        if not os.path.exists('log'):
            os.mkdir('log')

        logging_path = os.path.join('log', logging_name) 
    
        logging.basicConfig(level=logging.DEBUG,
                        filename=logging_path,
                        filemode='a',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                        datefmt='%H:%M:%S')
        logging.info('Logging for {}'.format(opt.arch))
        logging.info('======user config========')
        logging.info(pformat(self._state_dict()))
        logging.info('==========end============')
        # logging.info('optim : [{}], batch_size = {}, lr = {}, weight_decay= {}, momentum = {}'.format( \
        #                 args.optim, args.batch_size,
        #                 args.lr, args.weight_decay, args.momentum) )

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

class GPUMonitor(Thread):
    def __init__(self, delay):
        super(GPUMonitor, self).__init__()
        self.stopped = False
        self.delay = delay # Time between calls to GPUtil
        self.start()
        self.GPUs = GPUtil.getGPUs()

    def getInfo():
    	reture [(self.GPUs[i].load, self.GPUs[i].memoryUtil, self.GPUs[i].memoryUsed)
    		 for i in range(len(GPUs)]

    def run(self):
        while not self.stopped:
            
            time.sleep(self.delay)

    def stop(self):
        self.stopped = True

class GapMeter(object):
    """Computes and stores the average and current value"""	
    def __init__(self):
    	self.reset()

    def reset(self):
    	self.start = 0
    	self.end = 0
    	self.gap = 0
    	self.avemeter = AverageMeter()
    	self.metering = False

    def update_start(self, start):
    	self.start = start
    	self.metering = True

    def update_end(self, end):
    	try:
    		if self.metering:
    			self.end = end
    			self.gap = end - start
    			self.avemeter.update(gap)
    			metering = False
    		else:
    			raise RuntimeError('not metering')
    	except:
			print('========please start to            meter before end it ==============')    		


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


opt = Config()


