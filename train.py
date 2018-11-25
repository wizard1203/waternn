import cupy as cp
import argparse
import os
import random
import shutil
import time
import warnings
import sys
import logging
import matplotlib
import logging
from pprint import pprint
from pprint import pformat
from config import opt
from dataset import TrainDataset, TestDataset
import pprint
from pprint import pformat
from trainer import WaterNetTrainer
from mymodels import MyModels as mymodels
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
from torch.utils import data as data_
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
# from utils import array_tool as at
# from utils.vis_tool import visdom_bbox
# from utils.eval_tool import eval_detection_voc
# import resource

best_acc1 = 0

def main(**kwargs):
    """
    :param kwargs:
        input : CUDA_VISIBLE_DEVICES=0 python train.py main --arch waternetsf --lr 0.000075 --epoch 20 --kind b1
                --data_dir ~/water/waterdataset2 --save_path  '~/water/waternn/modelparams'
    :return:
    """
    
    opt._parse(kwargs)
    print(opt)
    # set path of saving log in opt._parse()
    # and save init info in opt._parse() too
    logging.debug('this is a logging debug message')
    main_worker()


def val_out(**kwargs):
    """
    :param kwargs:
        input  :  CUDA_VISIBLE_DEVICES=0 python train.py val_out --arch waternetsf -kind b1
                --data_dir ~/water/waterdataset2 --load_path '~/water/waternn/modelparams'
    :return:
    """
    opt._parse(kwargs)
    print("===========validate & predict mode ===============")
    criterion = nn.CrossEntropyLoss().cuda()
    opt.data_dir = kwargs['data_dir']
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=16,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    
    if kwargs['pretrained'] :
        print("=> using pre-trained model '{}'".format(opt.arch))
        model = models.__dict__[kwargs['arch']](pretrained=True)
    else:
        print("=> creating model '{}'".format(kwargs['arch']))
        if opt.customize:
            print("=> self-defined model '{}'".format(kwargs['arch']))
            model = mymodels.__dict__[kwargs['arch']]
        else:
            model = models.__dict__[kwargs['arch']]()
    
    trainer = WaterNetTrainer(model).cuda()
    trainer.load(kwargs['load_path'], parse_opt=True)
    print('load pretrained model from %s' % kwargs['loadpath'])
    acc1, acc5 = validate(test_dataloader, model, criterion, True)
    


def validate(val_loader, model, criterion, seeout = False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    outf = open('predict.txt','w')

    with torch.no_grad():
        end = time.time()
        for i, (target, datas) in enumerate(val_loader):
            # if args.gpu is not None:
            #     input = input.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            target = target.cuda()
            datas = datas.cuda().float()
            output = model(datas)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc, pred5 = accuracy(output, target, topk=(1, 5))
            if seeout:
                writepred = pred5.tolist()
                for item in writepred :
                    outf.writelines(str(item) + str(target.item()) + '\r\n')
                    
            acc1 = acc[0]
            acc5 = acc[1]
            losses.update(loss.item(), datas.size(0))
            top1.update(acc1[0], datas.size(0))
            top5.update(acc5[0], datas.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % opt.plot_every == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        logging.info(' validate-----* Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.val:.4f}'
              .format(top1=top1, top5=top5, loss=losses))
    if seeout:
        outf.writelines('* Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.val:.4f}\r\n'
                .format(top1=top1, top5=top5, loss=losses))
        outf.writelines('======user config========')
        outf.writelines(pformat(opt._state_dict()))
    outf.close()
    return top1.avg, top5.avg


def main_worker():
    global best_acc1
    # gpu = opt.gpu



    trainset = TrainDataset(opt)
    print('load data')
    train_dataloader = data_.DataLoader(trainset, \
                                  batch_size=16, \
                                  shuffle=False, \
                                  # pin_memory=True,
                                  num_workers=opt.train_num_workers)

    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=16,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )

    if opt.pretrained:
        print("=> using pre-trained model '{}'".format(opt.arch))
        model = models.__dict__[opt.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(opt.arch))
        if opt.customize:
            print("=> self-defined model '{}'".format(opt.arch))
            model = mymodels.__dict__[opt.arch]
        else:
            model = models.__dict__[opt.arch]()


    model.apply(weights_init)
    print('model construct completed')
    trainer = WaterNetTrainer(model).cuda()



    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    #trainer.vis.text(dataset.db.label_names, win='labels')
    #best_map = 0

    # define (criterion) for evaluation
    criterion = nn.CrossEntropyLoss().cuda()
    lr_ = opt.lr

    if opt.evaluate:
        validate(test_dataloader, model, criterion)
        return

    for epoch in range(opt.epoch):
        #trainer.reset_meters()
        train(train_dataloader, trainer, epoch)

        # evaluate on validation set
        validate(test_dataloader, model, criterion, seeout=False)

    validate(test_dataloader, model, criterion, seeout=True)
    trainer.save(save_optimizer=True, save_path=opt.save_path)

        # if epoch == 9:
        #     trainer.load(best_path)
        #     trainer.faster_rcnn.scale_lr(opt.lr_decay)
        #     lr_ = lr_ * opt.lr_decay




def train(train_loader, trainer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    # model.train()

    end = time.time()
    for ii, (label_, datas_) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        datas, label = datas_.cuda().float(), label_.cuda()
        trainloss, output = trainer.train_step(label, datas)
        # print('==========output=======[{}]===='.format(output))
        # measure accuracy and record loss
        acc, pred5 = accuracy(output, label, topk=(1, 5))
        acc1 = acc[0]
        acc5 = acc[1]
        losses.update(trainloss.item(), datas.size(0))
        top1.update(acc1[0], datas.size(0))
        top5.update(acc5[0], datas.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (ii + 1) % opt.plot_every == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, ii, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))
            logging.info(' train-----* ===Epoch: [{0}][{1}/{2}]\t Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.val:.4f}'
              .format(epoch, ii, len(train_loader), top1=top1, top5=top5, loss=losses))





def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred2 = pred.t()
        correct = pred2.eq(target.view(1, -1).expand_as(pred2))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res, pred
    
    
    

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


def weights_init(m):
    """ init weights of net   """
    if isinstance(m, nn.Linear): 
        nn.init.normal_(m.weight.data, mean=0, std=1)
        nn.init.normal_(m.bias.data, mean=0, std=1)
    if isinstance(m, nn.Conv2d): 
        nn.init.normal_(m.weight.data, mean=0, std=1)
        nn.init.normal_(m.bias.data, mean=0, std=1)
        
if __name__ == '__main__':
    import fire
    fire.Fire()

