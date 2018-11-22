import cupy as cp
import os

import matplotlib
import logging
from config import opt
from dataset import Dataset, TestDataset
from waternet import WaterNet
from torch.utils import data as data_
from trainer import WaterNetTrainer
from mymodels import MyModels as mymodels
# from utils import array_tool as at
# from utils.vis_tool import visdom_bbox
# from utils.eval_tool import eval_detection_voc
import os
import resource

best_acc1 = 0

def main(**kwargs):
    opt._parse(kwargs)

    # set path of saving log in opt._parse()
    # and save init info in opt._parse() too
    logging.debug('this is a logging debug message')
    main_worker()




def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (target, datas) in enumerate(val_loader):
            # if args.gpu is not None:
            #     input = input.cuda(args.gpu, non_blocking=True)
            # target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            target = target.cuda()
            datas = datas.cuda()
            output = model(datas)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
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

    return top1.avg


def main_worker():
    global best_acc1
    # gpu = opt.gpu



    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=False, \
                                  # pin_memory=True,
                                  num_workers=opt.train_num_workers)

    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )

    if opt.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(opt.arch))
        if args.customize:
            print("=> self-defined model '{}'".format(opt.arch))
            model = mymodels.__dict__[opt.arch]
        else:
            model = models.__dict__[opt.arch]()

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
        validate(val_loader, model, criterion)
        return

    for epoch in range(opt.epoch):
        #trainer.reset_meters()
        train(train_loader, epoch)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)


        # best_path = trainer.save(best_map=best_map)

        # if epoch == 9:
        #     trainer.load(best_path)
        #     trainer.faster_rcnn.scale_lr(opt.lr_decay)
        #     lr_ = lr_ * opt.lr_decay




def train(train_loader, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    # model.train()

    for ii, (label_, datas_) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        datas, label = datas_.cuda().float(), label_.cuda()
        trainloss, output = trainer.train_step(label, datas)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, label, topk=(1, 5))
        losses.update(loss.item(), datas.size(0))
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
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

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


if __name__ == '__main__':
    import fire
    fire.Fire()

