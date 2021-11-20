import os
import sys
import glob
import time
import copy
import random
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torch import Tensor
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

import resnet
from dataloader import FaceData
import torchvision.transforms as transforms
from torch.autograd import Variable

dset.CIFAR10
parser = argparse.ArgumentParser()

# Basic model parameters.
parser.add_argument('--mode', type=str, default='train',
                    choices=['train', 'test'])
parser.add_argument('--data', type=str, default='data/cifar10')
parser.add_argument('--autoaugment', action='store_true', default=False)
parser.add_argument('--output_dir', type=str, default='models')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--eval_batch_size', type=int, default=500)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--lr_max', type=float, default=0.01)
parser.add_argument('--lr_min', type=float, default=0)
parser.add_argument('--l2_reg', type=float, default=5e-4)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--grad_bound', type=float, default=5.0)
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

def built_face_detector( **kwargs):
    epoch = kwargs.pop('epoch')

    train_transform, valid_transform = utils._data_transforms()
    train_set =  FaceData(split='train',transform=train_transform)
    public_valid_set =  FaceData(split='public_valid',transform=valid_transform)
    private_valid_set = FaceData(split='private_valid', transform=valid_transform)
    # train_set = dset.CIFAR10(root=args.data, train=True, download=True,transform=train_transform)
    # valid_set = dset.CIFAR10(root=args.data, train=False, download=True,transform=valid_transform)
    train_queue = torch.utils.data.DataLoader(dataset=train_set,
                                              batch_size=args.batch_size,
                                              shuffle=True, pin_memory=True, num_workers=16)
    public_valid_queue = torch.utils.data.DataLoader(dataset=public_valid_set,
                                              batch_size=args.eval_batch_size,
                                              shuffle=False, pin_memory=True, num_workers=16)
    private_valid_set = torch.utils.data.DataLoader(dataset=private_valid_set,
                                              batch_size=args.eval_batch_size,
                                              shuffle=False, pin_memory=True, num_workers=16)

    model = resnet.resnet18(pretrained=True, num_classes=7)
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    logging.info("model params:", utils.display_parameters(model))
    model = model.cuda()

    train_criterion = nn.CrossEntropyLoss().cuda()
    eval_criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(
        [{'params': model.parameters(),'initial_lr':args.lr_max}],
        args.lr_max,
        momentum=0.9,
        weight_decay=args.l2_reg,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), args.lr_min, epoch)
    return train_queue, public_valid_queue,private_valid_set, model, train_criterion, eval_criterion, optimizer, scheduler


def train(train_queue, model, optimizer, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_queue):
        input = input.cuda().requires_grad_()
        target = target.cuda()

        input, target = Variable(input), Variable(target)
        optimizer.zero_grad()
        logits = model(input)
        # print(input.size())
        #
        # print(logits.size())
        # print(target.size())
        # print(target)
        # exit()
        loss = criterion(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_bound)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss.data, n)
        top1.update(prec1.data, n)
        top5.update(prec5.data, n)
        # print(step)
        if (step + 1) % 100 == 0:
            logging.info('train %03d loss %e top1 %f top5 %f', step + 1, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


def valid(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    with torch.no_grad():
        model.eval()
        for step, (input, target) in enumerate(valid_queue):
            bs, ncrops, c, h, w = np.shape(input)
            input = input.cuda()
            target = target.cuda()
            input = input.view(-1, c, h, w)

            input, target = Variable(input, volatile=True), Variable(target)
            logits = model(input)
            logits_avg = logits.view(bs, ncrops, -1).mean(1)  # avg over crops
            loss = criterion(logits_avg, target)

            prec1, prec5 = utils.accuracy(logits_avg, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)
            # print(logits)
            if (step + 1) % 100 == 0:
                logging.info('valid %03d %e %f %f', step + 1, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

def valid_private(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    with torch.no_grad():
        model.eval()
        for step, (input, target) in enumerate(valid_queue):
            bs, ncrops, c, h, w = np.shape(input)
            input = input.cuda()
            target = target.cuda()
            input = input.view(-1, c, h, w)

            input, target = Variable(input, volatile=True), Variable(target)
            logits = model(input)
            logits_avg = logits.view(bs, ncrops, -1).mean(1)  # avg over crops
            loss = criterion(logits_avg, target)

            prec1, prec5 = utils.accuracy(logits_avg, target, topk=(1, 5))
            n = input.size(0)
            objs.update(loss.data, n)
            top1.update(prec1.data, n)
            top5.update(prec5.data, n)
            # print(logits)
            if (step + 1) % 100 == 0:
                logging.info('valid %03d %e %f %f', step + 1, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU found!')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = True
    cudnn.benchmark = True

    logging.info("Args = %s", args)
    epoch = 0
    best_acc_top1=0
    train_queue, valid_queue, valid_private_queue, model, train_criterion, eval_criterion, optimizer, scheduler =  built_face_detector(epoch=epoch)


    while epoch < args.epochs:
        scheduler.step()
        logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
        train_acc, train_obj = train(train_queue, model, optimizer, train_criterion)
        logging.info('train_acc %f', train_acc)
        valid_acc_top1, valid_obj = valid(valid_queue, model, eval_criterion)
        logging.info('valid_acc %f', valid_acc_top1)
        epoch += 1
        is_best = False
        if valid_acc_top1 > best_acc_top1:
            best_acc_top1 = valid_acc_top1
            is_best = True
        utils.save(args.output_dir, args, model, epoch, optimizer, best_acc_top1, is_best)

if __name__=='__main__':
    # data_train = dset.CIFAR10(root=args.data, train=False, download=True,transform=transforms.ToTensor())
    # train_queue = torch.utils.data.DataLoader(
    #     data_train, batch_size=1, shuffle=True, pin_memory=True, num_workers=16)
    # print(data_train.__len__())
    # for step, (input, target) in enumerate(train_queue):
    #     print(input.size())
    #     print(target.size())
    #     print(input)
    #     print(target)
    #     exit()
    main()
