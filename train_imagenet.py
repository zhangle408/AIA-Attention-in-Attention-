# Copyright 2017 Queequeg92.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
"""

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

from datetime import datetime
import numpy as np

import os
import sys
import time
import argparse

import models
from torch.autograd import Variable

from utils import AverageMeter, accuracy#, fast_collate, ColorJitter, Lighting
import torchvision.datasets as datasets

model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable(models.__dict__[name]))
#parser.add_argument add parameters
parser = argparse.ArgumentParser(description='PyTorch CIFAR Classification Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='PyramidNet',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: PyramidNet)')
parser.add_argument('--dataset', default='imagenet', type=str,
                    help='dataset imagenet')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=256, type=int,
                    help='mini-batch size (default: 96)')
parser.add_argument('--lr', default=0.05, type=float,
                    help='initial learning rate')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--alpha', default=1.0, type=float,
                    help='alpha')
parser.add_argument('--reduction_coord', default=32, type=int,
                    help='reduction_coord')
parser.add_argument('--reduction_dct', default=32, type=int,
                    help='reduction_dct')
parser.add_argument('--num_frequency', default=4, type=int,
                    help='num_frequency')
'''parser.add_argument('--lr_schedule', default=1, type=int,
                    help='learning rate schedule to apply')'''
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, action='store_true', help='nesterov momentum')
parser.add_argument('--weight_decay', default=4e-5, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default=False, action='store_true', help='resume from checkpoint')
parser.add_argument('--ckpt_path', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')

parser.add_argument('--seed', type=int, default = 2)


def main():
    global args
    args = parser.parse_args()

    # Data preprocessing.
    print('==> Preparing data......')
    assert (args.dataset == 'imagenet'), "Only support imagenet dataset"
    '''normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_data = datasets.ImageFolder('/data/data/Imagenet2012/ILSVRC2012_img_train',
                                      transform=transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]))
           
    validation_data = datasets.ImageFolder('/data/data/Imagenet2012/ILSVRC2012_img_val',
                                           transform=transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize]))
    
        # Create training dataloader
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=4)
        # Create validation dataloader
    test_loader = torch.utils.data.DataLoader(validation_data, batch_size=1, shuffle=False, num_workers=4)'''
    #traindir = os.path.join('/data/data/Imagenet2012/imagenet2012_train')
    #valdir = os.path.join('/data/data/Imagenet2012/imagenet2012_val')
    traindir = os.path.join('/data/data/Imagenet2012/ILSVRC2012_img_train')
    valdir = os.path.join('/data/data/Imagenet2012/ILSVRC2012_img_val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    '''jittering = ColorJitter(brightness=0.4, contrast=0.4,
                                  saturation=0.4)
    lighting = Lighting(alphastd=0.1,
                              eigval=[0.2175, 0.0188, 0.0045],
                              eigvec=[[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]])'''

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers)

    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers)
    #train_loader, train_loader_len = get_train_loader(, args.batch_size, workers=args.workers, input_size=224)
    #test_loader, val_loader_len = get_val_loader(, args.batch_size, workers=args.workers, input_size=224)
    num_classes = 1000
    
    os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
    # args.resume=True
    # Model
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(args.ckpt_path), 'Error: checkpoint directory not exists!'
        checkpoint = torch.load(os.path.join(args.ckpt_path,'ckpt.t7'))
        model = checkpoint['model']
        best_acc = checkpoint['best_acc']
        print (best_acc)
        start_epoch = checkpoint['epoch']
        print (start_epoch)
    else:
        print('==> Building model..'+' '+args.arch)
        print()
        # M2_C_imagenet
        # model = models.__dict__[args.arch](num_classes=num_classes,width_mult=args.alpha)
        # M2_D_C_I_imagenet
        model = models.__dict__[args.arch](num_classes=num_classes, width_mult=args.alpha, dct_reduction=args.reduction_dct,coord_reduction=args.reduction_coord,num_frequency=args.num_frequency)
        # MX_D_C_I_imagenet
        #model = models.__dict__[args.arch](num_classes=num_classes, width_mult=args.alpha, num_frequency=args.num_frequency,reduction_coord=args.reduction_coord,reduction_dct=args.reduction_dct)
        #MX_C_imagenet
        #print('args.alpha', args.alpha)
        #print('args.batch_size', args.batch_size)
        #model = models.__dict__[args.arch](num_classes=num_classes, width_mult=args.alpha)
        #D_C_I
        '''print('args.alpha', args.alpha)
        print('args.batch_size', args.batch_size)
        print('args.dct_reduction',args.dct_reduction)
        print('args.coord_reduction',args.coord_reduction)
        print('args.num_frequency',args.num_frequency)
        model = models.__dict__[args.arch](num_classes=num_classes, width_mult=args.alpha,dct_reduction=args.dct_reduction, coord_reduction=args.coord_reduction, num_frequency=args.num_frequency)'''
        assert (not model is None)
        start_epoch = args.start_epoch


    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters() if p.requires_grad and len(p.data.size())<5])))
    
    # Use GPUs if available.
    if torch.cuda.is_available():
        
        model = torch.nn.DataParallel(model)
        #model = torch.nn.parallel.DistributedDataParallel(model)
        model.cuda()
        #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.enabled = True
        torch.manual_seed(args.seed)
    # Define loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD([p for n, p in model.named_parameters() if p.requires_grad ], lr=args.lr, momentum = args.momentum,nesterov=args.nesterov, weight_decay = args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs) #change to multi-step
    log_dir = 'logs/'+args.arch + '-dataset-'+args.dataset+'-alpha-'+str(args.alpha)+'-'+time.strftime("%Y%m%d-%H%M%S")
    print ('log_dir: '+ log_dir)
    
    if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

    best_acc1 = 0  # best test accuracy
    best_acc5 = 0  # best test accuracy

    for epoch in range(start_epoch, args.epochs):
        # Learning rate schedule.
        #lr = adjust_learning_rate(optimizer, epoch + 1)
        #lr = adjust_learning_rate(optimizer, epoch )

        scheduler.step()
        # Train for one epoch.
        losses_avg, acces_avg=train(train_loader, model, criterion, optimizer , epoch)

        # Eval on test set.
        num_iter = (epoch + 1) * len(train_loader)
        #
        losses_avg,acc1,acc5=eval(test_loader, model, criterion, epoch, num_iter)
  
        # Save checkpoint.
        print('Saving Checkpoint......')
        
        if torch.cuda.is_available():	
            state = {
                'model': model,
                'best_acc': best_acc1,
                'epoch': epoch,
                }
            state5 = {
                'model': model,
                'best_acc': best_acc5,
                'epoch': epoch,
                }
        else:
            state = {
                'model': model,
                'best_acc': best_acc1,
                'epoch': epoch,
                }
            state5 = {
                'model': model,
                'best_acc': best_acc5,
                'epoch': epoch,
                }
        if not os.path.isdir(os.path.join(log_dir, 'last_ckpt')):
            os.mkdir(os.path.join(log_dir, 'last_ckpt'))
        torch.save(state, os.path.join(log_dir, 'last_ckpt', 'ckpt.t7'))
        if acc1 > best_acc1:
            best_acc1 = acc1
            if not os.path.isdir(os.path.join(log_dir, 'best1_ckpt')):
                os.mkdir(os.path.join(log_dir, 'best1_ckpt'))
            torch.save(state, os.path.join(log_dir, 'best1_ckpt', 'ckpt.t7'))
        if acc5 > best_acc5:
            best_acc5 = acc5
            if not os.path.isdir(os.path.join(log_dir, 'best5_ckpt')):
                os.mkdir(os.path.join(log_dir, 'best5_ckpt'))
            torch.save(state5, os.path.join(log_dir, 'best5_ckpt', 'ckpt.t7'))
       
    print('best_acc1:', best_acc1)
    print('best_acc5:', best_acc5)
   

def adjust_learning_rate(optimizer, epoch):
    if args.lr_schedule == 0:
        lr = args.lr * ((0.2 ** int(epoch >= 80)) * (0.2 ** int(epoch >= 160)) * (0.2 ** int(epoch >= 240)))
    elif args.lr_schedule == 1:
        lr = args.lr * ((0.1 ** int(epoch >= 150)) * (0.1 ** int(epoch >= 225)))
    elif args.lr_schedule == 2:
        lr = args.lr * ((0.1 ** int(epoch >= 80)) * (0.1 ** int(epoch >= 120)))
    else:
        raise Exception("Invalid learning rate schedule!")
    
    for param_group in optimizer.param_groups:
        	param_group['lr'] = lr
    return lr



# Training
def train(train_loader, model, criterion, optimizer, epoch):
    print('\nEpoch: %d -> Training' % epoch)
    # Set to eval mode.
    model.train()
    sample_time = AverageMeter()
    losses = AverageMeter()
    acces1 = AverageMeter()
    acces5 = AverageMeter()

    end = time.time()
    
    #each batch calculates the values of loss and gradient
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        num_iter = epoch * len(train_loader) + batch_idx
        
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        # Compute gradients and do back propagation.
        outputs = model(inputs)
        #crossEntrypyLoss Criterion
        #criterion(outputs, targets)loss(x, class) = weight[class] * (-x[class] + log(\sum_j exp(x[j])))
        loss = criterion(outputs, targets)
        
        loss.backward()
        #model.copy_grad(args.balance_weight)
        #update the parameters after calcute the gradient with backward()
        optimizer.step()
        #if epoch < args.warmup:
            #model.copy_grad(args.balance_weight)
        optimizer.zero_grad()
        #losses.update(loss.item()*inputs.size(0), inputs.size(0))  
        #_, predicted = torch.max(outputs.data, 1)
        #correct = predicted.eq(targets.data).cpu().sum()
        prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
        losses.update(loss.item(), inputs.size(0))
        acces1.update(prec1[0], inputs.size(0))
        acces5.update(prec5[0], inputs.size(0))
        # measure elapsed time
        sample_time.update(time.time() - end, inputs.size(0))
        end = time.time()
    print('Loss: %.4f | Acc1: %.4f | Acc5: %.4f%% (%d/%d)' % (losses.avg, acces1.avg, acces5.avg, acces1.sum/100, acces1.count))
    return losses.avg, acces1.avg
    
# Evaluating
def eval(test_loader, model, criterion,  epoch, num_iter):
    print('\nEpoch: %d -> Evaluating' % epoch)
    # Set to eval mode.
    model.eval()
    losses = AverageMeter()
    acces1 = AverageMeter()
    acces5 = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
        #inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        with torch.no_grad():
            outputs = model(inputs)
            #calculate the losses
            loss = criterion(outputs, targets)

            losses.update(loss.item(), inputs.size(0))
            #_, predicted = torch.max(outputs.data, 1)
            #correct = predicted.eq(targets.data).cpu().sum()
            prec1, prec5 = accuracy(outputs.data, targets, topk=(1, 5))
            acces1.update(prec1[0], inputs.size(0))
            acces5.update(prec5[0], inputs.size(0))

        
    print('Loss: %.4f | Acc1: %.4f | Acc5: %.4f%% (%d/%d)' % (losses.avg,  acces1.avg,  acces5.avg, acces1.sum/100, acces1.count))
    
    return losses.avg,acces1.avg,acces5.avg
    



if __name__ == '__main__':
    main()


