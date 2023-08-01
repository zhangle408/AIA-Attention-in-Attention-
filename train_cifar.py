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

from utils import mean_cifar10, std_cifar10, mean_cifar100, std_cifar100
from utils import AverageMeter, accuracy

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
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='dataset (cifar10 [default] or cifar100)')
parser.add_argument('--epochs', default=300, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=128, type=int,
                    help='mini-batch size (default: 96)')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('--alpha', default=1.0, type=float,
                    help='alpha')
parser.add_argument('--reduction_coord', default=32, type=int,
                    help='reduction_coord')
parser.add_argument('--reduction_dct', default=32, type=int,
                    help='reduction_dct')
parser.add_argument('--num_frequency', default=4, type=int,
                    help='num_frequency')
parser.add_argument('--lr_schedule', default=1, type=int,
                    help='learning rate schedule to apply')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--nesterov', default=True, action='store_true', help='nesterov momentum')
parser.add_argument('--weight_decay', default=4e-5, type=float,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--resume', default=False, action='store_true', help='resume from checkpoint')
parser.add_argument('--ckpt_path', default='', type=str, metavar='PATH',
                    help='path to checkpoint (default: none)')

parser.add_argument('--seed', type = int, default = 2)


def main():
    global args
    args = parser.parse_args()

    # Data preprocessing.
    print('==> Preparing data......')
    assert (args.dataset == 'cifar10' or args.dataset == 'cifar100'), "Only support cifar10 or cifar100 dataset"
    if args.dataset == 'cifar10':
        print('To train and eval on cifar10 dataset......')
        num_classes = 10
        is_cifar10=True
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            #transform the valuesof pixels [0,255] to the range[0.0,1.0] 
            transforms.ToTensor(),
            #normalize the Tensor with channel=(channel-mean)/std
            transforms.Normalize(mean_cifar10, std_cifar10),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar10, std_cifar10),
        ])
        train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=4)
    else:
        print('To train and eval on cifar100 dataset......')
        num_classes = 100
        is_cifar10=False
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar100, std_cifar100),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean_cifar100, std_cifar100),
        ])
        train_set = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

        test_set = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=4)
    os.environ['CUDA_VISIBLE_DEVICES']='0'
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
        #print('args.alpha', args.alpha)
        '''
        print('args.reduction_coord',args.reduction_coord)
        print('args.reduction_dct', args.reduction_dct)
        print('args.num_frequency', args.num_frequency)
        #M2_DCT_coord_index_atten
        model = models.__dict__[args.arch](num_classes=num_classes,alpha=args.alpha,num_frequency=args.num_frequency,reduction_coord=args.reduction_coord,reduction_dct=args.reduction_dct)'''
        #M2, M2_C, M2_C_I,
        #model = models.__dict__[args.arch](num_classes=num_classes,alpha=args.alpha)
        #M2_D_C_I,
        model = models.__dict__[args.arch](num_classes=num_classes,alpha=args.alpha, num_frequency=args.num_frequency,reduction_coord=args.reduction_coord,reduction_dct=args.reduction_dct)
        #M2_C_imagenet
        #model = models.__dict__[args.arch](num_classes=num_classes,width_mult=args.alpha)
        #M2_D_C_I_imagenet
        ##model = models.__dict__[args.arch](num_classes=num_classes, width_mult=args.alpha, dct_reduction=args.reduction_dct,coord_reduction=args.reduction_coord,num_frequency=args.num_frequency)

        #MX , MX_C,  MX_C_I, MX_C_imagenet
        #model = models.__dict__[args.arch](num_classes=num_classes,width_mult=args.alpha)
        #MX_D_C_I,
        #model = models.__dict__[args.arch](num_classes=num_classes, width_mult=args.alpha, num_frequency=args.num_frequency)
        #MX_D_C_I_imagenet
        #model = models.__dict__[args.arch](num_classes=num_classes, width_mult=args.alpha, num_frequency=args.num_frequency,reduction_coord=args.reduction_coord,reduction_dct=args.reduction_dct)

        assert (not model is None)
        start_epoch = args.start_epoch


    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters() if p.requires_grad and len(p.data.size())<5])))
    
    # Use GPUs if available.
    if torch.cuda.is_available():
        
        model = torch.nn.DataParallel(model)
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
    #scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs) #change to multi-step
    log_dir = 'logs/'+args.arch + '-dataset-'+args.dataset +'-alpha-'+str(args.alpha)+'-'+time.strftime("%Y%m%d-%H%M%S")#+'-reduction_dct-'+str(args.reduction_dct)+'-reduction_coord-'+str(args.reduction_coord)
    print ('log_dir: '+ log_dir)
    
    if not os.path.isdir(log_dir):
            os.mkdir(log_dir)

    best_acc = 0  # best test accuracy

    for epoch in range(start_epoch, args.epochs):
        # Learning rate schedule.
        #lr = adjust_learning_rate(optimizer, epoch + 1)
        lr = adjust_learning_rate(optimizer, epoch )

        #scheduler.step()
        # Train for one epoch.
        losses_avg, acces_avg=train(train_loader, model, criterion, optimizer , epoch)

        # Eval on test set.
        num_iter = (epoch + 1) * len(train_loader)
        #
        losses_avg,acc=eval(test_loader, model, criterion, epoch, num_iter)
  
        # Save checkpoint.
        print('Saving Checkpoint......')
        
        if torch.cuda.is_available():	
            state = {
                'model': model,
                'best_acc': best_acc,
                'epoch': epoch,
                }
        else:
            state = {
                'model': model,
                'best_acc': best_acc,
                'epoch': epoch,
                }
        if not os.path.isdir(os.path.join(log_dir, 'last_ckpt')):
            os.mkdir(os.path.join(log_dir, 'last_ckpt'))
        torch.save(state, os.path.join(log_dir, 'last_ckpt', 'ckpt.t7'))
        if acc > best_acc:
            best_acc = acc
            if not os.path.isdir(os.path.join(log_dir ,'best_ckpt')):
                os.mkdir(os.path.join(log_dir, 'best_ckpt'))
            torch.save(state, os.path.join(log_dir ,'best_ckpt', 'ckpt.t7'))
       
    print(best_acc)
   

def adjust_learning_rate(optimizer, epoch):
    if args.lr_schedule == 0:
        lr = args.lr * ((0.2 ** int(epoch >= 80)) * (0.2 ** int(epoch >= 160)) * (0.2 ** int(epoch >= 240)))
    elif args.lr_schedule == 1:
        lr = args.lr * ((0.1 ** int(epoch >= 150)) * (0.1 ** int(epoch >= 225)))
    elif args.lr_schedule == 2:
        lr = args.lr * ((0.1 ** int(epoch >= 60)) * (0.1 ** int(epoch >= 120))*(0.1 ** int(epoch >= 180)))
    else:
        raise Exception("Invalid learning rate schedule!")
    
    for param_group in optimizer.param_groups:
        	param_group['lr'] = lr
    return lr
'''def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if args.dataset.startswith('cifar'):
        if args.epochs==210:
            lr = args.lr * (0.25 ** int(epoch >= (60))) * (0.25 ** int(epoch>= (120)))* (0.25 ** int(epoch >=(160)))
        else:
            lr = args.lr * (0.1 ** (epoch // (args.epochs * 0.5))) * (0.1 ** (epoch // (args.epochs * 0.75)))
    elif args.dataset == ('imagenet'):
        if args.epochs == 300:
            lr = args.lr * (0.1 ** (epoch // 75))
        else:
            lr = args.lr * (0.1 ** (epoch // 30))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr'''


# Training
def train(train_loader, model, criterion, optimizer, epoch):
    print('\nEpoch: %d -> Training' % epoch)
    # Set to eval mode.
    model.train()
    sample_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

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
        acces.update(prec1[0], inputs.size(0))
        # measure elapsed time
        sample_time.update(time.time() - end, inputs.size(0))
        end = time.time()
        
    print('Loss: %.4f | Acc: %.4f%% (%d/%d)' % (losses.avg, acces.avg, acces.sum/100, acces.count))
    return losses.avg, acces.avg
    
# Evaluating
def eval(test_loader, model, criterion,  epoch, num_iter):
    print('\nEpoch: %d -> Evaluating' % epoch)
    # Set to eval mode.
    model.eval()
    losses = AverageMeter()
    acces = AverageMeter()
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
            acces.update(prec1[0], inputs.size(0))

        
    print('Loss: %.4f | Acc: %.4f%% (%d/%d)' % (losses.avg,  acces.avg, acces.sum/100, acces.count))
    
    return losses.avg,acces.avg
    



if __name__ == '__main__':
    main()


