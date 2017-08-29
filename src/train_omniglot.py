import click
import os, sys
import numpy as np
from setproctitle import setproctitle
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import torchvision.transforms as transforms
from torch.nn.modules.loss import CrossEntropyLoss

from task import Task
from omniglot_net import Classifier
from omniglot_dataset import Omniglot
from custom_modules import *

# Set hyper-parameters
DEBUG=0
num_classes = 5
num_shot = 1
inner_batch_size = 5

def get_data_loader(task, split='train'):
    dset = Omniglot(task, transform=transforms.ToTensor(), split=split) 
    print 'img ids', dset.img_ids
    print 'labels', dset.labels
    loader = DataLoader(dset, batch_size=inner_batch_size, shuffle=True, num_workers=1, pin_memory=True)
    return loader

def forward(net, loader):
    ''' Run all data in the loader through net and return loss '''
    for i, (in_, target) in enumerate(loader):
        if DEBUG:
            in_ = torch.ones((1, 5)) 
            target = torch.from_numpy(np.ones(1, dtype=np.int64)) 
        #print 'input sizes', in_.numpy().shape, target.cpu().numpy().shape
        input_var = torch.autograd.Variable(in_).cuda(async=True)
        target_var = torch.autograd.Variable(target).cuda(async=True)
        # Run the batch through the net, compute loss
        out = net.forward(input_var)
        #print 'output shape', out.data.cpu().numpy().shape
        #print 'Output', out.data.cpu().numpy()
        loss = loss_fn(out, target_var)
    return loss, out

def evaluate(net, loader):
    ''' Evaluate the net on the data in the loader '''
    num_correct = 0
    for i, (in_, target) in enumerate(loader):
        input_var = torch.autograd.Variable(in_).cuda(async=True)
        target_var = torch.autograd.Variable(target).cuda(async=True)
        out = net.forward(input_var)
        loss = loss_fn(out, target_var)
        num_correct += count_correct(np.argmax(out.data.cpu().numpy(), axis=1), target.numpy())
    return float(num_correct) / (len(loader)*inner_batch_size)

def count_correct(pred, target):
    ''' count number of correct predictions in a batch '''
    pairs = [ int(x==y) for (x, y) in zip(pred, target)]
    return sum(pairs)

def train_step(task):
    train_loader = get_data_loader(task)
    ##### Test net before training, should be random accuracy ####
    print 'Before training update', evaluate(net, train_loader)
    for i in range(10):
        #print 'weight\n', net.weights['fc.weight'].data.cpu().numpy()[:5,:5]
        loss,_ = forward(net, train_loader)
        print 'Loss', loss.data.cpu().numpy()
        #print 'Loss', loss.data.cpu().numpy()
        opt.zero_grad()
        loss.backward()
        #print 'grad\n', net.weights['fc.weight'].grad.data.cpu().numpy()[:5,:5]
        opt.step() 
        print 'Iter ', i, evaluate(net, train_loader)
    ##### Test net after training, should be better than random ####
    print 'After training update', evaluate(net, train_loader)

# Script
for i in range(5):
    print 'Run ', i
    net = Classifier(num_classes, batch_norm=True, debug=DEBUG)
    opt = SGD(net.weights.values(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    #opt = Adam(net.weights.values(), lr=1)
    loss_fn = CrossEntropyLoss()
    task = Task('../data/omniglot', num_classes, num_shot)
    train_step(task)

