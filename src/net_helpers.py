import numpy as np
import random

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms

from dataset import Omniglot, MNIST

class ClassBalancedSampler(Sampler):
    ''' Samples 'num_per_class' examples each from 'num_cl' pools
        of examples of size 'num_inst' '''

    def __init__(self, num_per_class, num_cl, num_inst, replacement = True):
       self.num_per_class = num_per_class
       self.num_cl = num_cl
       self.num_inst = num_inst

    def __iter__(self):
       # return a single list of indices, assuming that items will be grouped by class
       batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
       batch = [item for sublist in batch for item in sublist]
       random.shuffle(batch)
       return iter(batch)

    def __len__(self):
       return 1
        
def get_data_loader(task, batch_size=1, split='train'):
    # NOTE: batch size here is # instances PER CLASS
    if task.dataset == 'mnist':
        dset = MNIST(task, transform=transforms.ToTensor(), split=split) 
    else:
        dset = Omniglot(task, transform=transforms.ToTensor(), split=split) 
    if split == 'train':
        sampler = ClassBalancedSampler(batch_size, task.num_cl, task.num_inst)
        loader = DataLoader(dset, batch_size=batch_size*task.num_cl, sampler=sampler, num_workers=1, pin_memory=True)
    else:
        loader = DataLoader(dset, batch_size=batch_size*task.num_cl, shuffle=False, num_workers=1, pin_memory=True)
    return loader

def count_correct(pred, target):
    ''' count number of correct predictions in a batch '''
    pairs = [int(x==y) for (x, y) in zip(pred, target)]
    return sum(pairs)

def forward_pass(net, in_, target, weights=None):
    ''' forward in_ through the net, return loss + output '''
    input_var = Variable(in_).cuda(async=True)
    target_var = Variable(target).cuda(async=True)
    out = net.net_forward(input_var, weights)
    loss = net.loss_fn(out, target_var)
    return loss, out

def evaluate(net, loader, weights=None):
    ''' Evaluate the net on the data in the loader '''
    num_correct = 0
    loss = 0
    for i, (in_, target) in enumerate(loader):
        batch_size = in_.numpy().shape[0]
        l, out = forward_pass(net, in_, target, weights)
        loss += l.data.cpu().numpy()[0]
        num_correct += count_correct(np.argmax(out.data.cpu().numpy(), axis=1), target.numpy())
    return float(loss) / len(loader), float(num_correct) / (len(loader)*batch_size)
