import numpy as np
import random

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms

from dataset import Omniglot, MNIST

'''
Helpers for loading class-balanced few-shot tasks
from datasets
'''

class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pools
        of examples of size 'num_per_class' '''

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
