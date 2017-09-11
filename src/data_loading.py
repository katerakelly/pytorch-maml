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
    '''
    Samples class-balanced batches from 'num_cl' pools each
    of size 'num_inst'
    If 'batch_cutoff' is None, indices for iterating over batches
    of the entire dataset will be returned
    Otherwise, indices for the number of batches up to the batch_cutoff
    will be returned
    (This is to allow sampling with replacement across training iterations) 
    '''

    def __init__(self, num_cl, num_inst, batch_cutoff=None):
       self.num_cl = num_cl
       self.num_inst = num_inst
       self.batch_cutoff = batch_cutoff

    def __iter__(self):
       '''return a single list of indices, assuming that items will be grouped by class '''
       # First construct batches of 1 instance per class
       batches = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)] for j in range(self.num_cl)]
       batches = [[batches[j][i] for j in range(self.num_cl)] for i in range(self.num_inst)]
       # Shuffle within each batch so that classes don't always appear in same order
       for sublist in batches:
           random.shuffle(sublist)

       if self.batch_cutoff is not None:
           random.shuffle(batches)
           batches = batches[:self.batch_cutoff]

       batches = [item for sublist in batches for item in sublist]
           
       return iter(batches)

    def __len__(self):
       return 1
        
def get_data_loader(task, batch_size=1, split='train'):
    # NOTE: batch size here is # instances PER CLASS
    if task.dataset == 'mnist':
        normalize = transforms.Normalize(mean=[0.13066, 0.13066, 0.13066], std=[0.30131, 0.30131, 0.30131])
        dset = MNIST(task, transform=transforms.Compose([transforms.ToTensor(), normalize]), split=split) 
    else:
        normalize = transforms.Normalize(mean=[0.92206, 0.92206, 0.92206], std=[0.08426, 0.08426, 0.08426])
        dset = Omniglot(task, transform=transforms.Compose([transforms.ToTensor(), normalize]), split=split) 
    sampler = ClassBalancedSampler(task.num_cl, task.num_inst, batch_cutoff = (None if split != 'train' else batch_size))
    loader = DataLoader(dset, batch_size=batch_size*task.num_cl, sampler=sampler, num_workers=1, pin_memory=True)
    return loader
