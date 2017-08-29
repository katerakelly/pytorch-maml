import numpy as np
import random
import math
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable

from omniglot_net import OmniglotNet
from layers import *
from net_helpers import *

class InnerLoop(OmniglotNet):
    '''
    This module performs the inner loop of MAML
    The forward method updates weights with gradient steps on training data, 
    then computes and returns a meta-gradient w.r.t. validation data
    '''

    def __init__(self, num_classes, loss_fn, num_updates, step_size, batch_size, meta_batch_size, num_in_channels=3):
        super(InnerLoop, self).__init__(num_classes, loss_fn, num_in_channels)
        # Number of updates to be taken
        self.num_updates = num_updates

        # Step size for the updates
        self.step_size = step_size

        # PER CLASS Batch size for the updates
        self.batch_size = batch_size

        # for loss normalization 
        self.meta_batch_size = meta_batch_size
    

    def net_forward(self, x, weights=None):
        return super(InnerLoop, self).forward(x, weights)

    def _forward(self, in_, target, weights=None):
        ''' Run data through net, return loss and output '''
        input_var = torch.autograd.Variable(in_).cuda(async=True)
        target_var = torch.autograd.Variable(target).cuda(async=True)
        # Run the batch through the net, compute loss
        out = self.net_forward(input_var, weights)
        loss = self.loss_fn(out, target_var)
        return loss, out
    
    def forward(self, data, task=None):
        # data is a dummy argument, not used by the method
        if task == None:
            print 'Must pass a task to the inner loop'
            raise(Exception)
        train_loader = get_data_loader(task, self.batch_size)
        val_loader = get_data_loader(task, self.batch_size, split='val')
        ##### Test net before training, should be random accuracy ####
        tr_pre_loss, tr_pre_acc = evaluate(self, train_loader)
        val_pre_loss, val_pre_acc = evaluate(self, val_loader)
        fast_weights = OrderedDict((name, param) for (name, param) in self.named_parameters())
        for i in range(self.num_updates):
            print 'inner step', i
            in_, target = train_loader.__iter__().next()
            print 'targets', target.numpy()
            if i==0:
                loss, _ = self._forward(in_, target)
                grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
            else:
                loss, _ = self._forward(in_, target, fast_weights)
                grads = torch.autograd.grad(loss, fast_weights.values(), create_graph=True)
            #print 'grads of fast weights \n', [(np.mean(np.abs(g.data.cpu().numpy()[0])), g.data.cpu().numpy().shape) for g in grads]
            fast_weights = OrderedDict((name, param - self.step_size*grad) for ((name, param), grad) in zip(fast_weights.items(), grads))
        ##### Test net after training, should be better than random ####
        tr_post_loss, tr_post_acc = evaluate(self, train_loader, fast_weights)
        val_post_loss, val_post_acc = evaluate(self, val_loader, fast_weights) 
        print '\n Train Inner step Loss', tr_pre_loss, tr_post_loss
        print 'Train Inner step Acc', tr_pre_acc, tr_post_acc
        print '\n Val Inner step Loss', val_pre_loss, val_post_loss
        print 'Val Inner step Acc', val_pre_acc, val_post_acc
        
        # Compute the meta gradient and return it
        in_, target = val_loader.__iter__().next()
        loss,_ = self._forward(in_, target, fast_weights) 
        loss = loss / self.meta_batch_size # normalize loss
        grads = torch.autograd.grad(loss, self.parameters())
        meta_grads = {name:g for ((name, _), g) in zip(self.named_parameters(), grads)}
        metrics = (tr_post_loss, tr_post_acc, val_post_loss, val_post_acc)
        return metrics, meta_grads

