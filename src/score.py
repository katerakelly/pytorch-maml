import numpy as np

import torch
from torch.autograd import Variable

'''
Helper methods for evaluating a classification network
'''

def count_correct(pred, target):
    ''' count number of correct classification predictions in a batch '''
    pairs = [int(x==y) for (x, y) in zip(pred, target)]
    return sum(pairs)

def forward_pass(net, in_, target, weights=None):
    ''' forward in_ through the net, return loss and output '''
    input_var = Variable(in_).cuda(async=True)
    target_var = Variable(target).cuda(async=True)
    out = net.net_forward(input_var, weights)
    loss = net.loss_fn(out, target_var)
    return loss, out

def evaluate(net, loader, weights=None):
    ''' evaluate the net on the data in the loader '''
    num_correct = 0
    loss = 0
    for i, (in_, target) in enumerate(loader):
        batch_size = in_.numpy().shape[0]
        l, out = forward_pass(net, in_, target, weights)
        loss += l.data.cpu().numpy()[0]
        num_correct += count_correct(np.argmax(out.data.cpu().numpy(), axis=1), target.numpy())
    return float(loss) / len(loader), float(num_correct) / (len(loader)*batch_size)
