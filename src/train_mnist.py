from setproctitle import setproctitle
import click
import numpy as np
import os, sys

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torchvision.transforms as transforms
from torch.nn.modules.loss import CrossEntropyLoss

from omniglot_net import OmniglotNet
from net_helpers import *
from mnist import MNIST

'''
Train convolutional network on the MNIST dataset classification task
Test different network initializations.
'''

def forward(net, in_, target, loss_fn):
    input_var = torch.autograd.Variable(in_).cuda(async=True)
    target_var = torch.autograd.Variable(target.long().cuda(async=True))
    out = net(input_var)
    loss = loss_fn(out, target_var)
    return out, loss

def accuracy(output, target):
    ''' Compute classification accuracy of output based on target ground truth '''
    # output should be batch x n_cl
    output = output.data.cpu().numpy()
    preds = np.argmax(output, axis=1)
    pairs = [int(x==y) for (x,y) in zip(preds, target)]
    return float(sum(pairs)) / output.shape[0]


def train(train_loader, val_loader, net, loss_fn, opt, epoch):
    tloss = []
    tacc = []
    vloss = []
    vacc = []
    for i, (in_, target) in enumerate(train_loader):
        net.train()
        out, loss = forward(net, in_, target, loss_fn)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 10 == 0:
            print 'Epoch {}, Iteration {} Loss: {}'.format(epoch, i, loss.data.cpu().numpy()[0])
            tl, ta = val(train_loader, net, loss_fn)
            vl, va = val(val_loader, net, loss_fn)
            print 'Training loss and acc:', tl, ta
            print 'Validation loss and acc:', vl, va
            tloss.append(tl)
            tacc.append(ta)
            vloss.append(vl)
            vacc.append(va)
    return tloss, tacc, vloss, vacc

def val(loader, net, loss_fn):
    net.eval() 
    losses = []
    acc = []
    for i, (in_, target) in enumerate(loader):
        out, loss = forward(net, in_, target, loss_fn)
        losses.append(loss.data.cpu().numpy()[0])
        acc.append(accuracy(out, target))
    losses = np.array(losses)
    return sum(losses) / len(losses), float(sum(acc)) / len(acc)

@click.command()
@click.argument('exp')
@click.option('--batch_size', default=1, type=int)
@click.option('--lr', default=.001)
@click.option('--gpu', default=0)
def main(exp, batch_size, lr, gpu):
    setproctitle(exp)
    output = '../output/{}'.format(exp)
    try:
        os.makedirs(output)
    except:
        pass
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    
    loss_fn = CrossEntropyLoss()

    net = OmniglotNet(10, loss_fn)
    # NOTE: load weights from pre-trained model
    net.load_state_dict(torch.load('../output/maml-mnist-10way-5shot/train_iter_4500.pth'))
    net.cuda()
    opt = Adam(net.parameters(), lr=lr)
    
    train_dataset = MNIST('../data/mnist/mnist_png', transform=transforms.ToTensor())
    test_dataset = MNIST('../data/mnist/mnist_png', split='test', transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, num_workers=1,
                                           pin_memory=True)
    val_loader = DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=False, num_workers=1,
                                           pin_memory=True)
    num_epochs = 10
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    for epoch in range(num_epochs):
        # train for 1 epoch
        t_loss, t_acc, v_loss, v_acc = train(train_loader, val_loader, net, loss_fn, opt, epoch)
        train_loss += t_loss
        train_acc += t_acc
        val_loss += v_loss
        val_acc += v_acc
        # eval on the val set
        np.save('{}/train_loss.npy'.format(output), np.array(train_loss))
        np.save('{}/train_acc.npy'.format(output), np.array(train_acc))
        np.save('{}/val_loss.npy'.format(output), np.array(val_loss))
        np.save('{}/val_acc.npy'.format(output), np.array(val_acc))

if __name__ == '__main__':
    main()
