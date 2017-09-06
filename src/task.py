import os
import random
import numpy as np
import torch

class OmniglotTask(object):
    '''
    Sample a few-shot learning task from the Omniglot dataset
    Sample N-way k-shot train and val sets according to
     - split (dataset/meta level train or test)
     - N-way classification (sample this many chars)
     - k-shot (sample this many examples from each char class)
    Assuming that the validation set is the same size as the train set!
    '''

    def __init__(self, root, num_cls, num_inst, split='train'):
        self.dataset = 'omniglot'
        self.root = '{}/images_background'.format(root) if split == 'train' else '{}/images_evaluation'.format(root)
        self.num_cl = num_cls
        self.num_inst = num_inst
        # Sample num_cls characters and num_inst instances of each
        languages = os.listdir(self.root)
        chars = []
        for l in languages:
            chars += [os.path.join(l, x) for x in os.listdir(os.path.join(self.root, l))]
        random.shuffle(chars)
        classes = chars[:num_cls]
        labels = np.array(range(len(classes)))
        labels = dict(zip(classes, labels)) 
        instances = dict()
        # Now sample from the chosen classes to create class-balanced train and val sets
        self.train_ids = []
        self.val_ids = []
        for c in classes:
            # First get all isntances of that class
            temp = [os.path.join(c, x) for x in os.listdir(os.path.join(self.root, c))]
            instances[c] = random.sample(temp, len(temp))
            # Sample num_inst instances randomly each for train and val
            self.train_ids += instances[c][:num_inst]
            self.val_ids += instances[c][num_inst:num_inst*2]
        # Keep instances separated by class for class-balanced mini-batches
        self.train_labels = [labels[self.get_class(x)] for x in self.train_ids]
        self.val_labels = [labels[self.get_class(x)] for x in self.val_ids]
        

    def get_class(self, instance):
        return os.path.join(*instance.split('/')[:-1])

class MNISTTask(object):
    '''
    Sample a few-shot learning task from the MNIST dataset
    Tasks are created by shuffling the labels
    Sample N-way k-shot train and val sets according to
     - split (dataset/meta level train or test)
     - N-way classification (sample this many chars)
     - k-shot (sample this many examples from each char class)
    '''

    def __init__(self, root, num_cls, num_inst, split='train'):
        self.dataset = 'mnist'
        self.root = root + '/' + split
        self.split = split
        all_ids = []
        for i in range(10):
            d = os.path.join(root, self.split, str(i))
            files = os.listdir(d)
            ids.append([ str(i) + '/' + f[:-4] for f in files])

        # To create a task, we randomly shuffle the labels
        self.label_map = dict(zip(range(10), np.random.permutation(np.array(range(10)))))
        
        # Choose num_inst ids from each of 10 classes
        self.train_ids = []
        self.val_ids = []
        for i in range(10):
            permutation = list(np.random.permutation(np.array(range(len(all_ids[i]))))[:num_inst*2])
            self.train_ids += [all_ids[i][j] for j in permutation[:num_inst]]
            self.train_labels = self.relabel(self.train_ids)
            self.val_ids += [all_ids[i][j] for j in permutation[num_inst:]]
            self.val_labels = self.relabel(self.val_ids)

    def relabel(self, img_ids):
        ''' Remap labels to new label map for this task '''
        orig_labels = [int(x[0]) for x in img_ids]
        return [self.label_map[x] for x in orig_labels]
        return np.array([self.label_map[x] for x in labels])
