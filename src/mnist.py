import os
import random
import numpy as np
import torch
from PIL import Image

class MNIST(object):
    '''
    Load MNIST images from PNG dataset
    '''

    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.dataset = 'mnist'
        self.root = root + '/' + split
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.all_ids = []
        for i in range(10):
            d = os.path.join(root, self.split, str(i))
            files = os.listdir(d)
            ids = [ str(i) + '/' + f[:-4] for f in files]

            # TODO: using the first 10k train examples and 5k test examples for the online training experiment
            if self.split == 'train':
                self.all_ids += ids[:1000]
            else:
                self.all_ids += ids[:500]
        self.all_labels = self.get_label(self.all_ids)

    def load_image(self, idx):
        ''' Load image '''
        im = Image.open('{}/{}.png'.format(self.root, idx)).convert('RGB')
        im = np.array(im, dtype=np.float32)
        im = im / 255.
        return im

    def get_label(self, img_ids):
        orig_labels = [int(x[0]) for x in img_ids]
        return np.array(orig_labels)
    
    def __getitem__(self, idx):
        img_id = self.all_ids[idx]
        img = self.load_image(img_id)
        if self.transform is not None:
            img = self.transform(img)
        target = self.all_labels[idx]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.all_labels)


