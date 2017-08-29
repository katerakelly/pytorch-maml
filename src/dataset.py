import sys
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data


class Omniglot(data.Dataset):
    """
    Load image-label pairs from a task to pass to Torch DataLoader
    Tasks consist of data and labels split into train / val splits
    """
    
    def __init__(self, task, split='train', seed=1337, transform=None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.root = self.task.root
        self.split = split
        self.img_ids = self.task.train_ids if self.split == 'train' else self.task.val_ids
        self.labels = self.task.train_labels if self.split == 'train' else self.task.val_labels
    
    def __len__(self):
        return len(self.img_ids)
    
    def load_image(self, idx):
        ''' Load image '''
        im = Image.open('{}/{}'.format(self.root, idx)).convert('RGB')
        im = im.resize((28,28), resample=Image.LANCZOS) # per Chelsea's implementation
        im = np.array(im, dtype=np.float32)
        im = im / 255.
        return im
    
    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        im = self.load_image(img_id)
        if self.transform is not None:
            im = self.transform(im)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return im, label

class MNIST(data.Dataset):
    """
    Load image-label pairs from a task to pass to Torch DataLoader
    Tasks consist of data and labels split into train / val splits
    """
    
    def __init__(self, task, split='train', seed=1337, transform=None, target_transform=None):
        self.transform = transform # Torch operations on the input image
        self.target_transform = target_transform
        self.task = task
        self.root = self.task.root
        self.split = split
        self.img_ids = self.task.train_ids if self.split == 'train' else self.task.val_ids
        self.labels = self.task.train_labels if self.split == 'train' else self.task.val_labels
    
    def load_image(self, idx):
        ''' Load image '''
        im = Image.open('{}/{}.png'.format(self.root, idx)).convert('RGB')
        im = np.array(im, dtype=np.float32)
        im = im / 255.
        return im

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img = self.load_image(img_id)
        if self.transform is not None:
            img = self.transform(img)
        target = self.labels[idx]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.labels)


