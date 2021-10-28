#from __future__ import print_function, division

import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os
import os.path

def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB',return_index=False,root=None):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader
        self.indexes = np.arange(len(self.imgs))
        self.return_index = return_index
        self.root = root

    def __getitem__(self, index):
        path, target = self.imgs[index]
        if self.root is not None:
            path = os.path.join(self.root,path)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.return_index:
            return img, target
        else:
            return img, target, self.indexes[index]

    def __len__(self):
        return len(self.imgs)

class SubDataset(Dataset):
    def __init__(self, dataset,indexes):
        self.dataset = dataset
        self.len = len(indexes)
        self.indexes = indexes

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img, target  = self.dataset[self.indexes[index]]
        return img,target,index
