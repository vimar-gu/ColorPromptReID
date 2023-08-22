from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
import torch
from PIL import Image

class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = []#dataset
        for inds, item in enumerate(dataset):
            self.dataset.append(item+(inds,))
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fname, pid, camid, inds = self.dataset[index]

        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return [img, fname, pid, camid, inds]


class TwinsPreprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, basic_transform=None):
        super(TwinsPreprocessor, self).__init__()
        self.dataset = []#dataset
        for inds, item in enumerate(dataset):
            self.dataset.append(item+(inds,))
        self.root = root
        self.transform = transform
        self.basic_transform = basic_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        fname, pid, camid, inds = self.dataset[index]

        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            trans_img = self.transform(img)
        if self.basic_transform is not None:
            img = self.basic_transform(img)

        return [img, trans_img, fname, pid, camid, inds]
