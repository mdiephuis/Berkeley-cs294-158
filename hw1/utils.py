import torch
from torch import nn
import torch.nn.init as init
from torch.utils.data import Dataset
import torchvision.transforms as T

import pickle
import numpy as np


class CMNIST(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx, :]

        if self.transform is None:
            self.transform = T.ToTensor()
        x = self.transform(x)

        return x
def load_data(path):
    with open(path, 'rb') as fp:
        data = pickle.load(fp)
    train_valid = data['train'].astype(np.float32)
    test = data['test'].astype(np.float32)
    return train_valid, test


def type_tdouble(use_cuda=False):
    return torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.xavier_normal_(m.weight.data)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for sub_mod in m:
                init_weights(sub_mod)


def one_hot(labels, n_class, use_cuda=False):
    # Ensure labels are [N x 1]
    if len(list(labels.size())) == 1:
        labels = labels.unsqueeze(1)
    mask = type_tdouble(use_cuda)(labels.size(0), n_class).fill_(0)
    # scatter dimension, position indices, fill_value
    return mask.scatter_(1, labels, 1)