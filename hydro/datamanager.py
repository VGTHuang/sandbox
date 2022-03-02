import enum
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import math
import random
import utils as UTILS

class HydroDataset(Dataset):
    
    def __init__(self, path, seq_len = 12, is_training = True, train_test_ratio = 0.75):
        self.data = torch.from_numpy(np.load(path)).type(torch.float)
        train_size = int(len(self.data) * train_test_ratio)
        if is_training:
            self.data = self.data[:train_size]
        else:
            self.data = self.data[train_size:]
        self.seq_len = seq_len

    def __len__(self):
        return self.data.shape[0] - self.seq_len

    def __getitem__(self, index):
        d = self.data[index:index+self.seq_len]
        d = UTILS.normalize_01_data(d.clone())
        input = d[:,:13]
        output = d[self.seq_len-1,13]
        return input, output

class SineDataset(Dataset):
    '''
    a testing dataset
    '''
    def __init__(self, channels=13, seq_len = 12, data_size = 500, is_training = True, train_test_ratio = 0.75):
        train_size = int(data_size * train_test_ratio)
        self.seq_len = seq_len
        self.channels = channels
        self.is_training = is_training
        if is_training:
            self.size = train_size
        else:
            self.size = data_size - train_size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.is_training:
            phase = random.random() * math.pi * 2.0
        else:
            phase = index * math.pi * 2.0 / self.size
        hor = torch.arange(0, self.seq_len) * math.pi * 2 / self.seq_len
        ver = torch.arange(0, self.channels) * math.pi * 2 / self.channels

        hor = hor[:,None]
        ver = ver[None,:]
        hor = hor.expand(self.seq_len, self.channels)
        ver = ver.expand(self.seq_len, self.channels)

        input = torch.sin(hor + ver + phase).float()
        target = torch.sin(torch.tensor([phase]))
        target = target[0]
        return (input + 1) / 2, (target + 1) / 2

if __name__ == '__main__':
    train_test_ratio = 0.8

    
    train_set = SineDataset(data_size = 90, is_training=True, train_test_ratio=train_test_ratio)
    test_set = SineDataset(data_size = 90, is_training=False, train_test_ratio=train_test_ratio)

    # train_set = HydroDataset('norm_data.npy', is_training=True, train_test_ratio=train_test_ratio)
    # test_set = HydroDataset('norm_data.npy', is_training=False, train_test_ratio=train_test_ratio)

    train_loader = DataLoader(train_set, 4, True)
    test_loader = DataLoader(test_set, 4, False)

    for i, a in enumerate(train_loader):
        test_input, test_output = a
        break
    print(len(train_loader), len(test_loader), test_input.shape, test_input.min(), test_input.max(), test_output, test_output.shape)

    
    # train_set = SineDataset(data_size = 90, is_training=True, train_test_ratio=train_test_ratio)
    # test_set = SineDataset(data_size = 90, is_training=False, train_test_ratio=train_test_ratio)

    # print(len(train_set))
    # for i in range(len(test_set)):
    #     inp, tar = test_set[i]
    #     print(tar)