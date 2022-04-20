import enum
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from vmdpy import VMD

import math
import random
import utils as UTILS
import consts as CONSTS

class AbstractHydroDataset(Dataset):

    def __init__(self, path, seq_len = CONSTS.SEQUENCE_LENGTH, is_training = True, train_test_ratio = 0.75, step_from=1, step_to=7):
        self.data = torch.load(path)
        self.data = UTILS.normalize_01_data(self.data)
        self.train_size = int(len(self.data) * train_test_ratio)
        self.is_training = is_training
        if is_training:
            self.data_size = self.train_size
        else:
            self.data_size = len(self.data) - self.train_size
        self.seq_len = seq_len
        self.step_from = step_from
        self.step_to = step_to
        self.channels = 12

    def __len__(self):
        return self.data_size

    def __getitem__(self, index) -> torch.Tensor:
        if self.is_training:
            d = self.data[index : index+self.seq_len+self.step_to+1]
        else:
            d = self.data[index+self.train_size-self.seq_len-self.step_to : index+self.train_size-self.seq_len+self.seq_len]
        return d


class HydroDataset(AbstractHydroDataset):
    
    def __init__(self, path, seq_len = CONSTS.SEQUENCE_LENGTH, is_training = True, train_test_ratio = 0.75, step_from=1, step_to=7):
        super(HydroDataset, self).__init__(path, seq_len, is_training, train_test_ratio, step_from, step_to)
        self.channels = 13

    def __getitem__(self, index):
        d = super(HydroDataset, self).__getitem__(index)
        input = d[:self.seq_len]
        output = d[self.seq_len+self.step_from-1:self.seq_len+self.step_to,12]
        # output_pow = torch.pow(output, 0.5)
        return input, output


class HydroDatasetCorr(AbstractHydroDataset):
    
    def __init__(self, path, seq_len = CONSTS.SEQUENCE_LENGTH, is_training = True, train_test_ratio = 0.75, step_from=1, step_to=7):
        super(HydroDatasetCorr, self).__init__(path, seq_len, is_training, train_test_ratio, step_from, step_to)
        self.channels = len(CONSTS.CORR)

    def __getitem__(self, index):
        d = super(HydroDatasetCorr, self).__getitem__(index)
        input = d[:self.seq_len, CONSTS.CORR]
        output = d[self.seq_len+self.step_from-1:self.seq_len+self.step_to,12]
        return input, output


class HydroDatasetVMD(AbstractHydroDataset):
    
    def __init__(self, path, seq_len = CONSTS.SEQUENCE_LENGTH, is_training = True, train_test_ratio = 0.75, step_from=1, step_to=7):
        super(HydroDatasetVMD, self).__init__(path, seq_len, is_training, train_test_ratio, step_from, step_to)
        #. some sample parameters for VMD  
        self.alpha = 2000       # moderate bandwidth constraint  
        self.tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
        self.K = 8              # 3 modes  
        self.DC = 0             # no DC part imposed  
        self.init = 1           # initialize omegas uniformly  
        self.tol = 1e-7
        self.channels = len(CONSTS.CORR) + self.K


    def __getitem__(self, index):
        d = super(HydroDatasetVMD, self).__getitem__(index)
        input = d[:self.seq_len, CONSTS.CORR]
        
        #. Run actual VMD code
        u, _, _ = VMD(input[:,-1].numpy(), self.alpha, self.tau, self.K, self.DC, self.init, self.tol)
        u = torch.from_numpy(u).float()
        input = torch.hstack((input, u.transpose(0,1)))

        output = d[self.seq_len+self.step_from-1:self.seq_len+self.step_to,12]
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

    
    # train_set = SineDataset(data_size = 90, is_training=True, train_test_ratio=train_test_ratio)
    # test_set = SineDataset(data_size = 90, is_training=False, train_test_ratio=train_test_ratio)

    train_set = HydroDataset('zt.pth', is_training=True, train_test_ratio=train_test_ratio)
    print(train_set[0][0].shape)
    # test_set = HydroDataset('xy.pth', is_training=False, train_test_ratio=train_test_ratio)
    # print(len(train_set))
    # torch.save(test_set[0], 'test_vmd.pth')

    # train_loader = DataLoader(train_set, 4, True)
    # test_loader = DataLoader(test_set, 4, False)

    # for i, a in enumerate(train_loader):
    #     test_input, test_output = a
    #     break
    # print(len(train_loader), len(test_loader), test_input.shape, test_input.min(), test_input.max(), test_output, test_output.shape)

    
    # train_set = SineDataset(data_size = 90, is_training=True, train_test_ratio=train_test_ratio)
    # test_set = SineDataset(data_size = 90, is_training=False, train_test_ratio=train_test_ratio)

    # print(len(train_set))
    # for i in range(len(test_set)):
    #     inp, tar = test_set[i]
    #     print(tar)