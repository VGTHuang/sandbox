import torch
from torch.utils.data import Dataset
from vmdpy import VMD

import math
import random
import utils as UTILS
import consts as CONSTS

class AbstractHydroDataset(Dataset):
    ''' 基类 '''
    def __init__(self, path, seq_len = CONSTS.SEQUENCE_LENGTH, is_training = True, train_test_ratio = 0.75, step_from=1, step_to=7):
        self.data = torch.load(path)
        # 数据归一化；见具体实现
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
    '''
    数据集；input为数据中前其他所有列，output为数据中最后一列
    '''
    def __init__(self, path, seq_len = CONSTS.SEQUENCE_LENGTH, is_training = True, train_test_ratio = 0.75, step_from=1, step_to=7):
        '''
        :param seq_len: 序列长度
        :param is_training: True - 为训练集；数据从path获得的数据中属于训练集的那一部分数据中取得. False - 为测试集
        :param train_test_ratio: 训练集占全部的比例
        :param step_from: output起始位置。第n日是输入序列的最后一日，第```(n + step_from)```日为输出序列的第一日
        :param step_to: output终止位置
        '''
        super(HydroDataset, self).__init__(path, seq_len, is_training, train_test_ratio, step_from, step_to)
        self.channels = 13

    def __getitem__(self, index):
        d = super(HydroDataset, self).__getitem__(index)
        input = d[:self.seq_len]
        output = d[self.seq_len+self.step_from-1:self.seq_len+self.step_to,12]
        # output_pow = torch.pow(output, 0.5)
        return input, output


class HydroDatasetCorr(AbstractHydroDataset):
    '''
    只选择相关性强的项；用CONSTS.CORR调节输出的项
    '''
    def __init__(self, path, seq_len = CONSTS.SEQUENCE_LENGTH, is_training = True, train_test_ratio = 0.75, step_from=1, step_to=7):
        super(HydroDatasetCorr, self).__init__(path, seq_len, is_training, train_test_ratio, step_from, step_to)
        self.channels = len(CONSTS.CORR)

    def __getitem__(self, index):
        d = super(HydroDatasetCorr, self).__getitem__(index)
        input = d[:self.seq_len, CONSTS.CORR]
        output = d[self.seq_len+self.step_from-1:self.seq_len+self.step_to,12]
        return input, output


class HydroDatasetVMD(AbstractHydroDataset):
    '''
    增加一VMD分解
    '''
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
    用于测试的数据，为一正弦波形
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