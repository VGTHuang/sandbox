import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

import consts as CONSTS
import utils as UTILS
import hydronet as NET
import datamanager as DataManager


def load_model(dict_path):
    net = NET.HydroNetCNNLSTM(input_channel=13, conv_channel=32, conv_filter_sizes=[1,2,3,4], lstm_hidden_channel=64, lstm_layers=2, bidirectional=True)
    net.load_state_dict(
        torch.load(dict_path, map_location=torch.device('cpu'))
    )
    return net

def predict(data: torch.Tensor, net: nn.Module, is_testing_data=False):
    '''
    expect data to be a [12, 13] tensor
    '''
    data = data.float()
    if not is_testing_data:
        data = UTILS.normalize_01_data(data)
    data = data.unsqueeze(0)
    net.eval()
    with torch.no_grad():
        pred = net(data)
    if not is_testing_data:
        pred = UTILS.denormalize_output(pred)
    return pred.item()
    
if __name__ == '__main__':
    # load model
    model_dict_path = 'net_dict/net_dict.pth'
    net = load_model(model_dict_path)

    train_test_ratio = 0.8

    # ************** predict with real data ***************
    data = torch.from_numpy(np.load('norm_data.npy'))
    train_size = int(len(data) * train_test_ratio)
    # data = data[:train_size]
    data = data[train_size:]

    predicts = []
    targets = []
    losses1 = []
    losses2 = []
    loss1 = NET.MixMSEEnhanceLoss(logarithmic_ratio=0)
    loss2 = NET.MixMSEEnhanceLoss(logarithmic_ratio=1)
    
    for i in range(len(data) - 12):
        input = data[i:i+12,:13]
        target = data[i+12,13].item()

        pred = predict(input, net)
        # print(pred, output_data)
        pred = pred/200
        target = target/200
        predicts.append(pred)
        targets.append(target)
        losses1.append(loss1(torch.tensor(pred), torch.tensor(target)))
        losses2.append(loss2(torch.tensor(pred), torch.tensor(target)))

    
    fig, axs = plt.subplots(1,1, figsize=(12, 5))
    axs.plot(targets, label='target')
    axs.plot(predicts, label='predict')
    # axs.plot(losses1, label='l1')
    # axs.plot(losses2, label='l2')
    plt.legend()
    plt.show()

    acc = NET.L1Accuracy()
    print(1-(1-acc(torch.tensor(targets), torch.tensor(predicts)))/200)

    
    # ************** predict with test sine data ***************
    # test_set = DataManager.SineDataset(data_size = 500, is_training=False, train_test_ratio=train_test_ratio)

    # predicts = []
    # targets = []
    
    # for i in range(len(test_set)):
    #     input, target = test_set[i]
    #     pred = predict(input, net, is_testing_data=True)
    #     predicts.append(pred)
    #     targets.append(target.item())
    
    # fig, axs = plt.subplots(1,1, figsize=(12, 5))
    # axs.plot(targets, label='target')
    # axs.plot(predicts, label='predict')
    # plt.legend()
    # plt.show()