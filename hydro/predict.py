from turtle import Shape
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

import consts as CONSTS
import utils as UTILS
import net as NET
import datamanager as DataManager


def load_model(dict_path):
    net = NET.HydroNetLSTM(lstm_hidden_channel=32, lstm_layers=2, bidirectional=True)
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


def plot_train(train_loader, net, device='cpu'):
    targets = []
    predicts = []
    for _, train_batch in enumerate(train_loader):
        input, target = train_batch
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            predict = net(input)
        targets.extend(UTILS.denormalize_output(target).cpu().detach().tolist())
        predicts.extend(UTILS.denormalize_output(predict).cpu().detach().tolist())
    fig, axs = plt.subplots(1,1, figsize=(28, 5))
    axs.plot(targets, label='target')
    axs.plot(predicts, label='predict')
    plt.legend()


def plot_test(test_loader, net, device='cpu'):
    targets = []
    predicts = []
    for _, train_batch in enumerate(test_loader):
        input, target = train_batch
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            predict = net(input)
        targets.extend(UTILS.denormalize_output(target).cpu().detach().tolist())
        predicts.extend(UTILS.denormalize_output(predict).cpu().detach().tolist())
    fig, axs = plt.subplots(1,1, figsize=(28, 5))
    axs.plot(targets, label='target')
    axs.plot(predicts, label='predict')
    plt.legend()


if __name__ == '__main__':

    # # ************** predict with real data ***************
    # model_dict_path = 'net_dict/net_dict_best.pth'
    # net = load_model(model_dict_path)

    # train_test_ratio = 0.8
    # data = torch.from_numpy(np.load('norm_data.npy'))
    # train_size = int(len(data) * train_test_ratio)
    # # data = data[:train_size]
    # data = data[train_size:]

    # predicts = []
    # targets = []
    # losses1 = []
    # losses2 = []
    # loss1 = NET.MixMSEEnhanceLoss(logarithmic_ratio=0)
    # loss2 = NET.MixMSEEnhanceLoss(logarithmic_ratio=1)
    
    # for i in range(len(data) - 12):
    #     input = data[i:i+12,:13]
    #     target = data[i+12,13].item()

    #     pred = predict(input, net)
    #     # print(pred, output_data)
    #     pred = pred/200
    #     target = target/200
    #     predicts.append(pred)
    #     targets.append(target)
    #     losses1.append(loss1(torch.tensor(pred), torch.tensor(target)))
    #     losses2.append(loss2(torch.tensor(pred), torch.tensor(target)))

    # predicts = UTILS.denormalize_output(torch.tensor(predicts))
    # targets = UTILS.denormalize_output(torch.tensor(targets))
    # fig, axs = plt.subplots(1,1, figsize=(12, 5))
    # axs.plot(targets, label='target')
    # axs.plot(predicts, label='predict')
    # # axs.plot(losses1, label='l1')
    # # axs.plot(losses2, label='l2')
    # plt.legend()
    # plt.show()

    
    # # ************** predict with various net dicts ***************
    # train_test_ratio = 0.8
    # data = torch.from_numpy(np.load('norm_data.npy'))
    # train_size = int(len(data) * train_test_ratio)
    # # data = data[:train_size]
    # data = data[train_size:]
    # predicts = []
    # all_preds = []

    # models = ['net_dict_lesslog_best','net_dict_morelog_best','net_dict_lesslog_100','net_dict_morelog_100']

    # for i in models:
    #     model_dict_path = f'net_dict/{i}.pth'
    #     net = load_model(model_dict_path)

    #     # ************** predict with real data ***************

    #     predicts = []
    #     targets = []
        
    #     for i in range(len(data) - 12):
    #         input = data[i:i+12,:13]
    #         target = data[i+12,13].item()

    #         pred = predict(input, net)
    #         # print(pred, output_data)
    #         pred = pred/200
    #         target = target/200
    #         predicts.append(pred)
    #         targets.append(target)

    #     all_preds.append(predicts)

    # all_preds = UTILS.denormalize_output(torch.tensor(all_preds))
    # targets = UTILS.denormalize_output(torch.tensor(targets))

    # fig, axs = plt.subplots(1,1, figsize=(12, 5))
    # axs.plot(targets, c='grey', linestyle='--', label='target')
    # for index, i in enumerate(models):
    #     axs.plot(all_preds[index], label=f'predict {i}')
    # plt.legend()
    # plt.show()
    

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


    import argparse
    import csv

    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', '-f', type=str, help='文本文件，格式应为csv形式（逗号分隔），共12行13列', default='test_data.txt')
    args = parser.parse_args()

    data = []
    with open(args.filepath, 'r') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            row = [float(i) for i in row]
            data.append(row)

    data = torch.tensor(data)
    assert data.shape[0] == 12 and data.shape[1] == 13, '输入数据尺寸有误，应为12行13列'

    model_dict_path = 'net_dict/net_dict_lesslog_best.pth'
    net = load_model(model_dict_path)
    pred = predict(data, net)

    # pred = UTILS.denormalize_output(pred)
    print(pred)
