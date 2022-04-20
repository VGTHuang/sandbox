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
from net import PearsonR, RMSE, MAPE, NSE


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
    return pred.flatten()


def plot_train(train_loader, net, device='cpu'):
    net.eval()
    targets = []
    predicts = []
    for _, train_batch in enumerate(train_loader):
        input, target = train_batch
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            predict = net(input)
        targets.extend(UTILS.denormalize_output(target).cpu().detach()[:,0].tolist())
        predicts.extend(UTILS.denormalize_output(predict).cpu().detach()[:,0].tolist())
    fig, axs = plt.subplots(1,1, figsize=(28, 5))
    axs.plot(targets, label='target')
    axs.plot(predicts, label='predict')
    plt.legend()
    plt.show()


def plot_test(test_loader, net, dict_paths, device='cpu', step=0):
    net.eval()
    _, axs = plt.subplots(2,1,figsize=(16, 4))
    if not dict_paths:
        dict_paths = ['']
    target_plotted = False
    for dict_path in dict_paths:
        targets = []
        predicts = []
        targets_pow = []
        predicts_pow = []
        if dict_path != '':
            net.load_state_dict(
                torch.load(dict_path, map_location=torch.device('cpu'))
            )
        for _, test_batch in enumerate(test_loader):
            input, target = test_batch
            input = input.to(device)
            target = target.to(device)
            with torch.no_grad():
                predict = net(input)
            predicts.extend(UTILS.denormalize_output(predict[:]).cpu().detach()[:,step].tolist())
            predicts_pow.extend(torch.sqrt(predict[:]).cpu().detach()[:,step].tolist())
            if not target_plotted:
                targets.extend(UTILS.denormalize_output(target[:]).cpu().detach()[:,step].tolist())
                targets_pow.extend(torch.sqrt(target[:]).cpu().detach()[:,step].tolist())
            # predicts.extend((UTILS.denormalize_output(target)*0.99).cpu().detach()[:,step].tolist())
        predicts = torch.tensor(predicts)
        if not target_plotted:
            targets = torch.tensor(targets)
            axs[0].plot(targets, label='target', c='lime')
            axs[1].plot(targets_pow, label='target', c='lime')
            target_plotted = True
        nse = NSE(predicts, targets).item()
        axs[1].plot(predicts_pow, label=f'{dict_path} {nse}', linestyle='dotted')
        axs[0].plot(predicts, label=f'{dict_path} {nse}', linestyle='dotted')
    plt.legend()
    # plt.show()


def get_eval_values(test_loader, net, device='cpu', steps=[0,2,4,6]):
    net.eval()
    targets = []
    predicts = []
    for _, test_batch in enumerate(test_loader):
        input, target = test_batch
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            predict = net(input)
        targets += (UTILS.denormalize_output(target).cpu().detach().tolist())
        predicts += (UTILS.denormalize_output(predict).cpu().detach().tolist())
        # predicts.extend((UTILS.denormalize_output(target)*0.99).cpu().detach()[:,step].tolist())
    predicts = torch.tensor(predicts)
    targets = torch.tensor(targets)
    rs = []
    rmses = []
    mapes = []
    nses = []
    for i in steps:
        rs.append(PearsonR(predicts[:,i], targets[:,i]))
        rmses.append(RMSE(predicts[:,i], targets[:,i]).item())
        mapes.append(MAPE(predicts[:,i], targets[:,i]).item())
        nses.append(NSE(predicts[:,i], targets[:,i]).item())
    return rs, rmses, mapes, nses

def pred_from_file():
    import argparse
    import csv
    net = NET.HydroNetLSTM(input_channel=len(CONSTS.CORR), lstm_hidden_channel=32, lstm_layers=1, bidirectional=True)
    net.load_state_dict(
        torch.load(model_dict_path, map_location=torch.device('cpu'))
    )
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
    pred = predict(data, net)
    pred = UTILS.denormalize_output(pred)
    print(pred)
    return pred

if __name__ == '__main__':

    import csv

    NET_DICT_DIR = 'net_dict_vmd_lstm'

    # net = NET.HydroNetDense(hidden_channels=64, hidden_layers=4)
    # net = NET.HydroNetLSTM(lstm_hidden_channel=64, lstm_layers=1, bidirectional=True)
    # net = NET.HydroNetLSTM(input_channel=len(CONSTS.CORR), lstm_hidden_channel=32, lstm_layers=1, bidirectional=True)
    # net = NET.HydroNetCNNLSTM(lstm_hidden_channel=32, lstm_layers=1, bidirectional=True)
    # net = NET.HydroNetCNNLSTM(lstm_hidden_channel=32, lstm_layers=2, bidirectional=True)
    # net = NET.HydroNetCNNLSTMAM(conv_channel=32, conv_filter_sizes=[1,2,3], lstm_hidden_channel=32, lstm_layers=1, bidirectional=True)

    
    # # plot
    train_test_ratio = 0.8
    # test_set = DataManager.HydroDatasetCorr('norm_data.npy', is_training=False, train_test_ratio=train_test_ratio)
    # test_set = DataManager.HydroDataset('norm_data.npy', is_training=True, train_test_ratio=train_test_ratio)
    test_set = DataManager.HydroDataset('xy.npy', is_training=False, train_test_ratio=train_test_ratio)
    # net = NET.HydroNetDense(input_channel=train_set.channels, output_channel=(train_set.step_to-train_set.step_from+1) * 2, seqlen=train_set.seq_len, hidden_channels=64, hidden_layers=4)
    net = NET.HydroNetLSTM(input_channel=test_set.channels, output_channel=(test_set.step_to-test_set.step_from+1) * 2, seqlen=test_set.seq_len, lstm_hidden_channel=64, lstm_layers=1, bidirectional=True)
    test_loader = DataLoader(test_set, 4, False)
    plot_test(test_loader, net, dict_paths=[
        f'{NET_DICT_DIR}/best.pth',
        f'{NET_DICT_DIR}/100.pth'
    ], step=6)
    plt.show()
    rs, rmses, mapes, nses = get_eval_values(test_loader, net, steps=[0,2,4,6])
    results = torch.tensor([rs, rmses, mapes, nses])
    with open('temp.csv', 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(results.tolist())
    print(results)