from cmath import inf
import os
import csv
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

import consts as CONSTS
import utils as UTILS
import net as NET
from net import RMSE, MAPE, NSE, MSEEnhanceLoss
import datamanager as DataManager
from predict import plot_test, get_eval_values

# netdict保存路径
NET_DICT_DIR = 'net_dict_cnnlstmam'

# loss, accuracy计算方式
mse_loss = nn.MSELoss()
mae_acc = NET.L1Accuracy()

# 超参数：
# 4, 8, 16, 32... batch大小
BATCH_SIZE = 16
# 初始学习率 1e-2, 1e-3, 5e-4, ... 如果前几次训练输出迅速收敛到一个常数值，说明学习率太大
LR = 1e-3
# 训练轮数
EPOCHS = 200

# cpu/gpu
device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda')


def test(net: nn.Module, test_loader: DataLoader, loss_module: nn.Module, acc_module: nn.Module):
    loss_sum = 0
    acc_sum = 0
    net.eval()

    all_predicts = []
    all_targets = []

    for _, train_batch in enumerate(test_loader):

        input, target = train_batch
        input = input.to(device)
        target = target.to(device)
        with torch.no_grad():
            predict = net(input)
        loss = loss_module(predict, target)
        acc = acc_module(predict, target)

        loss_sum += loss.item()
        acc_sum += acc.item()

        all_predicts.append(predict.cpu())
        all_targets.append(target.cpu())

    all_predicts = torch.vstack(all_predicts)
    all_targets = torch.vstack(all_targets)

    RMSEs = []
    MAPEs = []
    NSEs = []
    for step in (0, 2, 4, 6):
        RMSEs.append(RMSE(all_predicts[:,step], all_targets[:,step]).item())
        MAPEs.append(MAPE(all_predicts[:,step], all_targets[:,step]).item())
        NSEs.append(NSE(all_predicts[:,step], all_targets[:,step]).item())

    return loss_sum/len(test_loader), acc_sum/len(test_loader), RMSEs, MAPEs, NSEs



def train(net: nn.Module, train_loader: DataLoader, test_loader: DataLoader, train_fig_loader: DataLoader):
    ''' 训练 '''
    record = []
    # optimizer. 加一个weight_decay防止过拟合/梯度爆炸
    optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=1e-6)
    # scheduler. 余弦退火，让学习率反复波动
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=72, eta_min=1e-5)

    # 记录最小nse值。测试集测试的nse比它小时，保存当前网络为best.pth
    lowest_nse = inf
    
    # log
    with open(f'{NET_DICT_DIR}/log.csv', 'w') as csv_file:
        csv_file.seek(0)
        csv_file.truncate()

    for epoch in range(1, EPOCHS+1):

        train_loss = 0
        train_acc = 0
        net.train()
        
        # 训练
        for _, train_batch in enumerate(train_loader):

            input, target = train_batch
            input = input.to(device)
            target = target.to(device)

            predict = net(input)

            loss = mse_loss(predict, target)
            accuracy = mae_acc(predict, target)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy.item()

        train_loss = train_loss/len(train_loader)
        train_acc = train_acc/len(train_loader)

        # 测试
        test_loss, test_acc, RMSEs, MAPEs, NSEs = test(net, test_loader, mse_loss, mae_acc)
        if epoch % 1 == 0:
            print('epoch: ', epoch, 'train loss: ', train_loss, 'train acc: ', train_acc, 'test loss: ', test_loss, 'test acc: ', test_acc, 'NSE_1: ', NSEs[0])
            with open(f'{NET_DICT_DIR}/log.csv', 'a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([epoch] + RMSEs + MAPEs + NSEs)

        record.append([train_loss, train_acc, test_loss, test_acc])

        # 保存网络参数；作图
        if lowest_nse > NSEs[0]:
            # 当nse最小时 保存+作图
            net = net.cpu()
            torch.save(net.state_dict(), f'{NET_DICT_DIR}/best.pth')
            plot_test(test_loader, net, dict_paths=None, step=0)
            plt.savefig(f'{NET_DICT_DIR}/best_fig.png')
            plt.cla()
            plt.close('all')
            print(epoch)
        if epoch % 20 == 0:
            # 每隔20轮，保存+作图
            net = net.cpu()
            torch.save(net.state_dict(), f'{NET_DICT_DIR}/{epoch}.pth')
            plot_test(train_fig_loader, net, dict_paths=None, step=0)
            plt.savefig(f'{NET_DICT_DIR}/{epoch}_fig_tr.png')
            plt.cla()
            plt.close('all')
            plot_test(test_loader, net, dict_paths=None, step=0)
            plt.savefig(f'{NET_DICT_DIR}/{epoch}_fig_te.png')
            plt.cla()
            plt.close('all')
        net = net.to(device)
        lowest_nse = min(lowest_nse, NSEs[0])

        scheduler.step()
        

if __name__ == '__main__':

    if not os.path.exists(NET_DICT_DIR):
        os.mkdir(NET_DICT_DIR)

    # load data
    train_test_ratio = 0.8
    train_fig_set = DataManager.HydroDataset('xy.pth', is_training=True, train_test_ratio=1 - train_test_ratio)
    train_set = DataManager.HydroDataset('xy.pth', is_training=True, train_test_ratio=train_test_ratio)
    test_set = DataManager.HydroDataset('xy.pth', is_training=False, train_test_ratio=train_test_ratio)

    train_loader = DataLoader(train_set, BATCH_SIZE, True)
    train_fig_loader = DataLoader(train_fig_set, BATCH_SIZE, False)
    test_loader = DataLoader(test_set, BATCH_SIZE, False)

    # 各种模型：

    # net = NET.HydroNetDense(input_channel=train_set.channels, output_channel=(train_set.step_to-train_set.step_from+1), seqlen=train_set.seq_len, hidden_channels=64, hidden_layers=4)
    # net = NET.HydroNetLSTM(input_channel=train_set.channels, output_channel=(train_set.step_to-train_set.step_from+1), lstm_hidden_channel=64, lstm_layers=1, bidirectional=True)
    # net = NET.HydroNetCNN(
    #     input_channel=train_set.channels,
    #     output_channel=(train_set.step_to-train_set.step_from+1),
    #     seqlen = train_set.seq_len,
    #     cnn_channels=[64,64,64])
    # net = NET.HydroNetCNNLSTM(
    #     input_channel=train_set.channels,
    #     output_channel=(train_set.step_to-train_set.step_from+1),
    #     cnn_channels=[32,64],
    #     lstm_hidden_channel=64, lstm_layers=2, bidirectional=True)
    net = NET.HydroNetCNNLSTMAM(
        input_channel=train_set.channels,
        output_channel=(train_set.step_to-train_set.step_from+1),
        cnn_channels=[32,64],
        lstm_hidden_channel=64, lstm_layers=2, bidirectional=True)
    # net = NET.HydroNetLSTMAM(input_channel=train_set.channels,
    #     output_channel=(train_set.step_to-train_set.step_from+1),
    #     lstm_hidden_channel=64, lstm_layers=1, bidirectional=True)
    
    net = net.to(device)
    
    train(net, train_loader, test_loader, train_fig_loader)

    net = net.cpu()
    net.load_state_dict(
        torch.load(f'{NET_DICT_DIR}/best.pth', map_location=torch.device('cpu'))
    )

    rs, rmses, mapes, nses = get_eval_values(test_loader, net, steps=[0,2,4,6])
    results = torch.tensor([rs, rmses, mapes, nses])
    with open('temp.csv', 'a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(results.tolist())
    print(results)