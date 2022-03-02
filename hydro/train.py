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

mix_loss = NET.MixMSEEnhanceLoss(logarithmic_ratio=0.9)
mae_acc = NET.L1Accuracy()

BATCH_SIZE = 4
LR = 1e-2
EPOCHS = 100
SCHEDULER_MILESTONES = [10, 25, 50]

device = 'cpu'
if torch.cuda.is_available():
    device = torch.device('cuda')

def test(net: nn.Module, test_loader: DataLoader, loss_module: nn.Module, acc_module: nn.Module):
    loss_sum = 0
    acc_sum = 0
    net.eval()

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

    net.train()

    return loss_sum/len(test_loader), acc_sum/len(test_loader)



def train(net: nn.Module, train_loader: DataLoader, test_loader: DataLoader):

    record = []
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=SCHEDULER_MILESTONES, gamma=0.5)

    max_train_acc = 0
    max_test_acc = 0

    for epoch in range(1, EPOCHS+1):

        train_loss = 0
        train_acc = 0

        # mix_loss.logarithmic_ratio = (EPOCHS - epoch * 0.5) / EPOCHS
        mix_loss.logarithmic_ratio = 0.9
        
        for _, train_batch in enumerate(train_loader):

            input, target = train_batch
            input = input.to(device)
            target = target.to(device)

            predict = net(input)

            loss = mix_loss(predict, target)
            accuracy = mae_acc(predict, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += accuracy.item()

        train_loss = train_loss/len(train_loader)
        train_acc = train_acc/len(train_loader)
        test_loss, test_acc = test(net, test_loader, mix_loss, mae_acc)
        if epoch % 1 == 0:
            print('epoch count: ', epoch, 'train loss: ', train_loss, 'train acc: ', train_acc, 'test loss: ', test_loss, 'test acc: ', test_acc)

        record.append([train_loss, train_acc, test_loss, test_acc])

        if max_test_acc < test_acc:
            torch.save(net.state_dict(), 'net_dict/net_dict.pth')
            print(epoch)
        max_train_acc = max(max_train_acc, train_acc)
        max_test_acc = max(max_test_acc, test_acc)

        scheduler.step()

    return record, max_train_acc, max_test_acc
        


if __name__ == '__main__':
    # load data
    train_test_ratio = 0.8
    train_set = DataManager.HydroDataset('norm_data.npy', is_training=True, train_test_ratio=train_test_ratio)
    test_set = DataManager.HydroDataset('norm_data.npy', is_training=False, train_test_ratio=train_test_ratio)
    # train_set = DataManager.SineDataset(data_size = 500, is_training=True, train_test_ratio=train_test_ratio)
    # test_set = DataManager.SineDataset(data_size = 500, is_training=False, train_test_ratio=train_test_ratio)

    train_loader = DataLoader(train_set, BATCH_SIZE, True)
    test_loader = DataLoader(test_set, BATCH_SIZE, False)

    # hn = NET.HydroNet(input_channel=13, conv_channel=16, lstm_hidden_channel=32, lstm_layers=2, bidirectional=True)
    # hn = hn.to(device)
    
    # rec, max_train_acc, max_test_acc = train(hn, train_loader, test_loader)
    # print(max_train_acc, max_test_acc)

    # # lstm channel test
    # perf_rec = []
    # lstm_channels = [16, 32, 64, 128, 256]
    # for lc in lstm_channels:
    #     hn = NET.HydroNet(input_channel=13, conv_channel=16, lstm_hidden_channel=lc, lstm_layers=2, bidirectional=True)
    #     hn = hn.to(device)
        
    #     rec, max_train_acc, max_test_acc = train(hn, train_loader, test_loader)
    #     print(max_train_acc, max_test_acc)
    #     perf_rec.append([lc, rec, max_train_acc, max_test_acc])
    
    # torch.save(perf_rec, 'performance/lstm_channel.pth')

    # # conv channel test
    # perf_rec = []
    # conv_channels = [16, 32, 64, 128, 256]
    # for cc in conv_channels:
    #     hn = NET.HydroNet(input_channel=13, conv_channel=cc, lstm_hidden_channel=cc, lstm_layers=2, bidirectional=True)
    #     hn = hn.to(device)
        
    #     rec, max_train_acc, max_test_acc = train(hn, train_loader, test_loader)
    #     print(max_train_acc, max_test_acc)
    #     perf_rec.append([cc, rec, max_train_acc, max_test_acc])
    
    # torch.save(perf_rec, 'performance/conv_channel.pth')

    # # lstm layers test
    # perf_rec = []
    # lstm_layers = [1,2,3]
    # for ll in lstm_layers:
    #     hn = NET.HydroNet(input_channel=13, conv_channel=16, lstm_hidden_channel=32, lstm_layers=ll, bidirectional=True)
    #     hn = hn.to(device)
        
    #     rec, max_train_acc, max_test_acc = train(hn, train_loader, test_loader)
    #     print(max_train_acc, max_test_acc)
    #     perf_rec.append([ll, rec, max_train_acc, max_test_acc])
    
    # torch.save(perf_rec, 'performance/lstm_layers.pth')

    net = NET.HydroNetCNNLSTM(input_channel=13, conv_channel=32, conv_filter_sizes=[1,2,3,4], lstm_hidden_channel=64, lstm_layers=2, bidirectional=True)
    net = net.to(device)
    
    rec, max_train_acc, max_test_acc = train(net, train_loader, test_loader)
    print(max_train_acc, max_test_acc)
    torch.save(torch.tensor(rec), 'performance/rec_model=cl_logr=90_loss=msee.pth')