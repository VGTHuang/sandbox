'''
网络结构
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

import consts as CONSTS
import utils as UTILS


class HydroNetDense(nn.Module):
    '''
    简单全连接网络
    '''
    def __init__(self, input_channel=13, seqlen=12, output_channel=7, hidden_channels=64, hidden_layers=3):
        '''
        :param input_channel: 输入通道数
        :param seqlen: 序列长度
        :param output_channel: 输出通道数
        :param hidden_channels: 中间层通道数 (64, 128, 256, ...)
        :param hidden_layers: 中间层数 (1, 2, 3, ...)
        '''
        super(HydroNetDense, self).__init__()
        
        # 每层，除最后一层，都为 linear + elu
        self.linear_in = nn.Sequential(
            nn.Linear(input_channel * seqlen, hidden_channels),
            nn.ELU()
        )
        self.hidden = nn.Sequential()
        for i in range(hidden_layers - 1):
            self.hidden.add_module(f'hidden{i}', nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ELU()
            )
        )
        self.linear_out = nn.Sequential(
            nn.Linear(hidden_channels, output_channel),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        input = input.reshape(input.shape[0], -1)
        out = self.linear_in(input)
        out = self.hidden(out)
        out = self.linear_out(out)
        return out


class HydroNetCNN(nn.Module):
    '''
    简单卷积网络
    '''
    def __init__(self, input_channel=13, output_channel=7, seqlen=30, cnn_channels=[16, 32, 64]):
        '''
        :param input_channel: 输入通道数
        :param seqlen: 序列长度
        :param output_channel: 输出通道数
        :param cnn_channels: 中间层通道数 (一般设为逐层增加 如[32,64,128,256] 数组中有几个数，即为几层中间层)
        '''
        super(HydroNetCNN, self).__init__()
        
        self.input_channel = input_channel
        
        self.convs = nn.ModuleList()
        cnn_in_channels = [input_channel] + cnn_channels
        cnn_in_channels.pop(-1)
        for in_conv_size, out_conv_size in zip(cnn_in_channels, cnn_channels):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_conv_size, out_conv_size, kernel_size=3, padding=1, padding_mode='replicate'),
                    # nn.BatchNorm1d(out_conv_size),
                    nn.LeakyReLU(),
                    nn.Conv1d(out_conv_size, out_conv_size, kernel_size=3, padding=1, padding_mode='replicate', groups=4),
                    # nn.BatchNorm1d(out_conv_size),
                    nn.LeakyReLU(),
                )
            )
        # 最后全连接
        self.linear = nn.Sequential(
            nn.Linear(cnn_channels[-1] * seqlen, output_channel),
            nn.Dropout(0.1),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        out = input.permute(0,2,1)
        for conv in self.convs:
            out = conv(out)
        out = out.reshape(out.shape[0], -1)
        out = self.linear(out)
        return out


class HydroNetLSTM(nn.Module):
    '''
    简单LSTM网络
    '''
    def __init__(self, input_channel=13, output_channel=7, lstm_hidden_channel=64, lstm_layers=2, bidirectional=True):
        '''
        :param input_channel: 输入通道数
        :param output_channel: 输出通道数
        :param lstm_hidden_channel: lstm层通道数
        :param lstm_layers: lstm层数
        :param bidirectional: 是否为双向lstm
        '''
        super(HydroNetLSTM, self).__init__()
        
        self.input_channel = input_channel
        self.lstm_hidden_channel = lstm_hidden_channel
        self.lstm_layers = lstm_layers
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(input_channel, lstm_hidden_channel, lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_channel * self.num_directions, output_channel),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        out, _ = self.lstm(input)
        out = out[:,-1,:]
        out = self.linear(out)
        return out


class HydroNetCNNLSTM(nn.Module):
    '''
    先CNN后LSTM
    '''
    def __init__(self, input_channel=13, output_channel=7, cnn_channels=[16, 32, 64], lstm_hidden_channel=64, lstm_layers=2, bidirectional=True):
        super(HydroNetCNNLSTM, self).__init__()
        
        self.input_channel = input_channel
        self.lstm_hidden_channel = lstm_hidden_channel
        self.lstm_layers = lstm_layers
        self.num_directions = 2 if bidirectional else 1
        
        self.convs = nn.ModuleList()
        cnn_in_channels = [input_channel] + cnn_channels
        cnn_in_channels.pop(-1)
        for in_conv_size, out_conv_size in zip(cnn_in_channels, cnn_channels):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(in_conv_size, out_conv_size, kernel_size=1),
                    nn.ELU(),
                    nn.Conv1d(out_conv_size, out_conv_size, kernel_size=3, padding=1, padding_mode='replicate', groups=4),
                    nn.ELU(),
                    nn.Conv1d(out_conv_size, out_conv_size, kernel_size=1),
                    # nn.InstanceNorm1d(),
                    nn.ELU(),
                    # nn.Conv1d(out_conv_size, out_conv_size, kernel_size=1),
                    # nn.BatchNorm1d(out_conv_size),
                    # nn.InstanceNorm1d(out_conv_size),
                    # nn.AvgPool1d(2),
                    # nn.ELU()
                )
            )
        self.lstm = nn.LSTM(cnn_channels[-1], lstm_hidden_channel, lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_channel * self.num_directions, output_channel),
            nn.Dropout(0.1),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        out = input.permute(0,2,1)
        for conv in self.convs:
            out = conv(out)
        out = out.permute(0,2,1)
        out, _ = self.lstm(out)
        out = out[:,-1,:]
        out = self.linear(out)
        return out

class HydroNetLSTMAM(nn.Module):
    '''
    先LSTM后注意力
    '''
    def __init__(self, input_channel=13, output_channel=7, lstm_hidden_channel=32, lstm_layers=2, bidirectional=True):
        super(HydroNetLSTMAM, self).__init__()
        
        self.input_channel = input_channel
        self.lstm_hidden_channel = lstm_hidden_channel
        self.lstm_layers = lstm_layers
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(input_channel, lstm_hidden_channel, lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.linear = nn.Sequential(
            nn.Linear(lstm_hidden_channel * self.num_directions * self.lstm_layers, output_channel),
            nn.Sigmoid()
        )

    def attention_net(self, output, hn):
        '''
        注意力模块
        https://blog.csdn.net/qq_52785473/article/details/122852099
        '''
        hidden = hn.view(self.lstm_layers,-1,self.lstm_hidden_channel*self.num_directions).permute(1,0,2).transpose(1,2)
        hidden = torch.tanh(hidden)
        attn_weights = torch.bmm(output, hidden)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(output.transpose(1, 2), soft_attn_weights) # [batch_size, lstm_hidden_channel * self.num_directions, lstm_layers]
        return context

    def forward(self, input):
        out, (hn, _) = self.lstm(input)
        am_out = self.attention_net(out, hn)
        am_out = am_out.view(-1, self.lstm_hidden_channel * self.num_directions * self.lstm_layers)
        
        ln_out = self.linear(am_out)
        return ln_out

class HydroNetCNNLSTMAM(HydroNetCNNLSTM):
    '''
    
    CNN+LSTM+注意力
    '''
    def __init__(self, input_channel=13, output_channel=7, cnn_channels=[16, 32, 64], lstm_hidden_channel=64, lstm_layers=2, bidirectional=True):
        super(HydroNetCNNLSTMAM, self).__init__(input_channel, output_channel, cnn_channels, lstm_hidden_channel, lstm_layers, bidirectional)
        
    def attention_net(self, output, hn):
        hidden = hn.view(self.lstm_layers,-1,self.lstm_hidden_channel*self.num_directions).permute(1,0,2).transpose(1,2)
        hidden = torch.tanh(hidden)
        attn_weights = torch.bmm(output, hidden)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(output.transpose(1, 2), soft_attn_weights) # [batch_size, lstm_hidden_channel * self.num_directions, lstm_layers]
        return context

    def forward(self, input):
        
        out = input.permute(0,2,1)
        # out = input[:,None,:,:]
        for conv in self.convs:
            out = conv(out)
        out = out.permute(0,2,1)
        out, (hn, _) = self.lstm(out)
        out = self.attention_net(out, hn)
        out = out[:,:,-1]
        out = self.linear(out)
        return out

# 几种accuracy和loss计算

class L1Accuracy(nn.L1Loss):
    '''
    用``nn.L1Loss``计算``L1Acc``
    '''
    def __init__(self):
        super(L1Accuracy, self).__init__()

    def forward(self, a, b):
        out = super(L1Accuracy, self).forward( a, b )
        return 1 - out

def PearsonR(yhat, y):
    ''' pearson相关系数 '''
    r, _ = pearsonr(yhat, y)
    return r

def RMSE(yhat, y):
    ''' RMSE '''
    return torch.sqrt(torch.mean((yhat - y) ** 2))

def MAPE(yhat, y):
    ''' MAPE '''
    return torch.mean(torch.abs((yhat - y) / y))

def NSE(yhat, y):
    ''' NSE '''
    return (torch.sum((yhat - y)**2) / torch.sum((y - torch.mean(y)) ** 2))

if __name__ == '__main__':
    net = HydroNetCNNLSTMAM(
        input_channel=13,
        output_channel=7,
        cnn_channels=[32, 64, 128])
    input = torch.rand(5, 30, 13)
    print(net(input).shape)
    # l = MSEEnhanceLoss()
    # pred = torch.tensor([0.1,0.1,0.8])
    # targ = torch.tensor([0.1,0.1,0.9])
    # print(l(pred, targ))