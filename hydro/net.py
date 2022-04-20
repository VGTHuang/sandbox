import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr

import matplotlib.pyplot as plt

import consts as CONSTS
import utils as UTILS


class HydroNetDense(nn.Module):
    def __init__(self, input_channel=13, seqlen=12, output_channel=7, hidden_channels=64, hidden_layers=3):
        super(HydroNetDense, self).__init__()
        
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
    def __init__(self, input_channel=13, output_channel=7, seqlen=30, cnn_channels=[16, 32, 64]):
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
                    # nn.Conv1d(out_conv_size, out_conv_size, kernel_size=1),
                    # nn.BatchNorm1d(out_conv_size),
                    # nn.LeakyReLU(),
                    # nn.Conv1d(out_conv_size, out_conv_size, kernel_size=1),
                    # nn.BatchNorm1d(out_conv_size),
                    # nn.InstanceNorm1d(out_conv_size),
                    # nn.ELU()
                )
            )
        self.linear = nn.Sequential(
            nn.Linear(cnn_channels[-1] * seqlen, output_channel),
            nn.Dropout(0.1),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        out = input.permute(0,2,1)
        # out = input[:,None,:,:]
        for conv in self.convs:
            out = conv(out)
        out = out.reshape(out.shape[0], -1)
        out = self.linear(out)
        return out


class HydroNetLSTM(nn.Module):
    def __init__(self, input_channel=13, output_channel=7, lstm_hidden_channel=64, lstm_layers=2, bidirectional=True):
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
        # input = input.unsqueeze(1)
        out, _ = self.lstm(input)
        out = out[:,-1,:]
        # out = out.reshape(out.shape[0], -1)
        out = self.linear(out)
        return out


class HydroNetCNNLSTM(nn.Module):
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
        # out = input[:,None,:,:]
        for conv in self.convs:
            out = conv(out)
        out = out.permute(0,2,1)
        out, _ = self.lstm(out)
        out = out[:,-1,:]
        # out = out.reshape(out.shape[0], -1)
        out = self.linear(out)
        return out

class HydroNetLSTMAM(nn.Module):
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
        # hidden = hn.permute(1,0,2).reshape(-1,self.lstm_hidden_channel*self.num_directions,self.lstm_layers)
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

class HydroNetCNNLSTMAM(nn.Module):
    def __init__(self, input_channel=13, output_channel=7, conv_channel=32, conv_filter_sizes=[1,2,3], lstm_hidden_channel=64, lstm_layers=2, bidirectional=True):
        super(HydroNetCNNLSTMAM, self).__init__()
        
        self.input_channel = input_channel
        self.conv_channel = conv_channel
        self.lstm_hidden_channel = lstm_hidden_channel
        self.lstm_layers = lstm_layers
        self.num_directions = 2 if bidirectional else 1
        
        self.convs = nn.ModuleList()
        for conv_size in conv_filter_sizes:
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(1, conv_channel, (conv_size, input_channel)),
                    # nn.BatchNorm2d(conv_channel),
                    nn.LeakyReLU()
                )
            )
        self.lstm = nn.LSTM(conv_channel, lstm_hidden_channel, lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.linear1 = nn.Sequential(
            nn.Linear(lstm_hidden_channel * self.num_directions * lstm_layers * len(conv_filter_sizes), lstm_hidden_channel * self.num_directions * lstm_layers),
            nn.Dropout(),
            nn.LeakyReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(lstm_hidden_channel * self.num_directions * lstm_layers, output_channel),
            nn.Sigmoid()
        )

    def attention_net(self, output, hn):
        # hidden = hn.permute(1,0,2).reshape(-1,self.lstm_hidden_channel*self.num_directions,self.lstm_layers)
        hidden = hn.view(self.lstm_layers,-1,self.lstm_hidden_channel*self.num_directions).permute(1,0,2).transpose(1,2)
        hidden = torch.tanh(hidden)
        attn_weights = torch.bmm(output, hidden)
        soft_attn_weights = F.softmax(attn_weights, 1)
        context = torch.bmm(output.transpose(1, 2), soft_attn_weights) # [batch_size, lstm_hidden_channel * self.num_directions, lstm_layers]
        return context

    def forward(self, input):
        input = input.unsqueeze(1)
        outs = []
        for conv in self.convs:
            out = conv(input).squeeze(3).permute(0,2,1)
            out, (hn, _) = self.lstm(out)
            am_out = self.attention_net(out, hn)
            am_out = am_out.view(-1, self.lstm_hidden_channel * self.num_directions * self.lstm_layers)
            outs.append(am_out)
        outs = torch.hstack(outs)
        out = self.linear1(outs)
        out = self.linear2(out)
        return out

class L1Accuracy(nn.L1Loss):
    def __init__(self):
        super(L1Accuracy, self).__init__()

    def forward(self, a, b):
        out = super(L1Accuracy, self).forward( a, b )
        return 1 - out

class MSEEnhanceLoss(nn.Module):
    def __init__(self):
        super(MSEEnhanceLoss, self).__init__()
    
    def enhance_data(self, d):
        return torch.sqrt(d)
    
    def forward(self, pred, target):
        mask = target * 10 + 1
        mse = torch.pow(pred - target, 2) * mask
        # enhance_mse = torch.pow(self.enhance_data(pred) - self.enhance_data(target), 2) * (torch.floor(target*4)*5+1)
        # enhance_mse *= 0.1
        return torch.mean(mse) # + torch.mean(enhance_mse)

def PearsonR(yhat, y):
    r, _ = pearsonr(yhat, y)
    return r

def RMSE(yhat, y):
    return torch.sqrt(torch.mean((yhat - y) ** 2))

def MAPE(yhat, y):
    return torch.mean(torch.abs((yhat - y) / y))

def NSE(yhat, y):
    return (torch.sum((yhat - y)**2) / torch.sum((y - torch.mean(y)) ** 2))

if __name__ == '__main__':
    net = HydroNetCNN(
        input_channel=13,
        output_channel=14,
        seqlen = 30,
        cnn_channels=[32, 64, 128])
    input = torch.rand(5, 30, 13)
    print(net(input).shape)
    # l = MSEEnhanceLoss()
    # pred = torch.tensor([0.1,0.1,0.8])
    # targ = torch.tensor([0.1,0.1,0.9])
    # print(l(pred, targ))