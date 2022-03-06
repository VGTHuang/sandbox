import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

import consts as CONSTS
import utils as UTILS

class HydroNetLSTM(nn.Module):
    def __init__(self, input_channel=13, seqlen=12, lstm_hidden_channel=64, lstm_layers=2, bidirectional=True):
        super(HydroNetLSTM, self).__init__()
        
        self.input_channel = input_channel
        self.lstm_hidden_channel = lstm_hidden_channel
        self.lstm_layers = lstm_layers
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(input_channel, lstm_hidden_channel, lstm_layers, batch_first=True, bidirectional=bidirectional)
        self.linear1 = nn.Sequential(
            nn.Linear(lstm_hidden_channel * self.num_directions * seqlen, lstm_hidden_channel * self.num_directions),
            nn.Dropout(),
            nn.LeakyReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(lstm_hidden_channel * self.num_directions, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        # input = input.unsqueeze(1)
        # print(input.shape)
        out, _ = self.lstm(input)
        out = out.reshape(out.shape[0], -1)
        out = self.linear1(out)
        out = self.linear2(out).flatten()
        return out

class HydroNetCNNLSTM(nn.Module):
    def __init__(self, input_channel=13, conv_channel=32, conv_filter_sizes=[1,2,3], lstm_hidden_channel=64, lstm_layers=2, bidirectional=True):
        super(HydroNetCNNLSTM, self).__init__()
        
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
            nn.Linear(lstm_hidden_channel * self.num_directions * len(conv_filter_sizes), lstm_hidden_channel * self.num_directions),
            nn.Dropout(),
            nn.LeakyReLU()
        )
        self.linear2 = nn.Sequential(
            nn.Linear(lstm_hidden_channel * self.num_directions, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        input = input.unsqueeze(1)
        outs = []
        for conv in self.convs:
            out = conv(input).squeeze(3).permute(0,2,1)
            out, _ = self.lstm(out)
            out = out[:,-1,:]
            outs.append(out)
        outs = torch.hstack(outs)
        out = self.linear1(outs)
        out = self.linear2(out).flatten()
        return out
    
class HydroNetCNNLSTMAM(nn.Module):
    def __init__(self, input_channel=13, conv_channel=32, conv_filter_sizes=[1,2,3], lstm_hidden_channel=64, lstm_layers=2, bidirectional=True):
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
                    nn.BatchNorm2d(conv_channel),
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
            nn.Linear(lstm_hidden_channel * self.num_directions * lstm_layers, 1),
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
        out = self.linear2(out).flatten()
        return out

class HydroNetLSTMCNN(nn.Module):
    def __init__(self, input_channel=13, lstm_hidden_channel=64, conv_filter_sizes=[1,2,3], lstm_layers=2, bidirectional=True):
        super(HydroNetLSTMCNN, self).__init__()
        
        self.input_channel = input_channel
        self.lstm_hidden_channel = lstm_hidden_channel
        self.lstm_layers = lstm_layers
        self.num_directions = 2 if bidirectional else 1
        
        self.lstm = nn.LSTM(input_channel, lstm_hidden_channel, lstm_layers, batch_first=True, bidirectional=bidirectional)

        # self.convs = nn.ModuleList()
        # for conv_size in conv_filter_sizes:
        #     self.convs.append(
        #         nn.Sequential(
        #             nn.Conv2d(lstm_hidden_channel * self.num_directions, lstm_hidden_channel, (conv_size, input_channel)),
        #             # nn.BatchNorm2d(conv_channel),
        #             nn.LeakyReLU()
        #         )
        #     )
        # self.linear1 = nn.Sequential(
        #     nn.Linear(lstm_hidden_channel * self.num_directions * len(conv_filter_sizes), lstm_hidden_channel * self.num_directions),
        #     nn.Dropout(),
        #     nn.LeakyReLU()
        # )
        # self.linear2 = nn.Sequential(
        #     nn.Linear(lstm_hidden_channel * self.num_directions, 1),
        #     nn.Sigmoid()
        # )
    
    def attention_net(self, lstm_output, final_state):
        hidden = final_state.permute(1,0,2).reshape(-1, lstm_output.shape[2], self.lstm_layers)
        print(lstm_output.shape, hidden.shape)
        # hidden = final_state.view(-1, lstm_output.shape[2] * self.num_directions, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        # print(lstm_output.shape, hidden.shape)
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
        print(attn_weights.shape)
        soft_attn_weights = F.softmax(attn_weights, 1)
        print(soft_attn_weights.shape)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context

    def forward(self, input):
        lstm_out, (hn, _) = self.lstm(input)

        context = self.attention_net(lstm_out, hn)
        # input = input.unsqueeze(1)
        # outs = []
        # for conv in self.convs:
        #     out = conv(input).squeeze(3).permute(0,2,1)
        #     h0 = torch.zeros(self.lstm_layers * self.num_directions, out.shape[0], self.lstm_hidden_channel).to(input.device)
        #     c0 = torch.zeros(self.lstm_layers * self.num_directions, out.shape[0], self.lstm_hidden_channel).to(input.device)
        #     out, _ = self.lstm(out, (h0, c0))
        #     out = out[:,-1,:]
        #     outs.append(out)
        # outs = torch.hstack(outs)
        # out = self.linear1(outs)
        # out = self.linear2(out).flatten()
        return context

class MixMSELoss(nn.MSELoss):
    '''
    return 3 losses:
    1. mse loss with logarithmic values as input;
    2. mse loss with linear values as input;
    3. weighted mse loss
    '''
    def __init__(self, logarithmic_ratio = 0.5):
        super(MixMSELoss, self).__init__()
        self.logarithmic_ratio = logarithmic_ratio
    
    def forward(self, a, b):
        # denormalize
        logarithmic_mse = super(MixMSELoss, self).forward(
            UTILS.normalize_log_output(a, from_original_data=False),
            UTILS.normalize_log_output(b, from_original_data=False)
        )
        linear_mse = super(MixMSELoss, self).forward( a, b )
        return self.logarithmic_ratio * logarithmic_mse + (1-self.logarithmic_ratio) * linear_mse

class MixMSEEnhanceLoss(nn.MSELoss):
    '''
    return 3 losses:
    1. mse loss with logarithmic values as input;
    2. mse loss with linear values as input;
    3. weighted mse loss
    '''
    def __init__(self, logarithmic_ratio = 0.5):
        super(MixMSEEnhanceLoss, self).__init__()
        self.logarithmic_ratio = logarithmic_ratio
    
    def forward(self, a, b):
        # denormalize
        logarithmic_mse = (UTILS.normalize_log_output(a, from_original_data=False)
            - UTILS.normalize_log_output(b, from_original_data=False)) ** 2
        logarithmic_mse *= ((b+1) ** 4)
        logarithmic_mse = torch.mean(logarithmic_mse)
        # logarithmic_mse = super(MixMSEEnhanceLoss, self).forward(
        #     UTILS.normalize_log_output(a, from_original_data=False),
        #     UTILS.normalize_log_output(b, from_original_data=False)
        # )
        linear_mse = super(MixMSEEnhanceLoss, self).forward( a, b )
        
        return (self.logarithmic_ratio * logarithmic_mse + (1-self.logarithmic_ratio) * linear_mse)

class MixHuberLoss(nn.HuberLoss):
    '''
    return 3 losses:
    1. mse loss with logarithmic values as input;
    2. mse loss with linear values as input;
    3. weighted mse loss
    '''
    def __init__(self, logarithmic_ratio = 0.5):
        super(MixHuberLoss, self).__init__()
        self.logarithmic_ratio = logarithmic_ratio
    
    def forward(self, a, b):
        # denormalize
        logarithmic_mse = super(MixHuberLoss, self).forward(
            UTILS.normalize_log_output(a, from_original_data=False),
            UTILS.normalize_log_output(b, from_original_data=False)
        )
        linear_mse = super(MixHuberLoss, self).forward( a, b )
        return self.logarithmic_ratio * logarithmic_mse + (1-self.logarithmic_ratio) * linear_mse

class L1Accuracy(nn.L1Loss):
    def __init__(self):
        super(L1Accuracy, self).__init__()

    def forward(self, a, b):
        out = super(L1Accuracy, self).forward( a, b )
        return 1 - out


if __name__ == '__main__':
    # hn = HydroNetCNNLSTMAM(bidirectional=True)
    hn = HydroNetLSTM(bidirectional=True)


    # batch = 4, sequence length = 12, parameters = 13
    test_input = torch.rand(5, 12, 13)
    test_output = hn(test_input)

    print(test_output.shape)

    # mse_loss = MixMSELoss()
    # hub_loss = MixHuberLoss()
    # mae_acc = L1Accuracy()

    # print(mse_loss(
    #     torch.tensor([0.1, 0.8, 0.5]),
    #     torch.tensor([0.3, 0.8, 0.5])
    # ))
    # print(hub_loss(
    #     torch.tensor([0.1, 0.8, 0.5]),
    #     torch.tensor([0.3, 0.8, 0.5])
    # ))
    # print(mae_acc(
    #     torch.tensor([0.1, 0.8, 0.5]),
    #     torch.tensor([0.2, 0.7, 0.6])
    # ))