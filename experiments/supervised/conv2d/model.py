# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(self, kernel_size=32, kernel_length=64, classes=2):
        super(CNN_LSTM, self).__init__()
        self.kernel_length = kernel_length
        self.padding = nn.ReplicationPad2d((31, 32, 0, 0))

        self.conv = nn.Conv2d(1, kernel_size, (1, kernel_length))
        self.batch = BatchLayer(kernel_size)
        self.pool = nn.AvgPool2d((1, 8))
        self.softmax = nn.LogSoftmax(dim=1)
        self.lstm = nn.LSTM(32, classes)

    def forward(self, x):
        source = self.padding(x)
        source = self.conv(source)
        source = self.batch(source)

        source = nn.ELU()(source)
        source = self.pool(source)
        source = source.squeeze()
        source = source.permute(2, 0, 1)
        source = self.lstm(source)
        source = source[1][0].squeeze()
        source = self.softmax(source)
        return source


def normalize_layer(data):
    eps = 1e-05
    a_mean = data - torch.mean(data, [0, 2, 3], True).expand(int(data.size(0)), int(data.size(1)), int(data.size(2)),
                                                             int(data.size(3)))
    b = torch.div(a_mean, torch.sqrt(torch.mean((a_mean) ** 2, [0, 2, 3], True) + eps).expand(int(data.size(0)),
                                                                                              int(data.size(1)),
                                                                                              int(data.size(2)),
                                                                                              int(data.size(3))))
    return b


class BatchLayer(torch.nn.Module):
    def __init__(self, dim):
        super(BatchLayer, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(1, dim, 1, 1))
        self.gamma.data.uniform_(-0.1, 0.1)
        self.beta.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        data = normalize_layer(x)
        gamma_matrix = self.gamma.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)), int(data.size(3)))
        beta_matrix = self.beta.expand(int(data.size(0)), int(data.size(1)), int(data.size(2)), int(data.size(3)))
        return data * gamma_matrix + beta_matrix
