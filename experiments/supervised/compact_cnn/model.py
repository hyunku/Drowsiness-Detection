# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class CompactCNN(nn.Module):
    def __init__(self, kernel_size=32, kernel_length=64, sample_length=384, classes=2):
        super(CompactCNN, self).__init__()
        self.kernel_length = kernel_length

        self.conv = nn.Conv2d(1, kernel_size, (1, kernel_length))
        self.batch = BatchLayer(kernel_size)
        self.GAP = nn.AvgPool2d((1, sample_length - kernel_length + 1))
        self.fc = nn.Linear(kernel_size, classes)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        intermediate = self.conv(x)
        intermediate = self.batch(intermediate)
        intermediate = nn.ELU()(intermediate)
        intermediate = self.GAP(intermediate)
        intermediate = intermediate.view(intermediate.size()[0], -1)
        intermediate = self.fc(intermediate)
        output = self.softmax(intermediate)
        return output


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
