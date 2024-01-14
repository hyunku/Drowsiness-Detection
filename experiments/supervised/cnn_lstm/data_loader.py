# -*- coding:utf-8 -*-
import torch
from torch.utils.data import Dataset


class TorchDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float).unsqueeze(dim=1)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
