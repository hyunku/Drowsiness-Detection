# -*- coding:utf-8 -*-
import torch
import numpy as np
from typing import Dict, Tuple
from torch.utils.data import Dataset
from dataset.utils import stft_db
from torchvision.transforms import Resize


class TorchDataset(Dataset):
    def __init__(self, x, y, stft_parameter: Dict, resize: Tuple):
        self.x, self.y = x, y
        self.resize_inst = Resize(size=resize, antialias=True)
        self.x = np.array([stft_db(x_,
                                   sf=stft_parameter['sfreq'],
                                   window=stft_parameter['window'],
                                   step=stft_parameter['step'],
                                   band=stft_parameter['bands'])
                           for x_ in list(self.x.squeeze())])
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.x = self.resize_inst(self.x).detach().numpy()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
