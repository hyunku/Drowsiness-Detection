# -*- coding:utf-8 -*-
import os
import copy
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from typing import List
import torch.optim as opt
from experiments.ssl.mae.model import MaskedAutoEncoderViT
from experiments.ssl.mae.model import EncoderWrapper
from sklearn.metrics import accuracy_score, f1_score
from experiments.transfer_learning.data_loader import TorchDataset
from torch.utils.data import DataLoader
from models.backbone import CNNEncoder2D
from dataset.data_parser import DrowsinessDataset

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    # Load Checkpoint
    # parser.add_argument('--ssl_ckpt_path', default=os.path.join('..', '..', 'ckpt', 'ssl',
    #                                                             'barlow_twins', '0001.pth'))
    parser.add_argument('--ssl_ckpt_path', default=os.path.join('..', '..', 'ckpt', 'ssl',
                                                                'mae', '0001.pth'))

    # Dataset
    parser.add_argument('--base_path', default=os.path.join('..', '..', '..', '..', '..',
                                                            'Database', 'Drowsiness-Dataset', 'dataset.mat'))
    # Train
    parser.add_argument('--subject_idx', default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
    parser.add_argument('--ch_names', default=['F7'], type=List)
    parser.add_argument('--train_epochs', default=300, type=int)
    parser.add_argument('--train_lr_rate', default=0.1, type=float)
    parser.add_argument('--train_batch_size', default=32, type=int)
    return parser.parse_args()


class Model(nn.Module):
    def __init__(self, backbone: nn.Module, frozen_layers: List, classes=2):
        super().__init__()
        self.backbone = self.get_backbone(backbone=backbone, frozen_layers=frozen_layers)
        self.classes = classes
        self.fc = nn.Linear(backbone.final_length, self.classes)

    @staticmethod
    def get_backbone(backbone: nn.Module, frozen_layers: List):
        backbone = copy.deepcopy(backbone)
        for name, module in backbone.named_modules():
            if name in frozen_layers:
                for param in module.parameters():
                    param.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = True
        return backbone

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        return x


class Trainer(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.backbone = self.load_backbone()
        self.stft_parameter, self.input_size = self.input_parameter()
        self.backbone_frozen_layers = [name for name, _ in self.backbone.named_modules()]
        self.model = Model(backbone=self.backbone, frozen_layers=self.backbone_frozen_layers, classes=2).to(device)
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.args.train_lr_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        (train_x, train_y), (test_x, test_y) = self.load_dataset()
        train_dataset = TorchDataset(x=train_x, y=train_y, stft_parameter=self.stft_parameter,
                                     resize=self.input_size)
        test_dataset = TorchDataset(x=test_x, y=test_y, stft_parameter=self.stft_parameter,
                                     resize=self.input_size)
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=self.args.train_batch_size, shuffle=True)
        test_x, test_y = torch.tensor(test_dataset.x, dtype=torch.float32).to(device).unsqueeze(dim=1), \
                         torch.tensor(test_dataset.y, dtype=torch.long).to(device)

        scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.args.train_epochs):
            self.model.train()
            epoch_loss = []
            for batch in train_dataloader:
                self.optimizer.zero_grad()
                x, y = batch
                x, y = x.to(device).unsqueeze(dim=1), y.to(device)

                with torch.autocast(device_type='cuda', dtype=torch.float32):
                    o = self.model(x)
                    loss = self.criterion(o, y)

                epoch_loss.append(loss.detach().cpu().item())
                scaler.scale(loss).backward()
                scaler.step(optimizer=self.optimizer)
                scaler.update()

            self.model.eval()
            test_o = self.model(test_x)
            test_o = torch.argmax(test_o, dim=-1)
            pred, real = test_o.detach().cpu().numpy(), test_y.detach().cpu().numpy()
            acc, mf1 = accuracy_score(y_pred=pred, y_true=real), f1_score(y_pred=pred, y_true=real, average='macro')
            print('[Epoch] : {0:03d} \t [Loss]: {1:.4f} \t [Accuracy] : {2:.4f} \t [Macro-F1] : {3:.4f}'.format(
                epoch+1, np.mean(epoch_loss), acc, mf1
            ))

    def load_backbone(self):
        ckpt = torch.load(self.args.ssl_ckpt_path, map_location='cpu')
        backbone_parameter = ckpt['backbone_parameter']
        model = MaskedAutoEncoderViT(**backbone_parameter)
        model.load_state_dict(ckpt['model_state'])
        encoder = EncoderWrapper(
            input_size=ckpt['input_size'],
            patch_embed=model.patch_embed,
            encoder_block=model.encoder_block,
            embed_dim=backbone_parameter['embed_dim'],
            cls_token=model.cls_token,
            pos_embed=model.pos_embed,
            device=device
        ).to(device)
        del model
        return encoder

    def input_parameter(self):
        ckpt = torch.load(self.args.ssl_ckpt_path, map_location='cpu')
        stft_parameter = ckpt['short_time_fourier_transform_parameter']
        input_size = ckpt['input_size']
        return stft_parameter, input_size

    def load_dataset(self):
        dataset = DrowsinessDataset(path=self.args.base_path)
        # Train Dataset
        train_idx = [i for i in range(dataset.total_subject) if i != self.args.subject_idx]
        train_x, train_y = dataset.get_items(subject_ids=train_idx,
                                             select_ch_names=self.args.ch_names)
        # Test Dataset
        test_idx = [self.args.subject_idx]
        test_x, test_y = dataset.get_items(subject_ids=test_idx,
                                           select_ch_names=self.args.ch_names)
        return (train_x, train_y), (test_x, test_y)


if __name__ == '__main__':
    trainer = Trainer(get_args())
    trainer.train()
