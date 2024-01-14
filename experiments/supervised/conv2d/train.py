# -*- coding:utf-8 -*-
import os
import torch
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as opt
from typing import List, Tuple
from experiments.supervised.conv2d.model import CNN_LSTM
from experiments.supervised.conv2d.data_loader import TorchDataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from dataset.data_parser import DrowsinessDataset

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


def get_args(subject_idx):
    subject_idx = subject_idx
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--base_path', default=os.path.join('..', '..', '..', '..', '..', '..',
                                                            'Database', 'Drowsiness-Dataset', 'dataset.mat'))
    # Model Parameter
    parser.add_argument('--sampling_rate', default=128, type=int)
    parser.add_argument('--classes', default=2, type=int)

    # Train
    parser.add_argument('--subject_idx', default=subject_idx, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
    parser.add_argument('--ch_names', default=['Fp1'], type=List)
    parser.add_argument('--train_epochs', default=200, type=int)
    parser.add_argument('--train_lr_rate', default=0.0005, type=float)
    parser.add_argument('--train_batch_size', default=32, type=int)

    parser.add_argument('--ckpt_path', default=os.path.join('..', '..', '..', 'ckpt', 'supervised', 'conv2d'))
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = CNN_LSTM().to(device)
        self.optimizer = opt.Adam(self.model.parameters(), lr=self.args.train_lr_rate)
        self.criterion = nn.CrossEntropyLoss()

    def train(self):
        log_df = {'epoch': [], 'loss': [], 'accuracy': [], 'macro-f1': []}
        (train_x, train_y), (test_x, test_y) = self.load_dataset()
        train_dataset = TorchDataset(x=train_x, y=train_y)
        test_dataset = TorchDataset(x=test_x, y=test_y)
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
            epoch_mean_loss = np.mean(epoch_loss)
            pred, real = test_o.detach().cpu().numpy(), test_y.detach().cpu().numpy()
            acc, mf1 = accuracy_score(y_pred=pred, y_true=real), f1_score(y_pred=pred, y_true=real, average='macro')
            print('[Epoch] : {0:03d} \t [Loss]: {1:.4f} \t [Accuracy] : {2:.4f} \t [Macro-F1] : {3:.4f}'.format(
                epoch+1, epoch_mean_loss, acc, mf1
            ))

            log_df['epoch'].append(epoch+1)
            log_df['loss'].append(epoch_mean_loss)
            log_df['accuracy'].append(acc)
            log_df['macro-f1'].append(mf1)
            self.save_ckpt(epoch, pred=pred, real=real)

        log_df = pd.DataFrame(log_df)
        log_df.to_csv(os.path.join(self.args.ckpt_path,
                                   'subject_{0:02d}'.format(self.args.subject_idx+1),
                                   'result.csv'),
                      index=False)

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

    def save_ckpt(self, epoch, pred, real):
        base_ckpt = os.path.join(self.args.ckpt_path, 'subject_{0:02d}'.format(self.args.subject_idx+1), 'model')
        if not os.path.exists(base_ckpt):
            os.makedirs(base_ckpt)

        # ckpt_path = os.path.join(base_ckpt, '{0:04d}.pth'.format(epoch+1))
        # torch.save({
        #     'model_name': 'Conv2D',
        #     'model_parameter': {
        #         'sampling_rate': self.args.sampling_rate, 'input_size': self.args.resize
        #     },
        #     'short_time_fourier_transform_parameter': {
        #         'sf': self.args.sfreq, 'window': self.args.window, 'step': self.args.step,
        #         'bands': self.args.bands,
        #     },
        #     'input_size': self.args.resize,
        #     'model_state': self.model.state_dict(),
        #     'result': {'pred': pred, 'real': real},
        #     'hyperparameter': self.args.__dict__,
        # }, ckpt_path)


if __name__ == '__main__':
    for i_ in range(11):
        trainer = Trainer(get_args(i_))
        trainer.train()