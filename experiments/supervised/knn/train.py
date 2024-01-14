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
from scipy.signal import welch
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
import optuna


random_seed = 424
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_args():
    subject_idx = 10
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--base_path', default=os.path.join('..', '..', '..', '..', 'dataset', 'Drowsiness-Dataset', 'dataset.mat'))
    # Model Parameter
    parser.add_argument('--sampling_rate', default=128, type=int)
    parser.add_argument('--classes', default=2, type=int)

    # Train
    parser.add_argument('--subject_idx', default=subject_idx, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], type=int)
    parser.add_argument('--ch_names', default=['F7'], type=List) # F7
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
        (train_x, train_y), (test_x, test_y) = self.load_dataset()

        train_df = self.get_related_power(train_x, fs=128)
        test_df = self.get_related_power(test_x, fs=128)
        print(train_df.shape, train_y.shape)

        # def objective(trial):
        #     n_neighbors = trial.suggest_int('n_neighbors', 1, 30)
        #     weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
        #     p = trial.suggest_int('p', 1, 5)
        #
        #     knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
        #     knn.fit(train_df, train_y)
        #
        #     preds = knn.predict(test_df)
        #     return accuracy_score(test_y, preds)
        #
        # study = optuna.create_study(direction='maximize')
        # study.optimize(objective, n_trials=100)
        #
        # best_params = study.best_params
        # best_score = study.best_value
        #
        # print(f"Best parameters: {best_params}")
        # print(f"Best validation score: {best_score}")


        knn = KNeighborsClassifier(n_neighbors=16, weights='distance', p=2)
        knn.fit(train_df, train_y)

        y_pred = knn.predict(test_df)
        accuracy = accuracy_score(y_pred, test_y)
        print(f"Accuracy: {accuracy:.4f}")

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

    def get_related_power(self, data, fs=128):

        frequencies, _ = welch(data[0].squeeze(), fs, nperseg=fs * 2)

        # 각 주파수 영역의 인덱스 추출
        delta_idx = np.logical_and(frequencies >= 0.5, frequencies < 4)
        theta_idx = np.logical_and(frequencies >= 4, frequencies < 8)
        alpha_idx = np.logical_and(frequencies >= 8, frequencies < 12)
        beta_idx = np.logical_and(frequencies >= 12, frequencies < 30)

        delta_powers = []
        theta_powers = []
        alpha_powers = []
        beta_powers = []

        for sample in data:
            _, psd = welch(sample.squeeze(), fs, nperseg=fs * 2)

            total_power = np.sum(psd[delta_idx]) + np.sum(psd[theta_idx]) + np.sum(psd[alpha_idx]) + np.sum(psd[beta_idx])

            delta_powers.append(np.sum(psd[delta_idx]) / total_power)
            theta_powers.append(np.sum(psd[theta_idx]) / total_power)
            alpha_powers.append(np.sum(psd[alpha_idx]) / total_power)
            beta_powers.append(np.sum(psd[beta_idx]) / total_power)

        df = pd.DataFrame({'delta' : delta_powers,
                            'theta' : theta_powers,
                            'alpha' : alpha_powers,
                            'beta' : beta_powers})

        return df


if __name__ == '__main__':
    trainer = Trainer(get_args())
    trainer.train()
