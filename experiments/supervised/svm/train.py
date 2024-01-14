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
from sklearn.metrics import accuracy_score, log_loss
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
    parser.add_argument('--ch_names', default=['F7'], type=List)
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

        # 1. 학습시킬 데이터셋
        print(train_df.shape)

        # def objectiveSVM(trial): # 최적의 하이퍼파라미터 찾을 파라미터 범위 지정
        #     param = {
        #         'C': trial.suggest_float('C', 0.1, 1000),  # 1에서 100 대신 0.1에서 10으로 변경
        #         'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf', 'sigmoid']),  # 'poly' 제거
        #         'shrinking': trial.suggest_categorical('shrinking', [True, False]),
        #         'tol': trial.suggest_float('tol', 1e-4, 1e-2),  # 1e-5에서 1e-1 대신 1e-4에서 1e-2로 변경
        #         'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
        #     }
        #
        #     if param['kernel'] == 'poly':
        #         param['degree'] = trial.suggest_int('degree', 2, 500)  # 1에서 10 대신 2에서 5로 변경
        #
        #     if param['kernel'] in ['poly', 'rbf', 'sigmoid']:
        #         param['gamma'] = trial.suggest_float('gamma', 1e-2, 1000)  # 1e-2에서 100 대신 1e-2에서 10으로 변경
        #         param['coef0'] = trial.suggest_float('coef0', -5.0, 5.0)  # -10.0에서 10.0 대신 -5.0에서 5.0으로 변경
        #
        #     svm = SVC(**param, probability=True)  # 모델 정의
        #     svm.fit(train_df, train_y) # 모델 학습
        #
        #     probas = svm.predict_proba(test_df) # 예측값
        #     loss = log_loss(test_y, probas) # 실제값과 예측값 차이 (loss, 손실, 에러)
        #
        #     return loss
        #
        # sampler = optuna.samplers.TPESampler(seed=424)
        # study = optuna.create_study(
        #     study_name='svm_parameter_opt',
        #     direction='minimize', # 손실을 감소하는 방향으로 학습함
        #     sampler=sampler
        # )
        #
        # study.optimize(objectiveSVM, n_trials=100) # 총 파라미터 찾을 횟수
        #
        # print("Best Score:", study.best_value) # 최고 정확도
        # print("Best trial parameter", study.best_trial.params) # 최고일때 하이퍼 파라미터 전체

        svm = SVC(C=892.0553792189693,
                  kernel='linear',
                  shrinking=True,
                  tol= 0.012405195665974077,
                  class_weight='balanced')
        # svm = SVC(C=960.6643217836736,
        #           kernel='linear',
        #           shrinking=True,
        #           tol= 0.0006847699303460052,
        #           class_weight='balanced')
        svm.fit(train_df, train_y)

        y_pred = svm.predict(test_df)
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

    # def get_related_power(self, data, fs=128):
    #
    #     frequencies, _ = welch(data[0].squeeze(), fs, nperseg=fs * 2)
    #
    #     # 각 주파수 영역의 인덱스 추출
    #     delta_idx = np.logical_and(frequencies >= 0.5, frequencies < 4)
    #     theta_idx = np.logical_and(frequencies >= 4, frequencies < 8)
    #     alpha_idx = np.logical_and(frequencies >= 8, frequencies < 11)
    #     spindle_idx = np.logical_and(frequencies >= 11, frequencies < 16)
    #     beta_idx = np.logical_and(frequencies >= 16, frequencies < 30)
    #
    #     delta_powers = []
    #     theta_powers = []
    #     alpha_powers = []
    #     spindle_powers = []
    #     beta_powers = []
    #
    #     for sample in data:
    #         _, psd = welch(sample.squeeze(), fs, nperseg=fs * 2)
    #
    #         total_power = np.sum(psd[delta_idx]) + np.sum(psd[theta_idx]) + np.sum(psd[alpha_idx]) + np.sum(psd[spindle_idx]) + np.sum(psd[beta_idx])
    #
    #         delta_powers.append(np.sum(psd[delta_idx]) / total_power)
    #         theta_powers.append(np.sum(psd[theta_idx]) / total_power)
    #         alpha_powers.append(np.sum(psd[alpha_idx]) / total_power)
    #         spindle_powers.append(np.sum(psd[spindle_idx]) / total_power)
    #         beta_powers.append(np.sum(psd[beta_idx]) / total_power)
    #
    #     df = pd.DataFrame({'delta' : delta_powers,
    #                         'theta' : theta_powers,
    #                         'alpha' : alpha_powers,
    #                         'spindle' : spindle_powers,
    #                         'beta' : beta_powers})
    #
    #     return df



if __name__ == '__main__':
    trainer = Trainer(get_args())
    trainer.train()
