# -*- coding:utf-8 -*-
import os
import abc
import mne
import numpy as np
import scipy.io as sio
from typing import List
import warnings

mne.set_log_level(False)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

second = 3
rfreq = 128


# 1. Classification Dataset
class DrowsinessDataset(object):
    def __init__(self, path: str):
        super().__init__()
        self.total_subject = 11
        self.data = self.load_mat(path)
        self.info = mne.create_info(
            ch_names=['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FT7', 'FC3', 'FCZ', 'FC4', 'FT8', 'T3',
                      'C3', 'Cz', 'C4', 'T4', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'T5', 'P3', 'PZ', 'P4', 'T6',
                      'O1', 'Oz', 'O2'],
            sfreq=128,
            ch_types='eeg'
        )
        self.l_freq, self.h_freq = 1, 50
        self.rfreq = rfreq

    def load_mat(self, path):
        mat = sio.loadmat(path)
        data, sub_index, sub_state = mat['EEGsample'], mat['subindex'], mat['substate']
        sub_state, sub_index = np.squeeze(sub_state), np.squeeze(sub_index) - 1

        temp = {}
        for subject_idx in range(self.total_subject):
            temp[subject_idx] = {'x': data[sub_index == subject_idx],
                                 'y': sub_state[sub_index == subject_idx]}
        return temp

    def get_items(self, subject_ids: List, select_ch_names: List) -> (np.array, np.array):
        # 1. Concatenated Datasets
        total_x, total_y = [], []
        for idx in subject_ids:
            x, y = self.data[idx]['x'], self.data[idx]['y']
            total_x.append(x)
            total_y.append(y)
        total_x, total_y = np.concatenate(total_x, axis=0), np.concatenate(total_y, axis=0)

        # 2. EEG Preprocessing
        total_x = mne.EpochsArray(total_x, info=self.info)
        total_x.drop_channels([ch_name
                               for ch_name in self.info.ch_names
                               if ch_name not in select_ch_names])
        total_x = self.preprocessing(total_x)
        return total_x, total_y

    def preprocessing(self, raw: mne.EpochsArray) -> np.array:
        # 1. Band Pass Filter
        # raw = raw.filter(l_freq=self.l_freq, h_freq=self.h_freq)

        # 2. Resampling
        # raw = raw.resample(self.rfreq)
        return raw.get_data()


# 2. Self-Supervised Learning Dataset
class SSL_Dataset(object):
    def __init__(self):
        super().__init__()
        self.info = None
        self.select_ch_names = []

        self.second = second
        self.l_freq, self.h_freq = 1, 50
        self.rfreq = rfreq
        self.set_select_ch_names()

    @abc.abstractmethod
    def load_mat(self, path):
        pass

    @abc.abstractmethod
    def set_select_ch_names(self):
        pass

    @staticmethod
    def sliding_window(elements: np.array, window_size: int, step: int):
        temp = []
        if elements.shape[-1] <= window_size:
            return elements
        for i in range(0, elements.shape[-1] - window_size + 1, step):
            temp.append(elements[:, i:i + window_size])
        temp = np.array(temp)
        return temp

    def preprocessing(self, raw: mne.EpochsArray) -> np.array:
        # 1. Band Pass Filter
        raw = raw.filter(l_freq=self.l_freq, h_freq=self.h_freq)

        # 2. Resampling
        raw = raw.resample(self.rfreq)
        return raw.get_data()

    def get_items(self, path: str):
        data = self.load_mat(path)
        sfreq = int(self.info['sfreq'])

        # 1. Sliding Window
        x = self.sliding_window(data, window_size=sfreq * second, step=sfreq)

        # 2. EEG Preprocessing
        x = mne.EpochsArray(x, info=self.info)
        x.drop_channels([ch_name
                         for ch_name in self.info.ch_names
                         if ch_name not in self.select_ch_names])
        x = self.preprocessing(x)
        return x


class MentalAttention(SSL_Dataset):
    # Distinguishing mental attention states of humans via an EEG-based passive BCI using machine learning methods
    # https://www.sciencedirect.com/science/article/pii/S0957417419303926
    def __init__(self):
        super().__init__()
        self.info = mne.create_info(
            ch_names=['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'],
            sfreq=200,
            ch_types='eeg')

    def load_mat(self, path):
        mat = sio.loadmat(path)
        eeg = mat['o']['data'][0][0][:, 3:17]
        eeg = np.swapaxes(eeg, axis1=1, axis2=0)
        return eeg

    def set_select_ch_names(self):
        self.select_ch_names = ['AF3', 'F7', 'F3', 'FC5', 'FC6', 'F4', 'F8', 'AF4']
        self.select_ch_names = ['F7', 'F3', 'FC5', 'FC6', 'F4', 'F8', 'AF4']


class RestStateEyeOpenDataset(SSL_Dataset):
    # Prediction of Reaction Time and Vigilance Variability from Spatio-Spectral Features of Resting-State EEG
    # in a Long Sustained Attention Task
    # https://ieeexplore.ieee.org/document/9034192
    def __init__(self):
        super().__init__()
        self.info = mne.create_info(
            ch_names=['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1', 'C1', 'C3', 'C5', 'T7',
                      'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9', 'PO7', 'PO3', 'O1', 'Iz', 'Oz',
                      'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'Afz', 'Fz', 'F2', 'F4', 'F6', 'F8',
                      'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz', 'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2',
                      'P2', 'P4', 'P6', 'P8', 'P10', 'PO8', 'PO4', 'O2'],
            sfreq=256,
            ch_types='eeg')

    def load_mat(self, path):
        mat = sio.loadmat(path)
        mat = mat['dataRest'][:64, ...]
        return mat

    def set_select_ch_names(self):
        self.select_ch_names = ['Fp1', 'AF7', 'AF3', 'F1', 'F3', 'F5', 'F7', 'FT7', 'FC5', 'FC3', 'FC1',
                                'C1', 'C3', 'C5', 'T7', 'TP7', 'CP5', 'CP3', 'CP1', 'P1', 'P3', 'P5', 'P7', 'P9',
                                'PO7', 'PO3', 'O1', 'Iz', 'Oz', 'POz', 'Pz', 'CPz', 'Fpz', 'Fp2', 'AF8', 'AF4', 'Afz',
                                'Fz', 'F2', 'F4', 'F6', 'F8', 'FT8', 'FC6', 'FC4', 'FC2', 'FCz', 'Cz',
                                'C2', 'C4', 'C6', 'T8', 'TP8', 'CP6', 'CP4', 'CP2', 'P2', 'P4', 'P6',
                                'P8', 'P10', 'PO8', 'PO4', 'O2']
