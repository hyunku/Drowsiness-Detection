# -*- coding:utf-8 -*-
import os
import librosa
import numpy as np
from librosa import util
from dataset.data_parser import MentalAttention, RestStateEyeOpenDataset, DrowsinessDataset
from scipy.signal import butter, lfilter, stft


def butter_bandpass_filter(signal, low_cut, high_cut, fs, order=5):
    if low_cut == 0:
        low_cut = 0.5
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, signal, axis=-1)
    return y


def get_path2parser(base_path):
    # # 1. Mental Attention Dataset
    parser = []
    # path = os.path.join(base_path, 'Mental-Attention')
    # parser.extend([(os.path.join(path, p), MentalAttention) for p in os.listdir(path)])

    # 2. Resting State EEG Dataset
    path = os.path.join(base_path, 'SPIS-Resting-State-Dataset', 'EO')
    parser.extend([(os.path.join(path, p), RestStateEyeOpenDataset) for p in os.listdir(path)])
    return parser


def normalize_mel(S):
    min_level_db = -100
    return np.clip((S-min_level_db)/-min_level_db, 0, 1)


def stft_db(data, sf, window=2.0, step=0.2, band=(1, 30), power=True, normalize=True):
    # Safety check
    data = np.asarray(data)
    assert step <= window
    step = 1 / sf if step == 0 else step

    # Define STFT parameters
    nperseg = int(window * sf)
    noverlap = int(nperseg - (step * sf))

    # Compute STFT and remove the last epoch
    f, t, sxx = stft(
        data, sf, nperseg=nperseg, noverlap=noverlap, detrend=False, padded=True
    )

    # Let's keep only the frequency of interest
    if band is not None:
        idx_band = np.logical_and(f >= band[0], f <= band[1])
        sxx = sxx[idx_band, :]

    sxx = np.abs(sxx)
    if power:
        sxx = librosa.power_to_db(sxx, ref=np.max)
    if normalize:
        sxx = librosa.util.normalize(sxx)
    return sxx


if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    from torchvision.transforms import Resize, Normalize

    # dd = DrowsinessDataset(path='/home/brainlab/Database/Drowsiness-Dataset/dataset.mat')
    # x, _ = dd.get_items(subject_ids=[1], select_ch_names=['F7'])
    # x = x[1, 0, :]
    # x = stft_db(x, sf=128, window=0.5, step=0.01, band=(0, 50), power=True, normalize=True)
    # print(x)
    # print(x.shape)
    # plt.imshow(x)
    # plt.show()

    p = '/home/brainlab/Database/SPIS-Resting-State-Dataset/EO/S02_restingPre_EO.mat'
    x = RestStateEyeOpenDataset().get_items(p)
    x = x[11, 0, :]
    x = stft_db(x, sf=128, window=0.5, step=0.01, band=(0, 50), power=True, normalize=True)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(dim=0)
    print(x)
    # normalize_inst = Normalize(std=0.5, mean=0.5)
    # x = normalize_inst(x)
    # print(x)
    x = x.squeeze().numpy()
    plt.imshow(x)
    plt.show()
