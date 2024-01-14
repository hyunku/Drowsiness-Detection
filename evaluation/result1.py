# -*- coding:utf-8 -*-
import os
import torch
import argparse
import pandas as pd
from sklearn.metrics import classification_report


def get_args():
    model_name = 'cnn_lstm'
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path',
                        default=os.path.join('..', 'ckpt', 'supervised', model_name))
    return parser.parse_args()


def result1(args):
    for subject_idx in range(1, 12):
        base_path = os.path.join(args.base_path, 'subject_{0:02d}'.format(subject_idx))
        log_df = pd.read_csv(os.path.join(base_path, 'result.csv'))
        epoch = log_df[log_df['accuracy'] == log_df['accuracy'].max()]['epoch'].values[0]
        model_path = os.path.join(base_path, 'model', '{0:04d}.pth'.format(epoch))

        ckpt = torch.load(model_path, map_location='cpu')['result']
        pred, real = ckpt['pred'], ckpt['real']
        print(classification_report(y_true=real, y_pred=pred,
                                    target_names=['Focused', 'Drowsy'], digits=4))


if __name__ == '__main__':
    result1(get_args())
