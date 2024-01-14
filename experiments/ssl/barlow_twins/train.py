# -*- coding:utf-8 -*-
import os
import ray
import mne
import torch
import shutil
import random
import argparse
import numpy as np
from typing import Tuple
from models.utils import LARC
from dataset.utils import get_path2parser
from models.backbone import CNNEncoder2D
from torch.utils.tensorboard import SummaryWriter
from experiments.ssl.barlow_twins.model import BarlowTwins
from experiments.ssl.barlow_twins.data_loader import batch_dataloader


random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mne.set_log_level(False)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def get_args():
    sampling_rate = 128
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--sampling_rate', default=sampling_rate)
    parser.add_argument('--second', default=3)
    parser.add_argument('--base_path', default=os.path.join('..', '..', '..', '..', '..', '..', 'Database'))

    # Train (for Barlow Twins)
    parser.add_argument('--train_epochs', default=300, type=int)
    parser.add_argument('--train_lr_rate', default=0.001, type=float)
    parser.add_argument('--train_weight_decay', default=0, type=float)
    parser.add_argument('--train_batch_size', default=512, type=int)
    parser.add_argument('--train_batch_accumulation', default=4, type=int)

    # Setting Data Augmentation
    parser.add_argument('--augmentations', default=[('random_permutation', 0.85),
                                                    ('random_bandpass_filter', 0.85),
                                                    ('random_temporal_cutout', 0.85),
                                                    ('random_crop', 0.85)])
    parser.add_argument('--sfreq', default=sampling_rate, type=int)
    parser.add_argument('--window', default=0.5, type=int)
    parser.add_argument('--step', default=0.01, type=float)
    parser.add_argument('--bands', default=(0, 50), type=Tuple)
    parser.add_argument('--resize', default=(30, 200), type=Tuple)

    parser.add_argument('--projection_hidden', default=[2048, 2048], type=int)
    parser.add_argument('--projection_size', default=1024, type=int)
    parser.add_argument('--lambd', default=0.005, type=float)

    # Setting Checkpoint Path
    parser.add_argument('--ckpt_path', default=os.path.join('..', '..', '..', 'ckpt',
                                                            'ssl', 'barlow_twins'), type=str)
    parser.add_argument('--tensorboard_path', default=os.path.join('..', '..', '..', 'tensorboard',
                                                                   'ssl', 'barlow_twins'))
    parser.add_argument('--print_step', default=50, type=int)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.backbone = CNNEncoder2D(sampling_rate=args.sampling_rate, input_size=args.resize)
        self.model = BarlowTwins(backbone=self.backbone,
                                 projection_hidden=self.args.projection_hidden,
                                 projection_size=self.args.projection_size,
                                 lambd=self.args.lambd).to(device)
        self.optimizer = LARC(self.model.parameters(),
                              lr=self.args.train_lr_rate,
                              weight_decay=self.args.train_weight_decay)

        # remote tensorboard files
        if os.path.exists(self.args.tensorboard_path):
            shutil.rmtree(self.args.tensorboard_path)

        self.writer = SummaryWriter(log_dir=self.args.tensorboard_path)
        self.path2parser_list = get_path2parser(base_path=self.args.base_path)

    def train(self):
        ray.init(log_to_driver=False, num_cpus=4, num_gpus=2)
        train_dataloader = batch_dataloader(path2parser_list=self.path2parser_list,
                                            batch_size=self.args.train_batch_size,
                                            augmentations=self.args.augmentations,
                                            freqs=self.args.sampling_rate,
                                            window=self.args.window, step=self.args.step, bands=self.args.bands,
                                            resize=self.args.resize)
        scaler = torch.cuda.amp.GradScaler()

        # Train (for Barlow Twins)
        total_step = 0
        for epoch in range(self.args.train_epochs):
            self.model.train()
            step = 0
            epoch_train_loss = []
            for batch in train_dataloader.gather_async(num_async=5):
                x1, x2 = batch
                x1, x2 = x1.to(device), x2.to(device)
                x1, x2 = x1.unsqueeze(dim=1), x2.unsqueeze(dim=1)

                with torch.cuda.amp.autocast():
                    loss = self.model((x1, x2))
                    loss = loss / self.args.train_batch_accumulation

                # Accumulates scaled gradients.
                scaler.scale(loss).backward()
                if (step + 1) % self.args.train_batch_accumulation == 0:
                    scaler.unscale_(optimizer=self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

                    scaler.step(optimizer=self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()

                # Print Console Log.
                if (step + 1) % self.args.print_step == 0:
                    print_loss = np.mean(epoch_train_loss[-self.args.print_step:])
                    print('[Epoch] : {0:03d} \t'
                          '[Step]: {1:06d} \t'
                          '[Train Loss] => {2:.4f}'.format(epoch + 1,
                                                           step + 1,
                                                           print_loss))

                # Writing Tensorboard.
                self.writer.add_scalar('Loss', loss * self.args.train_batch_accumulation, total_step)
                epoch_train_loss.append(float(loss.detach().cpu().item()) *
                                        self.args.train_batch_accumulation)
                step += 1
                total_step += 1

            # Accumulates scaled gradients.
            scaler.unscale_(optimizer=self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)

            scaler.step(optimizer=self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()

            # Print Log & Save Checkpoint Path
            epoch_train_loss = np.mean(epoch_train_loss)
            self.save_ckpt(epoch, epoch_train_loss)

        ray.shutdown()

    def save_ckpt(self, epoch, train_loss):
        if not os.path.exists(os.path.join(self.args.ckpt_path)):
            os.makedirs(os.path.join(self.args.ckpt_path))

        ckpt_path = os.path.join(self.args.ckpt_path, '{0:04d}.pth'.format(epoch+1))
        torch.save({
            'epoch': epoch+1,
            'backbone_name': 'CNNEncoder2D',
            'backbone_parameter': {'sampling_rate': self.args.sampling_rate, 'input_size': self.args.resize},
            'short_time_fourier_transform_parameter': {
                'sfreq': self.args.sfreq, 'window': self.args.window, 'step': self.args.step,
                'bands': self.args.bands,
            },
            'input_size': self.args.resize,
            'model_state': self.model.backbone.state_dict(),
            'hyperparameter': self.args.__dict__,
            'loss': train_loss,
        }, ckpt_path)


if __name__ == '__main__':
    augments = get_args()
    trainer = Trainer(augments)
    trainer.train()
