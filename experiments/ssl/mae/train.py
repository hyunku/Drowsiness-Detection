# -*- coding:utf-8 -*-
import os
import mne
import shutil
import argparse
import pandas as pd
import torch.optim as opt
import matplotlib.pyplot as plt
from dataset.utils import get_path2parser
from torch.utils.tensorboard import SummaryWriter
from experiments.ssl.mae.data_loader import *
from experiments.ssl.mae.model import MaskedAutoEncoderViT, EncoderWrapper
from experiments.ssl.mae.data_loader import batch_dataloader
from models.utils import NativeScalerWithGradNormCount as NativeScaler
from sklearn.metrics import accuracy_score, f1_score


warnings.filterwarnings(action='ignore')


random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

mne.set_log_level(False)
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def get_args():
    parser = argparse.ArgumentParser()
    # Dataset
    sampling_rate = 128
    parser.add_argument('--sampling_rate', default=sampling_rate)
    parser.add_argument('--second', default=3)
    parser.add_argument('--base_path', default=os.path.join('..', '..', '..', '..', '..', '..', 'Database'))

    # Train (for MAE)
    parser.add_argument('--train_epochs', default=150, type=int)
    parser.add_argument('--train_lr_rate', default=0.0005, type=float)
    parser.add_argument('--train_batch_size', default=16, type=int)
    parser.add_argument('--train_batch_accumulation', default=32, type=int)
    parser.add_argument('--train_stft_window', default=0.5, type=int)
    parser.add_argument('--train_stft_step', default=0.01, type=int)
    parser.add_argument('--train_stft_band', default=(0, 50), type=int)

    # Masked Autoencoder Hyperparameter
    parser.add_argument('--input_size', default=(30, 200), type=tuple)
    parser.add_argument('--patch_size', default=5, type=int)
    parser.add_argument('--encoder_embed_dim', default=512, type=int)
    parser.add_argument('--encoder_heads', default=8, type=int)
    parser.add_argument('--encoder_depths', default=10, type=int)

    parser.add_argument('--decoder_embed_dim', default=256, type=int)
    parser.add_argument('--decoder_heads', default=8, type=int)
    parser.add_argument('--decoder_depths', default=10, type=int)
    parser.add_argument('--mask_ratio', default=0.75, type=float)
    parser.add_argument('--norm_pix_loss', default=False, type=bool)

    # Setting Checkpoint Path
    parser.add_argument('--ckpt_path', default=os.path.join('..', '..', '..', 'ckpt',
                                                            'ssl', 'mae'), type=str)
    parser.add_argument('--tensorboard_path', default=os.path.join('..', '..', '..', 'tensorboard',
                                                                   'ssl', 'mae'))
    parser.add_argument('--print_step', default=500, type=int)
    return parser.parse_args()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.model = MaskedAutoEncoderViT(input_size=args.input_size,
                                          patch_size=args.patch_size,
                                          channels=1,
                                          embed_dim=args.encoder_embed_dim,
                                          encoder_heads=args.encoder_heads,
                                          encoder_depths=args.encoder_depths,
                                          decoder_embed_dim=args.decoder_embed_dim,
                                          decoder_heads=args.decoder_heads,
                                          decoder_depths=args.decoder_depths).to(device)
        self.optimizer = opt.AdamW(self.model.parameters(), lr=self.args.train_lr_rate)
        self.loss_scaler = NativeScaler()

        # remote tensorboard files
        if os.path.exists(self.args.tensorboard_path):
            shutil.rmtree(self.args.tensorboard_path)

        self.writer = SummaryWriter(log_dir=self.args.tensorboard_path)
        self.path2parser_list = get_path2parser(base_path=self.args.base_path)

    def get_loss(self, real, pred, mask):
        if self.args.norm_pix_loss:
            mean = real.mean(dim=-1, keepdim=True)
            var = real.var(dim=-1, keepdim=True)
            real = (real - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - real) ** 2
        loss = loss.mean(dim=-1)
        mask = mask.view(loss.shape)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def train(self):
        ray.init(log_to_driver=False, num_cpus=4, num_gpus=2)
        train_dataloader = batch_dataloader(path2parser_list=self.path2parser_list,
                                            batch_size=self.args.train_batch_size,
                                            freqs=self.args.sampling_rate, window=self.args.train_stft_window,
                                            step=self.args.train_stft_step, bands=self.args.train_stft_band,
                                            resize=self.args.input_size)

        # Train (for MAE)
        for epoch in range(self.args.train_epochs):
            self.model.train()
            step = 0
            epoch_train_loss = []
            for batch in train_dataloader.gather_async(num_async=5):
                x = batch.to(device)
                with torch.cuda.amp.autocast():
                    real, pred, mask, latent = self.model(x, mask_ratio=self.args.mask_ratio)
                    loss = self.get_loss(real=real, pred=pred, mask=mask)

                loss = loss / self.args.train_batch_accumulation
                self.loss_scaler(
                    loss=loss, optimizer=self.optimizer, clip_grad=5,
                    parameters=self.model.parameters(), create_graph=False,
                    update_grad=(step + 1) % self.args.train_batch_accumulation == 0
                )

                if (step + 1) % self.args.train_batch_accumulation == 0:
                    self.optimizer.zero_grad()

                torch.cuda.synchronize()

                # Print Console Log.
                if (step + 1) % self.args.print_step == 0:
                    self.show_figure(epoch=epoch, real=real, pred=pred)

                epoch_train_loss.append(float(loss.detach().cpu().item()))
                step += 1

            # Print Console Log.
            epoch_train_loss = np.mean(epoch_train_loss)
            print('[Epoch] : {0:03d} \t [Train Loss] => {1:.6f}'.format(epoch + 1, epoch_train_loss))
            # Save Checkpoint File
            self.save_ckpt(epoch=epoch, train_loss=epoch_train_loss)
        ray.shutdown()

    def save_ckpt(self, epoch, train_loss):
        if not os.path.exists(os.path.join(self.args.ckpt_path)):
            os.makedirs(os.path.join(self.args.ckpt_path))

        ckpt_path = os.path.join(self.args.ckpt_path, '{0:04d}.pth'.format(epoch + 1))
        torch.save({
            'epoch': epoch+1,
            'backbone_name': 'MaskedAutoEncoderViT',
            'backbone_parameter': {
                'input_size': self.args.input_size, 'patch_size': self.args.patch_size, 'channels': 1,
                'embed_dim': self.args.encoder_embed_dim,
                'encoder_heads': self.args.encoder_heads, 'encoder_depths': self.args.encoder_depths,
                'decoder_embed_dim': self.args.decoder_embed_dim,
                'decoder_heads': self.args.decoder_heads, 'decoder_depths': self.args.decoder_depths,
            },
            'short_time_fourier_transform_parameter': {
                'sfreq': self.args.sampling_rate, 'window': self.args.train_stft_window,
                'step': self.args.train_stft_step, 'bands': self.args.train_stft_band,
            },
            'input_size': self.args.input_size,
            'model_state': self.model.state_dict(),
            'hyperparameter': self.args.__dict__,
            'loss': train_loss,
        }, ckpt_path)

    def show_figure(self, epoch: int, real: torch.Tensor, pred: torch.Tensor):
        fig = plt.figure(figsize=(5, 5))
        ax1 = fig.add_subplot(2, 1, 1)
        real = self.model.unpatchify(real).squeeze()
        ax1.set_title('Epoch : {0:02d} - real'.format(epoch))
        ax1.imshow(real[0].detach().cpu().numpy())

        ax2 = fig.add_subplot(2, 1, 2)
        pred = self.model.unpatchify(pred).squeeze()
        ax2.set_title('Epoch : {0:02d} - pred'.format(epoch))
        ax2.imshow(pred[0].detach().cpu().numpy())
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    augments = get_args()
    trainer = Trainer(augments)
    trainer.train()

