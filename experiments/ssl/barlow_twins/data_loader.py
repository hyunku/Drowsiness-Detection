# -*- coding:utf-8 -*-
import ray
import torch
import random
import numpy as np
from typing import List, Tuple
from dataset.utils import stft_db
from torchvision.transforms import Resize
from dataset.data_parser import SSL_Dataset
from dataset.augmentation import SignalAugmentation as SigAug
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)


def batch_dataloader(path2parser_list: List[Tuple[str, SSL_Dataset]],
                     augmentations: List, batch_size: int,
                     freqs: int, window: int = 1, step: float = 0.05, bands: Tuple[int, int] = (0, 50),
                     resize: Tuple[int, int] = (50, 56)):
    resize_inst = Resize(size=resize)

    augmentation = SigAug(sampling_rate=freqs)
    np.random.shuffle(path2parser_list)

    def get_data(path2parse: Tuple) -> (np.array, np.array):
        path, parser = path2parse
        data = parser().get_items(path=path)
        data = np.reshape(data, [-1, data.shape[-1]])
        data = list(data)
        return data

    def convert_augmentation(x: List) -> (List, List):
        def converter(sample):
            if sample.ndim == 1:
                sample = np.expand_dims(sample, axis=0)
            augmentation_1, augmentation_2 = random.sample(augmentations, 2)
            aug_name_1, aug_prob_1 = augmentation_1
            aug_name_2, aug_prob_2 = augmentation_2
            sample1 = augmentation.process(sample, aug_name=aug_name_1, p=aug_prob_1).squeeze()
            sample2 = augmentation.process(sample, aug_name=aug_name_2, p=aug_prob_2).squeeze()
            return sample1, sample2

        data = [converter(x_) for x_ in x]
        x1 = [t[0] for t in data]
        x2 = [t[1] for t in data]
        return x1, x2

    def convert_tensor(x1: List, x2: List) -> (torch.Tensor, torch.Tensor):
        x1 = np.array([stft_db(x_, sf=freqs, window=window, step=step, band=bands) for x_ in x1])
        x2 = np.array([stft_db(x_, sf=freqs, window=window, step=step, band=bands) for x_ in x2])
        x1 = resize_inst(torch.tensor(x1, dtype=torch.float32))
        x2 = resize_inst(torch.tensor(x2, dtype=torch.float32))
        return x1, x2

    it = (
        ray.util.iter.from_items(path2parser_list, num_shards=10)
                     .for_each(lambda x_: get_data(x_))
                     .flatten()
                     .local_shuffle(shuffle_buffer_size=2)
                     .batch(batch_size)
                     .for_each(lambda x_: convert_augmentation(x_))
                     .for_each(lambda x_: convert_tensor(x_[0], x_[1]))
    )
    return it
