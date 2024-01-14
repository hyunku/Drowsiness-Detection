# -*- coding:utf-8 -*-
import ray
import torch
import random
import numpy as np
from typing import List, Tuple
from dataset.utils import stft_db
from torchvision.transforms import Resize, Normalize
from dataset.data_parser import SSL_Dataset
import warnings

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

random_seed = 777
np.random.seed(random_seed)
torch.manual_seed(random_seed)
random.seed(random_seed)


def batch_dataloader(path2parser_list: List[Tuple[str, SSL_Dataset]], batch_size: int,
                     freqs: int, window: int = 1, step: float = 0.05, bands: Tuple[int, int] = (0, 50),
                     resize: Tuple[int, int] = (50, 56)):
    resize_inst = Resize(size=resize)
    normalize_inst = Normalize(mean=0.5, std=0.5)
    np.random.shuffle(path2parser_list)

    def get_data(path2parse: Tuple) -> (np.array, np.array):
        path, parser = path2parse
        data = parser().get_items(path=path)
        data = np.reshape(data, [-1, data.shape[-1]])
        data = list(data)
        return data

    def convert_tensor(x: List) -> torch.Tensor:
        x = [stft_db(x_, sf=freqs, window=window, step=step, band=bands) for x_ in x]
        x = np.array(x)
        x = torch.tensor(x, dtype=torch.float32)
        x = resize_inst(x)
        x = normalize_inst(x)
        x = x.unsqueeze(dim=1)
        return x

    it = (
        ray.util.iter.from_items(path2parser_list, num_shards=10)
                     .for_each(lambda x_: get_data(x_))
                     .flatten()
                     .local_shuffle(shuffle_buffer_size=2)
                     .batch(batch_size)
                     .for_each(lambda x_: convert_tensor(x_))
    )
    return it
