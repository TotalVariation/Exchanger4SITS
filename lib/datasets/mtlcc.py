"""
Adapted from https://github.com/michaeltrs/DeepSatModels
by X. Cai
"""

from typing import List, Tuple, Dict, Optional

import os
import math
import glob
import datetime as dt
import json
import pickle

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils import data


original_label_dict = {0: "unknown", 1: "sugar_beet", 2: "summer_oat", 3: "meadow", 5: "rape", 8: "hop",
                       9: "winter_spelt", 12: "winter_triticale", 13: "beans", 15: "peas", 16: "potatoes",
                       17: "soybeans", 19: "asparagus", 22: "winter_wheat", 23: "winter_barley", 24: "winter_rye",
                       25: "summer_barley", 26: "maize"}
remap_label_dict = {0: 17,  1: 0,  2: 1,  3: 2,  5: 3,  8: 4,  9: 5, 12: 6, 13: 7, 15: 8,
                    16: 9, 17: 10, 19: 11, 22: 12, 23: 13, 24: 14, 25: 15, 26: 16}


BANDS = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12']


class MTLCCReader(data.Dataset):
    def __init__(
        self,
        config,
        year=16,
        mode='train',
        **kwargs
    ):
        super(MTLCCReader, self).__init__()

        self.data_rootdir = os.path.join(os.path.expanduser('~'), 'data')
        self.root = config.DATASET.ROOT
        self.modality = config.DATASET.MODALITY[0]
        self.task_type = config.DATASET.TASK_TYPE
        self.temp_drop_rate = config.DATASET.TEMP_DROP_RATE[0]
        self.ignore_index = config.LOSS.IGNORE_INDEX
        self.void_label = 17 # 18 ignore unknown
        self.max_val = 65535
        self.data_paths = pd.read_csv(os.path.join(self.root, f'{mode}_paths.csv'))
        self.label_remap = remap_label_dict
        self.z_norm = config.DATASET.Z_NORM
        self.random_crop = config.TRAIN.RANDOM_CROP
        self.crop_size = config.TRAIN.CROP_SIZE
        self.mode = mode

        stats_pth = os.path.join(self.root, 'stats.pkl')
        with open(stats_pth, 'rb') as handle:
            stats = pickle.load(handle)
        self.mean = stats[:, 0][None, :, None, None]
        self.std = stats[:, 1][None, :, None, None]

    def __len__(self):
        return len(self.data_paths)

    @staticmethod
    def get_random_crop_params(
        img_size: Tuple[int, int], output_size: Tuple[int, int],
    ) -> Tuple[int, int, int, int]:

        h, w = img_size
        th, tw = output_size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()
        return i, j, th, tw

    def __getitem__(self, index):

        data_path = os.path.join(self.data_rootdir, self.data_paths.iloc[index].paths)

        with open(data_path, 'rb') as handle:
            sample = pickle.load(handle, encoding='latin1')

        data = self.upsample_bands(sample) # --> [T, C, H, W] for pad_collate
        date_positions = torch.from_numpy(sample['day'] + 1)
        T = data.shape[0]
        if self.mode != 'test':
            tdr = np.random.uniform(low=self.temp_drop_rate[0], high=self.temp_drop_rate[1])
            ntimesteps = int(T * (1 - tdr))
            time_idx = sorted(np.random.choice(T, ntimesteps, replace=False))
            data = data[time_idx, ...]
            date_positions = date_positions[time_idx]

        # normalize
        if self.z_norm:
            data = data - self.mean
            data = data / self.std
        else:
            data = data / self.max_val

        pixel_semantic_annotation = self.remap_label(sample)

        if self.random_crop and self.mode != 'test':
            img_size = data.shape[-2:]
            i, j, th, tw = self.get_random_crop_params(img_size, self.crop_size)

            data = data[..., i:i+th, j:j+tw]
            pixel_semantic_annotation = pixel_semantic_annotation[i:i+th, j:j+tw]

        return {f'data_{self.modality}': data.transpose(0, 1),
                f'date_positions_{self.modality}': date_positions[None, :],
                'label': pixel_semantic_annotation[None],
                }

    def upsample_bands(self, sample):

        # upsample to 10m resolution
        x10 = torch.from_numpy(sample['x10']).float().permute(0, 3, 1, 2)
        size = x10.shape[-2:]

        x20 = torch.from_numpy(sample['x20']).float().permute(0, 3, 1, 2)
        x20 = F.interpolate(x20, size=size, mode='bilinear', align_corners=False)

        return torch.cat([x10, x20], dim=1)

    def remap_label(self, sample):
        label = torch.from_numpy(sample['labels'])[0] # [T, H, W]
        new = torch.full_like(label, self.void_label)
        not_remapped = torch.ones_like(label).bool()
        for v in label.unique():
            mask = (label == v) & not_remapped
            new[mask] = self.label_remap[int(v)]
            not_remapped[mask] = False

        new[new == self.void_label] = self.ignore_index

        return new

