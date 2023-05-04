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
import zarr

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils import data


remap_label_dict = {
    'labels_20k2k': {0: 20,
                     1: 0, 2: 1, 3: 2,
                     4: 20, 5: 20,
                     6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11,
                     15: 20, 16: 20, 17: 20, 18: 20, 19: 20,
                     20: 12,
                     21: 20, 22: 20, 23: 20, 24: 20,
                     25: 13,
                     26: 20, 27: 20, 28: 20, 29: 20, 30: 20, 31: 20, 32: 20,
                     33: 14, 34: 15, 35: 16, 36: 17,
                     37: 20, 38: 20, 39: 20, 40: 20, 41: 20, 42: 20, 43: 20, 44: 20, 45: 20, 46: 20, 47: 20,
                     48: 18,
                     49: 20, 50: 20, 51: 20, 52: 20, 53: 20, 54: 20, 55: 20, 56: 20, 57: 20, 58: 20, 59: 20,
                     60: 20,
                     61: 20, 62: 20, 63: 20, 64: 20, 65: 20, 66: 20, 67: 20, 68: 20, 69: 20, 70: 20, 71: 20,
                     72: 20,
                     73: 19,
                     74: 20, 75: 20, 76: 20, 77: 20, 78: 20, 79: 20, 80: 20, 81: 20, 82: 20, 83: 20, 84: 20,
                     85: 20,
                     86: 20, 87: 20, 88: 20, 89: 20, 90: 20, 91: 20, 92: 20, 93: 20, 94: 20, 95: 20, 96: 20,
                     97: 20,
                     98: 20, 99: 20, 100: 20, 101: 20, 102: 20, 103: 20, 104: 20, 105: 20, 106: 20, 107: 20,
                     108: 20, 109: 20, 110: 20, 111: 20, 112: 20, 113: 20, 114: 20, 115: 20, 116: 20, 117: 20,
                     118: 20, 119: 20, 120: 20, 121: 20, 122: 20, 123: 20, 124: 20, 125: 20, 126: 20, 127: 20,
                     128: 20, 129: 20, 130: 20, 131: 20, 132: 20, 133: 20, 134: 20, 135: 20, 136: 20, 137: 20,
                     138: 20, 139: 20, 140: 20, 141: 20, 142: 20, 143: 20, 144: 20, 145: 20, 146: 20, 147: 20,
                     148: 20, 149: 20, 150: 20, 151: 20, 152: 20, 153: 20, 154: 20, 155: 20, 156: 20, 157: 20,
                     158: 20, 159: 20, 160: 20, 161: 20, 162: 20, 163: 20, 164: 20, 165: 20, 166: 20, 167: 20
                     }
                    }


class T31TFMPixelSetReader(data.Dataset):
    def __init__(
        self,
        config,
        mode='train',
        **kwargs):
        super(T31TFMPixelSetReader, self).__init__()

        self.root = config.DATASET.ROOT
        self.modality = config.DATASET.MODALITY[0]
        self.target_type = config.DATASET.TASK_TYPE
        self.temp_drop_rate = config.DATASET.TEMP_DROP_RATE[0]
        self.z_norm = config.DATASET.Z_NORM
        self.ignore_index = config.LOSS.IGNORE_INDEX
        self.void_label = 20
        self.max_val = 65535
        self.data_paths = pd.read_csv(
            os.path.join(self.root, 'paths', f'pixelset_{mode}_paths.csv'),
            converters={'doy': lambda x: list(map(int, x.strip('[]').split(',')))}
        )
        self.label_remap = remap_label_dict['labels_20k2k']
        self.nbins = config.DATASET.NBINS
        self.mode = mode

        stats_pth = os.path.join(self.root, 'stats.pkl')
        with open(stats_pth, 'rb') as f:
            stats = pickle.load(f)
        self.mean = stats[:, 0][None, :, None]
        self.std = stats[:, 1][None, :, None]

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, index):

        row = self.data_paths.iloc[index]
        data_path = os.path.join(self.root, row.paths)
        data = zarr.load(data_path)
        date_positions = np.array(row.doy)

        T, C, N = data.shape
        data = np.clip(data, 0, self.max_val).astype(np.float32)

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

        bin_size = N // self.nbins if N >= self.nbins else 1
        num_pixels = bin_size * self.nbins
        pixel_idx = np.random.choice(N, size=num_pixels, replace=False if N >= num_pixels else True)
        data = data[:, :, pixel_idx]
        data = data.reshape(len(date_positions), C, self.nbins, bin_size).mean(axis=-1)

        label = self.label_remap[int(row.label)]
        if label == self.void_label:
            label = self.ignore_index

        return {f'data_{self.modality}': torch.from_numpy(data).transpose(0, 1),
                f'date_positions_{self.modality}': torch.from_numpy(date_positions + 1)[None, :],
                f'gdd_{self.modality}': torch.ones((1, len(date_positions))),
                'label': torch.tensor(label)
                }

