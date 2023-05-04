"""
Refactor the code from https://github.com/jnyborg/tpe
by X.Cai
"""

import os
import math
from collections import defaultdict
import glob
import datetime as dt
import json
import pickle
import zarr
from typing import List

import numpy as np
import pandas as pd

import torch
from torch.utils import data

from .timematch_helper import *


class TimeMatchReader(data.Dataset):
    def __init__(
        self,
        config,
        dataset_splits=None,
        country=None,
        tile=None,
        year=None,
        nfolds=0,
        mode='train',
        **kwargs
    ):
        super(TimeMatchReader, self).__init__()

        root = config.DATASET.ROOT
        year = str(year)
        dataset = '_'.join([country, tile, year])

        self.modality = config.DATASET.MODALITY[0]
        self.max_val = 65535
        self.nbins = config.DATASET.NBINS
        self.temp_drop_rate = config.DATASET.TEMP_DROP_RATE[0]
        self.z_norm = config.DATASET.Z_NORM

        self.classes = sorted(['corn', 'horsebeans', 'meadow', 'spring_barley',
                               'unknown', 'winter_barley', 'winter_rapeseed',
                               'winter_triticale', 'winter_wheat'])
        self.cls2idx = {c: i for i, c in enumerate(self.classes)}

        dataset_path = os.path.join(root, country, tile, year)
        meta_folder = os.path.join(dataset_path, "meta")
        metadata = pickle.load(open(os.path.join(meta_folder, "metadata.pkl"), "rb"))
        dates = metadata["dates"]
        self.date_positions = np.array(self.days_after(metadata["start_date"], dates))

        self.labels = pd.read_json(os.path.join(dataset_path, 'labels.json'))
        self.labels = self.labels[self.labels.crop_name.isin(self.classes)]

        indices = dataset_splits[dataset][nfolds] if dataset_splits is not None else None
        if mode == 'train' and indices is not None:
            self.labels = self.labels[~self.labels.block_idx.isin(indices)]
        elif mode == 'val' and indices is not None:
            self.labels = self.labels[self.labels.block_idx.isin(indices)]
        # for compatibility
        self.labels.rename(columns={"parcel_idx": "field_id"}, inplace=True)

        stats_pth = os.path.join(root, 'stats_splits.pkl')
        with open(stats_pth, 'rb') as f:
            stats = pickle.load(f)
        test_dataset = config.DATASET.TEST_TILES[0]
        stats = stats[test_dataset['country'] + '_' + test_dataset['tile'] + '_' + str(test_dataset['year'])].astype(np.float32)
        self.mean = stats[:, 0][None, :, None]
        self.std = stats[:, 1][None, :, None]

        self.mode = mode

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        row = self.labels.iloc[index]

        data = zarr.load(row.parcel_path)
        date_positions = self.date_positions

        # random sample pixels or timesteps
        T, C, N = data.shape # without spatial geometry

        if self.mode != 'test':
            tdr = np.random.uniform(low=self.temp_drop_rate[0], high=self.temp_drop_rate[1])
            ntimesteps = int(T * (1 - tdr))
            time_idx = sorted(np.random.choice(T, ntimesteps, replace=False))
            data = data[time_idx, :, :]
            date_positions = date_positions[time_idx]

        data = np.clip(data, 0, self.max_val).astype(np.float32)

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

        label = self.cls2idx[str(row.crop_name)]

        gdd = np.array(row.gdd)[date_positions]

        return {f'data_{self.modality}': torch.from_numpy(data).transpose(0, 1), # [C, T, N]
                f'date_positions_{self.modality}': torch.from_numpy(date_positions + 1)[None, :],
                f'gdd_{self.modality}': torch.from_numpy(gdd + 1)[None, :],
                'label': torch.tensor(label),
                'field_id': torch.tensor(row.field_id),
                }

    def days_after(self, start_date, dates):
        def parse(date):
            d = str(date)
            return int(d[:4]), int(d[4:6]), int(d[6:])

        def interval_days(date1, date2):
            return abs((dt.datetime(*parse(date1)) - dt.datetime(*parse(date2))).days)

        date_positions = [interval_days(d, start_date) for d in dates]
        return date_positions

