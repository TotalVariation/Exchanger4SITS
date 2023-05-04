from typing import List, Tuple, Dict, Optional

import math
from collections import defaultdict
import glob
import datetime as dt
import json
import os
import pickle
import zarr

import numpy as np
import pandas as pd
import geopandas as gpd

import torch
from torch.utils import data


class PASTISPixelSetReader(data.Dataset):
    def __init__(
        self,
        config,
        nfolds=[1, 2, 3],
        reference_date='2018-09-01',
        mode='train',
        **kwargs
    ):
        super(PASTISPixelSetReader, self).__init__()

        root = config.DATASET.ROOT
        self.modality = config.DATASET.MODALITY
        self.ignore_index = config.LOSS.IGNORE_INDEX
        self.data_folders = [os.path.join(root, f'DATA_{s}') for s in self.modality]
        self.reference_date = dt.datetime(*map(int, reference_date.split("-")))
        self.max_val = 65535
        self.temp_drop_rate = config.DATASET.TEMP_DROP_RATE
        self.nbins = config.DATASET.NBINS
        self.z_norm = config.DATASET.Z_NORM

        self.metadata = pd.read_csv(os.path.join(root, "metadata_parcel.csv"))
        self.geo_df = gpd.read_file(os.path.join(root, "metadata.geojson"))
        self.geo_df = self.geo_df.set_index('ID_PATCH')

        if nfolds is not None:
            self.metadata = self.metadata[self.metadata.Fold.isin(nfolds)]

        self.stats = {}
        for s in self.modality:
            stats_pth = os.path.join(root, f'NORM_PARCEL_{s}_set.json')
            stats_df = pd.read_json(stats_pth)
            selected_folds = nfolds if nfolds is not None else list(range(1, 6))
            mean = np.stack(
                [stats_df[f'Fold_{f}']['mean'] for f in selected_folds]
            ).astype(np.float32).mean(axis=0)[None, :, None]
            std = np.stack(
                [stats_df[f'Fold_{f}']['std'] for f in selected_folds]
            ).astype(np.float32).mean(axis=0)[None, :, None]
            self.stats[s] = np.stack([mean, std])

        self.mode = mode

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):

        row = self.metadata.iloc[index]

        sample = {}
        for s, folder, tdr in zip(self.modality, self.data_folders, self.temp_drop_rate):
            data = zarr.load(os.path.join(folder, f'{s}_{int(row.ID_PARCEL)}.zarr'))
            date_positions = self.prepare_dates(
                self.geo_df.loc[int(row.ID_PATCH)][f'dates-{s}']
            )

            data = np.clip(data, 0, self.max_val).astype(np.float32)

            # random sample timesteps
            T, C, N = data.shape

            if self.mode != 'test':
                ntimesteps = int(T * (1 - np.random.uniform(low=tdr[0], high=tdr[1])))
                time_idx = sorted(np.random.choice(T, ntimesteps, replace=False))
                data = data[time_idx, ...]
                date_positions = date_positions[time_idx]

            if self.z_norm:
                data = data - self.stats[s][0]
                data = data / self.stats[s][1]
            else:
                data = data / self.max_val

            bin_size = N // self.nbins if N >= self.nbins else 1
            num_pixels = bin_size * self.nbins
            pixel_idx = np.random.choice(N, size=num_pixels, replace=False if N >= num_pixels else True)
            data = data[:, :, pixel_idx]
            data = data.reshape(len(date_positions), C, self.nbins, bin_size).mean(axis=-1)

            sample[f'data_{s}'] = torch.from_numpy(data).transpose(0, 1) # --> [C, T, H, W] for pad_collate
            sample[f'date_positions_{s}'] = torch.from_numpy(date_positions + 1)[None, :]
            sample[f'gdd_{s}'] = torch.ones_like(sample[f'date_positions_{s}']) # dummy

        sample['label'] = torch.tensor(int(row.Label) - 1) # start from 0
        sample['field_id'] = torch.tensor(int(row.ID_PARCEL))

        return sample

    def prepare_dates(self, date_dict):
        d = pd.DataFrame().from_dict(date_dict, orient='index')
        d = d[0].apply(
            lambda x: (
                dt.datetime(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:]))
                - self.reference_date
            ).days
        )
        return d.values

