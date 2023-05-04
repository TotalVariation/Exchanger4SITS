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


class PASTISReader(data.Dataset):
    def __init__(
        self,
        config,
        nfolds=[1, 2, 3],
        reference_date='2018-09-01',
        mode='train',
        **kwargs
    ):
        super(PASTISReader, self).__init__()

        root = config.DATASET.ROOT
        self.modality = config.DATASET.MODALITY
        self.task_type = config.DATASET.TASK_TYPE
        self.ignore_index = config.LOSS.IGNORE_INDEX
        """
        The void label is reserved for out-of-scope parcels, either because their crop type
        is not in nomenclature or their overlap with the selected square patch is too small.
        """
        self.void_label = 19
        self.data_folders = [os.path.join(root, f'DATA_{s}') for s in self.modality]
        self.semantic_label_folder = os.path.join(root, 'ANNOTATIONS')
        self.instance_label_folder = os.path.join(root, 'INSTANCE_ANNOTATIONS')

        self.max_val = 65535
        self.reference_date = dt.datetime(*map(int, reference_date.split("-")))
        self.temp_drop_rate = config.DATASET.TEMP_DROP_RATE
        self.random_crop = config.TRAIN.RANDOM_CROP
        self.crop_size = config.TRAIN.CROP_SIZE
        self.z_norm = config.DATASET.Z_NORM

        self.metadata = gpd.read_file(os.path.join(root, "metadata.geojson"))

        if nfolds is not None:
            self.metadata = self.metadata[self.metadata.Fold.isin(nfolds)]

        self.stats = {}
        for s in self.modality:
            stats_pth = os.path.join(root, f'NORM_{s}_patch.json')
            stats_df = pd.read_json(stats_pth)
            selected_folds = nfolds if nfolds is not None else list(range(1, 6))
            mean = np.stack(
                [stats_df[f'Fold_{f}']['mean'] for f in selected_folds]
            ).astype(np.float32).mean(axis=0)[None, :, None, None]
            std = np.stack(
                [stats_df[f'Fold_{f}']['std'] for f in selected_folds]
            ).astype(np.float32).mean(axis=0)[None, :, None, None]
            self.stats[s] = np.stack([mean, std])

        self.mode = mode

    def __len__(self):
        return len(self.metadata)

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

        row = self.metadata.iloc[index]

        sample = {}
        for s, folder, tdr in zip(self.modality, self.data_folders, self.temp_drop_rate):
            data = zarr.load(os.path.join(folder, f'{s}_{row.id}.zarr'))
            date_positions = self.prepare_dates(row[f'dates-{s}'])

            data = np.clip(data, 0, self.max_val).astype(np.float32)

            # random sample timesteps
            T = data.shape[0]

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

            sample[f'data_{s}'] = torch.from_numpy(data).transpose(0, 1) # --> [C, T, H, W] for pad_collate
            sample[f'date_positions_{s}'] = torch.from_numpy(date_positions + 1)[None, :]

        if self.task_type == 'sem_seg':
            pixel_semantic_annotation = zarr.load(os.path.join(
                self.semantic_label_folder, f'TARGET_{row.id}.zarr'))[0].astype(np.int32)
            pixel_semantic_annotation[
                pixel_semantic_annotation == self.void_label
            ] = self.ignore_index
            sample['label'] = torch.from_numpy(pixel_semantic_annotation[None])
        elif self.task_type == 'paps_pano': # for panoptic segmentation with PaPs head
            heatmap = zarr.load(os.path.join(
                self.instance_label_folder, f'HEATMAP_{row.id}.zarr'))
            instance_ids = zarr.load(os.path.join(
                self.instance_label_folder, f'INSTANCES_{row.id}.zarr'))
            pixel_to_object_mapping = zarr.load(os.path.join(
                self.instance_label_folder, f'ZONES_{row.id}.zarr'))
            pixel_semantic_annotation = zarr.load(os.path.join(
                self.semantic_label_folder, f'TARGET_{row.id}.zarr'))[0].astype(np.int32)
            pixel_semantic_annotation[
                pixel_semantic_annotation == self.void_label
            ] = self.ignore_index

            size = np.zeros((*instance_ids.shape, 2))
            object_semantic_annotation = np.full(instance_ids.shape, self.ignore_index)
            for instance_id in np.unique(instance_ids):
                # instance_id 0 is for background class (stuff)
                if instance_id != 0:
                    h = (instance_ids == instance_id).any(axis=-1).sum()
                    w = (instance_ids == instance_id).any(axis=-2).sum()
                    size[pixel_to_object_mapping == instance_id] = (h, w)
                    object_semantic_annotation[
                        pixel_to_object_mapping == instance_id
                    ] = pixel_semantic_annotation[pixel_to_object_mapping == instance_id]

            label = np.concatenate(
                [
                    heatmap[:, :, None],  # 0
                    instance_ids[:, :, None],  # 1
                    pixel_to_object_mapping[:, :, None],  # 2
                    size,  # 3-4
                    object_semantic_annotation[:, :, None],  # 5
                    pixel_semantic_annotation[:, :, None],  # 6
                ],
                axis=-1,
            )
            sample['label'] = torch.from_numpy(label).permute(2, 0, 1)
        elif self.task_type == 'maskformer_pano':
            instance_ids = zarr.load(os.path.join(
                self.instance_label_folder, f'INSTANCES_{row.id}.zarr'))
            pixel_semantic_annotation = zarr.load(os.path.join(
                self.semantic_label_folder, f'TARGET_{row.id}.zarr'))[0].astype(np.int32)
            pixel_semantic_annotation[
                pixel_semantic_annotation == self.void_label
            ] = self.ignore_index

            label = np.concatenate(
                [
                    instance_ids[None, ...],
                    pixel_semantic_annotation[None, ...],
                ],
                axis=0
            )
            sample['label'] = torch.from_numpy(label)
        else:
            raise NotImplementedError

        if self.random_crop and self.mode != 'test':
            # workaround for CUDA OOM
            img_size = sample[f'data_{self.modality[0]}'].shape[-2:]
            i, j, th, tw = self.get_random_crop_params(img_size, self.crop_size)

            for s in self.modality:
                img = sample[f'data_{s}']
                sample[f'data_{s}'] = img[..., i:i+th, j:j+tw]

            label = sample['label']
            sample['label'] = label[:, i:i+th, j:j+tw]

        sample['field_id'] = torch.tensor(row.ID_PATCH)

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

