import collections

import numpy as np

import torch

import torchvision.transforms as T
import torchvision.transforms.functional as TF


from utils import to_2tuple


def pad_if_smaller(data, size, fill=0):
    h, w = data.shape[-2:]
    if h < size[0] or w < size[1]:
        pad_h = size[0] - h if h < size[0] else 0
        pad_w = size[1] - w if w < size[1] else 0
        data = TF.pad(data, (0, 0, pad_w, pad_h), fill=fill)

    return data

