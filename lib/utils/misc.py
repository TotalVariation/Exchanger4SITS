# ------------------------------------------------------------------------------
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import List, Optional
import collections
from itertools import repeat

import logging
import json
import copy

import numpy as np
import pandas as pd

import torch
from torch import nn, Tensor
import torchvision


logger = logging.getLogger(__name__)


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return tuple(maxes)


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor] = None):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor], pad_value=0.):
    # TODO make this more general
    # tensor shape: [C, ...]
    # mask: where True indicates the padded positions
    bsz = len(tensor_list)
    max_size = _max_by_axis([list(img.shape) for img in tensor_list])
    batch_shape = (bsz,) + max_size
    max_shape = max_size[1:]
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device
    tensor = torch.full(batch_shape, pad_value, dtype=dtype, device=device)
    mask = torch.ones((bsz,) + max_shape, dtype=torch.bool, device=device)
    for img, pad_img, m in zip(tensor_list, tensor, mask):
        slices = [slice(0, s) for s in img.shape]
        pad_img[slices].copy_(img)
        m[slices[1:]] = False
    return NestedTensor(tensor, mask)


# From PyTorch internals
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def recursive2device(x, device):
    if isinstance(x, Tensor):
        return x.to(device, non_blocking=True)
    elif isinstance(x, collections.abc.Mapping):
        return {k: recursive2device(x[k], device) for k in x}
    elif isinstance(x, collections.abc.Sequence):
        return [recursive2device(elem, device) for elem in x]
    else:
        raise TypeError(f'Support Tensor, Dict, List, but found unrecognized input type: {type(x)}')


# make numpy JSON serializable
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (numpy.int_, numpy.intc, numpy.intp, numpy.int8,
            numpy.int16, numpy.int32, numpy.int64, numpy.uint8,
            numpy.uint16,numpy.uint32, numpy.uint64)):
            return int(obj)
        elif isinstance(obj, (numpy.float_, numpy.float16, numpy.float32,
            numpy.float64)):
            return float(obj)
        elif isinstance(obj,(numpy.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class ScalerScheduler(object):
    def __init__(self, init_val, end_val, scheduler_type):

        self.end_val = end_val
        self.scale = init_val - end_val
        self.scheduler_type = scheduler_type

    def update(self, ratio):

        if self.scheduler_type == 'cos':
            val = self.end_val + self.scale * 0.5 * (1.0 + np.cos(np.pi * ratio))
        else:
            raise NotImplementedError

        return torch.tensor(val)


def add_spectral_norm(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        m = nn.utils.parametrizations.spectral_norm(m)


def remove_spectral_norm(m):
    if nn.utils.parametrize.is_parametrized(m):
        nn.utils.parametrize.remove_parametrizations(m, 'weight')


def remove_weight_norm(m):
    if isinstance(m, (nn.Linear, nn.Conv1d)):
        if hasattr(m, 'weight_g'):
            m = nn.utils.remove_weight_norm(m, name='weight')
