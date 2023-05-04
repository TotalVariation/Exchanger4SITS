"""
Mostly Copy-Paste from https://github.com/facebookresearch/Mask2Former
"""

from typing import Callable, List, Tuple, Dict, Optional, Union

import math

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from timm.models.layers import get_act_layer
from detectron2.layers import get_norm


class FPN(nn.Module):
    def __init__(
        self,
        feature_channels: List[int],
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
        act_type: str = 'relu',
        maskformer_num_feature_levels: int = 3,
        **kwargs
    ):
        super(FPN, self).__init__()

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_chans in enumerate(feature_channels):
            if idx == len(feature_channels) - 1:
                output_conv = nn.Sequential(
                    nn.Conv2d(in_chans, conv_dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    get_norm(norm, conv_dim),
                    get_act_layer(act_type)()
                )
                self.add_module(f'layer_{idx + 1}', output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                l_conv = nn.Sequential(
                    nn.Conv2d(in_chans, conv_dim, kernel_size=1, stride=1, padding=0, bias=use_bias),
                    get_norm(norm, conv_dim),
                )
                output_conv = nn.Sequential(
                    nn.Conv2d(conv_dim, conv_dim, kernel_size=3, stride=1, padding=1, bias=use_bias),
                    get_norm(norm, conv_dim),
                    get_act_layer(act_type)()
                )
                self.add_module(f'adapter_{idx + 1}', l_conv)
                self.add_module(f'layer_{idx + 1}', output_conv)

                lateral_convs.append(l_conv)
                output_convs.append(output_conv)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_features = nn.Conv2d(conv_dim, mask_dim, kernel_size=3, stride=1, padding=1)

        self.mask_dim = mask_dim
        self.maskformer_num_feature_levels = maskformer_num_feature_levels # 3

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, features: List[Tensor]):
        multi_scale_features = []
        num_cur_levels = 0

        # Reverse feature maps into top-down order (from low to high resolution)
        features = features[::-1]

        for idx, feat in enumerate(features):
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                y = output_conv(feat)
            else:
                cur_fpn = lateral_conv(feat)
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode='nearest')
                y = output_conv(y)
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1

        return self.mask_features(y), multi_scale_features
