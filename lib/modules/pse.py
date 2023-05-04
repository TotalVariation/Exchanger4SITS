"""
Mostly Copy-Paste from
Code: https://github.com/VSainteuf/pytorch-psetae
Modified by X.Cai
"""
import math

import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from timm.models.layers import trunc_normal_

from layers import LinearLayer


class PixelSetEncoder(nn.Module):

    def __init__(self, config, **kwargs):
        super(PixelSetEncoder, self).__init__()

        spec_dict = {k.lower(): v for k, v in dict(config.PSE).items()}
        inp_dim = kwargs['input_dim']
        mlp1_dims = spec_dict['mlp1_dims']
        mlp2_dims = spec_dict['mlp2_dims']
        norm_type = spec_dict['norm_type']
        act_type = spec_dict['act_type']

        # Feature Extraction
        layers = []
        for i in range(len(mlp1_dims)):
            layers.append(LinearLayer(
                inp_dim if i == 0 else mlp1_dims[i-1],
                mlp1_dims[i],
                norm_type=norm_type,
                act_type=act_type,
                bias=False,
            ))
        self.mlp1 = nn.ModuleList(layers)

        inter_dim = mlp1_dims[-1] * 2
        # MLP after pooling
        layers = []
        for i in range(len(mlp2_dims)):
            layers.append(LinearLayer(
                inter_dim if i == 0 else mlp2_dims[i-1],
                mlp2_dims[i],
                norm_type=norm_type,
                act_type=act_type,
                bias=False
            ))
        self.mlp2 = nn.ModuleList(layers)

        self.last_dim = mlp2_dims[-1]

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02, a=-0.02, b=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    def forward(
            self,
            x: Tensor,
            img_mask: Tensor,
            seq_mask: Tensor,
    ):

        B, C, T, N = x.shape

        mask = img_mask.contiguous().view(B * T, N)
        x = x.permute(0, 2, 3, 1).contiguous().view(B * T, N, -1)
        for i in range(len(self.mlp1)):
            x = self.mlp1[i](x, mask=mask)
        x = x.transpose(1, 2)

        x = x.contiguous().view(B, T, -1, N)
        # summarize stats over spatial dimension
        x = masked_mean_std(x, ~(img_mask.bool())) # [B, T, C]

        for i in range(len(self.mlp2)):
            x = self.mlp2[i](x, mask=seq_mask)
        x = x.view(B, T, -1)

        return x


def masked_mean_std(x, mask, eps=1e-8):
    """
    Args:
        x shape: [B, T, C, N]
        mask shape: [B, T, N]
    """

    num_elements = mask.sum(dim=-1, keepdim=True)
    mean = (x * mask.unsqueeze(dim=2)).sum(dim=-1) / (num_elements + eps)
    var = torch.square((x - mean.unsqueeze(dim=-1)) * mask.unsqueeze(dim=2)).sum(dim=-1) / (num_elements + eps)
    std = torch.sqrt(var + eps)

    return torch.cat([mean, std], dim=-1)

