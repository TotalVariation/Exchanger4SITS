from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from timm.models.layers import trunc_normal_, get_act_layer

from .normlayers import NormLayer


class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim, norm_type='layernorm', act_type='gelu', bias=False):
        super(LinearLayer, self).__init__()

        self.fc = nn.Linear(in_dim, out_dim, bias=bias)
        self.norm_layer = NormLayer(norm_type, out_dim)
        self.act = get_act_layer(act_type)()

        self._init_weights()

    def _init_weights(self,):
        trunc_normal_(self.fc.weight, std=0.02, a=-0.02, b=0.02)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0.)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None):
        x = self.fc(x)
        x = self.norm_layer(x, mask)
        x = self.act(x)
        return x
