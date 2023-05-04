# Copied from timm
# https://github.com/rwightman/pytorch-image-models

from typing import List, Tuple, Dict, Optional

import math

import torch
from torch.nn import functional as F
from torch import nn, Tensor


def inv_freq_bands(
        num_bands: int,
        temperature: float = 100000.,
        step: int = 2,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
) -> torch.Tensor:
    inv_freq = 1. / (temperature ** (torch.arange(0, num_bands, step, dtype=dtype, device=device) / num_bands))
    return inv_freq


def build_sincos2d_pos_embed(
        feat_shape: List[int],
        dim: int = 64,
        temperature: float = 10000.,
        reverse_coord: bool = False,
        interleave_sin_cos: bool = False,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None
) -> torch.Tensor:
    """
    Args:
        feat_shape:
        dim:
        temperature:
        reverse_coord: stack grid order W, H instead of H, W
        interleave_sin_cos: sin, cos, sin, cos stack instead of sin, sin, cos, cos
        dtype:
        device:
    Returns:
    """
    assert dim % 4 == 0, 'Embed dimension must be divisible by 4 for sin-cos 2D position embedding'
    pos_dim = dim // 4
    bands = inv_freq_bands(pos_dim, temperature=temperature, step=1, dtype=dtype, device=device)

    if reverse_coord:
        feat_shape = feat_shape[::-1]  # stack W, H instead of H, W
    grid = torch.stack(
        torch.meshgrid(
            [torch.arange(s, device=device, dtype=dtype) for s in feat_shape],
            indexing='ij')).flatten(1).transpose(0, 1)
    pos2 = grid.unsqueeze(-1) * bands.unsqueeze(0)
    # FIXME add support for unflattened spatial dim?

    stack_dim = 2 if interleave_sin_cos else 1  # stack sin, cos, sin, cos  instead of sin sin cos cos
    pos_emb = torch.stack([torch.sin(pos2), torch.cos(pos2)], dim=stack_dim).flatten(1)
    return pos_emb # [H * W, 4 * pos_dim]


class CPE(nn.Module):
    """
    Conditional Positional Encoding
    Local Smoothing
    """
    def __init__(self, dim, kernel_size=3, dropout=0.):
        super(CPE, self).__init__()

        self.dwconv = nn.Conv2d(dim, dim, kernel_size, 1, kernel_size//2, groups=dim)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, feat_size: List[int]):
        L, B, C = x.shape
        x = x.permute(1, 2, 0).contiguous().view(B, C, feat_size[0], feat_size[1]) # [seq_len, batch, dim] --> [batch, dim, seq_len]
        x = x + self.dwconv(x)
        x = self.dropout(x)
        x = x.flatten(2).permute(2, 0, 1) # restore back
        return x


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask: Optional[Tensor] = None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t.div(2, rounding_mode='floor')) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2) # [bsz, num_pos_feats*2, H, W]
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
