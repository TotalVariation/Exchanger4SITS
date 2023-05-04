import math
from functools import partial
from typing import Tuple, List, Callable, Union

import torch
from torch import nn, Tensor
import torch.utils.checkpoint as checkpoint

from timm.models.layers import DropPath, to_2tuple, to_ntuple, trunc_normal_, get_act_layer

from layers import NormLayer


class MlpWithDepthwiseConv(nn.Module):
    def __init__(
            self, in_features, hidden_features=None, out_features=None, act_type='gelu',
            drop=0., extra_relu=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.relu = nn.ReLU() if extra_relu else nn.Identity()
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = get_act_layer(act_type)()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, feat_size: List[int]):
        x = self.fc1(x)
        B, N, C = x.shape
        x = x.transpose(1, 2).contiguous().view(B, C, feat_size[0], feat_size[1])
        x = self.relu(x)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            sr_ratio=1,
            linear_attn=False,
            pool_size=8,
            qkv_bias=True,
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if not linear_attn:
            self.pool = None
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
            else:
                self.sr = None
                self.norm = None
            self.act = None
        else:
            self.pool = nn.AdaptiveAvgPool2d(pool_size)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

    def forward(self, x, feat_size: List[int]):
        B, N, C = x.shape
        H, W = feat_size
        q = self.q(x).reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3)

        if self.pool is not None:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            if self.sr is not None:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., sr_ratio=1, linear_attn=False, qkv_bias=False,
            drop=0., attn_drop=0., drop_path=0., act_type='gelu', norm_type='layernorm'):
        super().__init__()
        self.norm1 = NormLayer(norm_type, dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            linear_attn=linear_attn,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = NormLayer(norm_type, dim)
        self.mlp = MlpWithDepthwiseConv(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_type=act_type,
            drop=drop,
            extra_relu=linear_attn
        )

    def forward(self, x, feat_size: List[int]):
        x = x + self.drop_path(self.attn(self.norm1(x), feat_size))
        x = x + self.drop_path(self.mlp(self.norm2(x), feat_size))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_type='layernorm'):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        assert max(patch_size) > stride, "Set larger patch_size than stride"
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = NormLayer(norm_type, embed_dim)

    def forward(self, x):
        x = self.proj(x)
        feat_size = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm(x)
        return x, feat_size


class PyramidVisionTransformerStage(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_out: int,
            depth: int,
            downsample: bool = True,
            num_heads: int = 8,
            sr_ratio: int = 1,
            linear_attn: bool = False,
            mlp_ratio: float = 4.0,
            qkv_bias: bool = True,
            drop: float = 0.,
            attn_drop: float = 0.,
            drop_path: Union[List[float], float] = 0.0,
            norm_type: str = 'layernorm',
    ):
        super().__init__()
        self.grad_checkpointing = False

        if downsample:
            self.downsample = OverlapPatchEmbed(
                patch_size=3,
                stride=2,
                in_chans=dim,
                embed_dim=dim_out)
        else:
            assert dim == dim_out
            self.downsample = None

        self.blocks = nn.ModuleList([Block(
            dim=dim_out,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            linear_attn=linear_attn,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_type=norm_type,
        ) for i in range(depth)])

        self.norm = NormLayer(norm_type, dim_out)

    def forward(self, x, feat_size: List[int]) -> Tuple[torch.Tensor, List[int]]:
        if self.downsample is not None:
            x, feat_size = self.downsample(x)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(blk, x, feat_size)
            else:
                x = blk(x, feat_size)
        x = self.norm(x)
        x = x.reshape(x.shape[0], feat_size[0], feat_size[1], -1).permute(0, 3, 1, 2).contiguous()
        return x, feat_size


class PVT2(nn.Module):
    def __init__(self, config, **kwargs):
        super(PVT2, self).__init__()

        spec_dict = {k.lower(): v for k, v in dict(config.PVT2).items()}
        depths = spec_dict['depths']
        embed_dims = spec_dict['embed_dims']
        num_heads = spec_dict['num_heads']
        sr_ratios = spec_dict['sr_ratios']
        mlp_ratios = spec_dict['mlp_ratios']
        qkv_bias = spec_dict['qkv_bias']
        linear = spec_dict['linear']
        drop_rate = spec_dict['drop_rate']
        attn_drop_rate = spec_dict['attn_drop_rate']
        drop_path_rate = spec_dict['drop_path_rate']
        norm_type = spec_dict['norm_type']
        in_chans = kwargs['in_channels']

        self.depths = depths
        num_stages = len(depths)
        mlp_ratios = to_ntuple(num_stages)(mlp_ratios)
        num_heads = to_ntuple(num_stages)(num_heads)
        sr_ratios = to_ntuple(num_stages)(sr_ratios)
        assert(len(embed_dims)) == num_stages

        self.patch_embed = OverlapPatchEmbed(
            patch_size=3,
            stride=1,
            in_chans=in_chans,
            embed_dim=embed_dims[0]
        )

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        cur = 0
        prev_dim = embed_dims[0]
        self.stages = nn.ModuleList()
        for i in range(num_stages):
            self.stages.append(PyramidVisionTransformerStage(
                dim=prev_dim,
                dim_out=embed_dims[i],
                depth=depths[i],
                downsample=i > 0,
                num_heads=num_heads[i],
                sr_ratio=sr_ratios[i],
                mlp_ratio=mlp_ratios[i],
                linear_attn=linear,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_type=norm_type
            ))
            prev_dim = embed_dims[i]
            cur += depths[i]

        self.num_features = embed_dims

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02, a=-0.02, b=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x, feat_size = self.patch_embed(x)
        out = []
        for stage in self.stages:
            x, feat_size = stage(x, feat_size=feat_size)
            out.append(x)
        return out

